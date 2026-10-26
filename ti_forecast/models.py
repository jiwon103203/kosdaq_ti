"""방향 예측 모델. SVM · ANN · kNN 은 논문 Sec. 5.5 의 학습·검증 절차이고, 트리 계열 · 계량 모형은 비교용으로 더했다.

SVM  sigmoid 커널(LibSVM 과 같이 coef0 = 0). γ ∈ {2^-15, 2^-13, …, 2^3} × C ∈ {2^-5, 2^-3, …, 2^15} 격자를
     5-fold 교차검증 정확도로 고른 뒤 학습구간 전체로 다시 학습한다.
ANN  입력 10 – 은닉 10 (tanh) – softmax. 학습구간 앞 75% 로 학습하고 뒤 25% 의 검증 손실이 ``patience`` epoch 동안
     줄지 않으면 멈춰 가장 좋았던 가중치를 쓴다. 논문은 Matlab trainscg 지만 여기서는 Adam 이며, 표본이 한 지수뿐이라
     초기값 운을 줄이려고 시드 여러 개의 확률을 평균한다.
kNN  k ∈ {1, …, 10} 을 5-fold 교차검증 정확도로 고른다 (유클리드 거리).

트리 계열 (같은 10개 지표 입력, 하이퍼파라미터는 같은 purged k-fold 정확도로 선택)
rf     랜덤 포레스트 200그루, max_depth ∈ {3, 6, 제한 없음} × min_samples_leaf ∈ {5, 20, 80}
xgb    XGBoost (hist), max_depth ∈ {2, 3, 5} × learning_rate ∈ {0.03, 0.1} × 나무 수 ∈ {50, 100, 200, 400}.
       나무 수는 fold 마다 400그루를 한 번 학습하고 앞쪽 50/100/200그루로 예측해 고른다. ``device="cuda"`` 면 GPU.
lgbm   LightGBM, xgb 와 같은 격자 (num_leaves = 2^max_depth)

계량 모형
logit  로지스틱 회귀 (L2), C ∈ {10^-3, …, 10^2} 를 purged k-fold 로 선택 (3 클래스는 다항 로짓)
arima  기술지표 대신 일간 로그수익률 r 에 ARMA(p, q) + 상수 = 로그가격의 ARIMA(p, 1, q) 를 맞춘다.
       (p, q) ∈ {0, 1, 2}² 를 AIC 로 고르고, 계수를 고정한 채 평가구간 매일 그날까지의 수익률로 s일 누적수익률을
       예측해 레이블 규칙(2 클래스: > 0, 3 클래스: ±δ)으로 방향을 정한다. 입력 윈도우 n 은 쓰지 않는다.

ANN ``backend="torch"`` 는 같은 구조 · 학습 규칙(Adam, minibatch 200, L2 1e-4, 같은 early stopping)을 PyTorch 로
시드 전부를 한 번에(배치 차원으로) 학습한다. CPU 에서도 빠르고 ``device="cuda"`` 면 GPU 를 쓴다. 초기값 난수가 달라
sklearn 결과와 숫자가 똑같지는 않다. SVM ``backend="cuml"`` 은 RAPIDS cuML 의 GPU SVC 로 같은 격자를 탐색한다.

논문과 달리 fold 를 시간 순서의 연속 블록으로 나누고, 레이블 구간 (t, t+s] 가 검증 블록과 겹치는 학습 표본을 버린다
(purged k-fold, ``gap = s - 1``). 무작위 fold 로 나누면 겹치는 s일 레이블 때문에 검증 정확도가 부풀어 과적합된
하이퍼파라미터가 뽑힌다. ANN 의 학습·검증 사이에도 같은 gap 을 둔다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import warnings
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

PAPER_MODELS = ("svm", "ann", "knn")
TREE_MODELS = ("rf", "xgb", "lgbm")
ECON_MODELS = ("logit", "arima")
MODELS = PAPER_MODELS + TREE_MODELS + ECON_MODELS
PRICE_MODELS = ("arima",)  # 기술지표 대신 일간 로그수익률을 입력으로 받는 모델
SVM_GAMMAS = tuple(2.0 ** np.arange(-15, 4, 2))
SVM_CS = tuple(2.0 ** np.arange(-5, 16, 2))
KNN_KS = tuple(range(1, 11))
RF_DEPTHS = (3, 6, None)
RF_LEAVES = (5, 20, 80)
GB_DEPTHS = (2, 3, 5)
GB_LRS = (0.03, 0.1)
GB_TREES = (50, 100, 200, 400)
LOGIT_CS = tuple(10.0 ** np.arange(-3, 3))
ARIMA_ORDERS = tuple((p, q) for p in range(3) for q in range(3))

# 모델 · 설정별로 필요한 선택 패키지 (pip 이름)
OPTIONAL_PACKAGES = {"xgb": ("xgboost", "xgboost"), "lgbm": ("lightgbm", "lightgbm"),
                     "arima": ("statsmodels", "statsmodels"), "torch": ("torch", "torch"),
                     "cuml": ("cuml", "cuml-cu12  (RAPIDS, https://docs.rapids.ai/install)")}


def purged_kfold(n: int, n_splits: int = 5, gap: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """시간 순서 연속 블록 k-fold. 검증 블록 앞뒤 ``gap`` 개의 학습 표본을 버린다."""
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        train = np.r_[np.arange(0, max(a - gap, 0)), np.arange(min(b + gap, n), n)]
        yield train, np.arange(a, b)


def _folds(y: np.ndarray, gap: int, n_splits: int, all_classes: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """학습 fold 에 클래스가 2개 이상(``all_classes`` 면 전부) 있는 fold 만."""
    need = len(np.unique(y)) if all_classes else 2
    return [(tr, va) for tr, va in purged_kfold(len(y), n_splits, gap) if len(np.unique(y[tr])) >= need]


def _constant_if_single_class(y: np.ndarray, n: int):
    classes = np.unique(y)
    return np.full(n, classes[0]) if len(classes) == 1 else None


def check_dependencies(models: Sequence[str], model_kwargs: Dict[str, Dict]) -> None:
    """선택한 모델 · 설정에 필요한 패키지가 없으면 설치 방법과 함께 ValueError."""
    need = [m for m in models if m in OPTIONAL_PACKAGES]
    for m in models:
        backend = model_kwargs.get(m, {}).get("backend")
        if backend in OPTIONAL_PACKAGES:
            need.append(backend)
    missing = [OPTIONAL_PACKAGES[k] for k in dict.fromkeys(need)
               if importlib.util.find_spec(OPTIONAL_PACKAGES[k][0]) is None]
    if missing:
        raise ValueError("필요한 패키지가 없습니다: " + ", ".join(f"pip install {pip}" for _, pip in missing))
    if any(uses_gpu(model_kwargs.get(m, {})) for m in models) and not gpu_available():
        raise ValueError("CUDA GPU 를 찾지 못했습니다 (--device cpu, --svm-backend sklearn 으로 실행하세요)")


def uses_gpu(kwargs: Dict) -> bool:
    return kwargs.get("device") == "cuda" or kwargs.get("backend") == "cuml"


def gpu_available() -> bool:
    """CUDA GPU 가 보이는지. torch 가 있으면 torch 로, 없으면 nvidia-smi 로 확인한다 (XGBoost 는 GPU 가 없으면
    경고 없이 CPU 로 돌기 때문에 미리 확인한다)."""
    if importlib.util.find_spec("torch") is not None:
        import torch
        return bool(torch.cuda.is_available())
    return shutil.which("nvidia-smi") is not None


# ---------------------------------------------------------------------------
# 논문 모델: SVM · ANN · kNN
# ---------------------------------------------------------------------------

def fit_predict_svm(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                    gammas: Sequence[float] = SVM_GAMMAS, cs: Sequence[float] = SVM_CS,
                    max_iter: int = 1_000_000, backend: str = "sklearn") -> Tuple[np.ndarray, Dict]:
    """sigmoid 커널 SVM. sklearn 은 커널 행렬 tanh(γ x·x') 를 γ 마다 한 번만 만들어 C 와 fold 에 재사용한다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}
    if backend == "cuml":
        from cuml.svm import SVC as CuSVC

        def make(gamma, c):
            return CuSVC(kernel="sigmoid", gamma=gamma, coef0=0.0, C=c, max_iter=max_iter, output_type="numpy")
        return _svm_direct(make, X, y, X_test, gap, n_splits, gammas, cs)
    if backend != "sklearn":
        raise ValueError(f"svm backend must be 'sklearn' or 'cuml', got {backend!r}")
    gram = X @ X.T
    folds = _folds(y, gap, n_splits)
    best = (-np.inf, gammas[0], cs[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gamma in gammas:
            k = np.tanh(gamma * gram)
            blocks = [(k[np.ix_(tr, tr)], k[np.ix_(va, tr)], y[tr], y[va]) for tr, va in folds]
            for c in cs:
                acc = np.mean([np.mean(SVC(kernel="precomputed", C=c, max_iter=max_iter).fit(k_tr, y_tr)
                                       .predict(k_va) == y_va) for k_tr, k_va, y_tr, y_va in blocks])
                if acc > best[0]:
                    best = (acc, gamma, c)
        acc, gamma, c = best
        model = SVC(kernel="precomputed", C=c, max_iter=max_iter).fit(np.tanh(gamma * gram), y)
        pred = model.predict(np.tanh(gamma * (X_test @ X.T)))
    return pred, {"gamma": gamma, "C": c, "cv_acc": acc}


def _svm_direct(make: Callable, X, y, X_test, gap, n_splits, gammas, cs) -> Tuple[np.ndarray, Dict]:
    """커널을 SVC 가 직접 계산하는 격자 탐색 (cuML 처럼 precomputed 커널이 없는 구현용)."""
    folds = _folds(y, gap, n_splits)
    best = (-np.inf, gammas[0], cs[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gamma in gammas:
            for c in cs:
                acc = np.mean([np.mean(np.asarray(make(gamma, c).fit(X[tr], y[tr]).predict(X[va])) == y[va])
                               for tr, va in folds])
                if acc > best[0]:
                    best = (acc, gamma, c)
        acc, gamma, c = best
        pred = np.asarray(make(gamma, c).fit(X, y).predict(X_test)).astype(y.dtype)
    return pred, {"gamma": gamma, "C": c, "cv_acc": acc}


def fit_predict_ann(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, valid_frac: float = 0.25,
                    hidden: int = 10, seeds: Sequence[int] = (0, 1, 2, 3, 4), max_epochs: int = 500,
                    patience: int = 10, lr: float = 1e-3, backend: str = "sklearn",
                    device: str = "cpu") -> Tuple[np.ndarray, Dict]:
    """은닉층 1개 MLP. 시드마다 검증 손실 early stopping 후 테스트 확률을 평균한다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}
    classes = np.unique(y)
    n_val = int(round(len(X) * valid_frac))
    tr, va = slice(0, len(X) - n_val - gap), slice(len(X) - n_val, len(X))
    if backend == "torch":
        return _ann_torch(X, y, X_test, tr, va, classes, hidden, seeds, max_epochs, patience, lr, device)
    if backend != "sklearn":
        raise ValueError(f"ann backend must be 'sklearn' or 'torch', got {backend!r}")
    proba = np.zeros((len(X_test), len(classes)))
    epochs, losses = [], []
    for seed in seeds:
        net = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
                            learning_rate_init=lr, random_state=seed)
        best_loss, best_w, best_epoch, bad = np.inf, None, 0, 0
        for epoch in range(1, max_epochs + 1):
            net.partial_fit(X[tr], y[tr], classes=classes)
            loss = log_loss(y[va], net.predict_proba(X[va]), labels=classes)
            if loss < best_loss - 1e-6:
                best_loss, best_epoch, bad = loss, epoch, 0
                best_w = ([w.copy() for w in net.coefs_], [b.copy() for b in net.intercepts_])
            else:
                bad += 1
                if bad >= patience:
                    break
        net.coefs_, net.intercepts_ = best_w
        proba += net.predict_proba(X_test)
        epochs.append(best_epoch)
        losses.append(best_loss)
    return classes[proba.argmax(axis=1)], {"epochs": float(np.mean(epochs)), "valid_loss": float(np.mean(losses))}


def _ann_torch(X, y, X_test, tr, va, classes, hidden, seeds, max_epochs, patience, lr, device,
               alpha: float = 1e-4, batch_size: int = 200) -> Tuple[np.ndarray, Dict]:
    """시드 S 개의 MLP 를 가중치 텐서 (S, ·, ·) 하나로 묶어 동시에 학습한다.

    sklearn MLPClassifier(adam) 와 같은 규칙: Glorot 균등 초기화(tanh), 매 epoch 섞은 minibatch 200, Adam(lr, 0.9,
    0.999, 1e-8), 배치마다 0.5·alpha·||W||² / 배치크기 L2. 시드마다 따로 early stopping 하고 가장 좋았던 가중치를 쓴다.
    출력은 클래스 수와 같은 softmax (sklearn 은 2 클래스에서 logistic 출력 1개).
    """
    import torch

    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(1)  # 바깥에서 작업을 프로세스 병렬로 돌린다
    yi = np.searchsorted(classes, y)
    k, d, S = len(classes), X.shape[1], len(seeds)

    def tensor(a, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)

    X_tr, y_tr, X_va, y_va, X_te = tensor(X[tr]), tensor(yi[tr], torch.long), tensor(X[va]), tensor(yi[va], torch.long), tensor(X_test)

    def glorot(fan_in, fan_out, gens):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        w = [(torch.rand(fan_in, fan_out, generator=g) * 2 - 1) * bound for g in gens]
        b = [(torch.rand(1, fan_out, generator=g) * 2 - 1) * bound for g in gens]
        return torch.stack(w).to(dev).requires_grad_(), torch.stack(b).to(dev).requires_grad_()

    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    W1, b1 = glorot(d, hidden, gens)
    W2, b2 = glorot(hidden, k, gens)
    params = [W1, b1, W2, b2]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    shuffle = torch.Generator().manual_seed(int(seeds[0]))

    def logits(x, w1, c1, w2, c2):  # x: (S, B, d)
        return torch.baddbmm(c2, torch.tanh(torch.baddbmm(c1, x, w1)), w2)

    def ce(x, yb, w=params):  # 시드별 평균 cross entropy (S,)
        z = logits(x, *w)
        return torch.nn.functional.cross_entropy(z.reshape(-1, k), yb.reshape(-1), reduction="none").view(S, -1).mean(1)

    n_tr, bs = len(X_tr), min(batch_size, len(X_tr))
    best_loss = torch.full((S,), np.inf, device=dev)
    best_w = [p.detach().clone() for p in params]
    best_epoch = torch.zeros(S, device=dev)
    bad = torch.zeros(S, device=dev)
    active = torch.ones(S, dtype=torch.bool, device=dev)
    X_va_s, y_va_s = X_va.expand(S, -1, -1), y_va.expand(S, -1)
    for epoch in range(1, max_epochs + 1):
        perm = torch.argsort(torch.rand(S, n_tr, generator=shuffle), dim=1).to(dev)
        for start in range(0, n_tr, bs):
            idx = perm[:, start:start + bs]
            loss = ce(X_tr[idx], y_tr[idx]) + 0.5 * alpha * ((W1 ** 2).sum((1, 2)) + (W2 ** 2).sum((1, 2))) / idx.shape[1]
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()  # 시드끼리 독립 (합의 기울기 = 시드별 기울기)
            opt.step()
        with torch.no_grad():
            v = ce(X_va_s, y_va_s)
            better = active & (v < best_loss - 1e-6)
            best_loss = torch.where(better, v, best_loss)
            best_epoch = torch.where(better, torch.full_like(best_epoch, epoch), best_epoch)
            for bw, p in zip(best_w, params):
                bw[better] = p[better]
            bad = torch.where(better, torch.zeros_like(bad), bad + 1)
            active &= bad < patience
        if not bool(active.any()):
            break
    with torch.no_grad():
        proba = torch.softmax(logits(X_te.expand(S, -1, -1), *best_w), dim=2).mean(0).cpu().numpy()
    return classes[proba.argmax(axis=1)], {"epochs": float(best_epoch.mean()), "valid_loss": float(best_loss.mean())}


def fit_predict_knn(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                    ks: Sequence[int] = KNN_KS) -> Tuple[np.ndarray, Dict]:
    """kNN. k 를 purged k-fold 정확도로 고른다 (동점이면 작은 k)."""
    folds = list(purged_kfold(len(X), n_splits, gap))
    best = (-np.inf, ks[0])
    for k in ks:
        acc = np.mean([np.mean(KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr]).predict(X[va]) == y[va])
                       for tr, va in folds])
        if acc > best[0]:
            best = (acc, k)
    acc, k = best
    return KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(X_test), {"k": k, "cv_acc": acc}


# ---------------------------------------------------------------------------
# 트리 계열: 랜덤 포레스트 · XGBoost · LightGBM
# ---------------------------------------------------------------------------

def fit_predict_rf(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                   depths: Sequence[Optional[int]] = RF_DEPTHS, leaves: Sequence[int] = RF_LEAVES,
                   n_estimators: int = 200, seed: int = 0, threads: int = 1) -> Tuple[np.ndarray, Dict]:
    """랜덤 포레스트 (max_features = sqrt). max_depth × min_samples_leaf 를 purged k-fold 정확도로 고른다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}

    def make(depth, leaf):
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, min_samples_leaf=leaf,
                                      max_features="sqrt", n_jobs=threads, random_state=seed)

    folds = _folds(y, gap, n_splits)
    best = (-np.inf, depths[0], leaves[0])
    for depth in depths:
        for leaf in leaves:
            acc = np.mean([np.mean(make(depth, leaf).fit(X[tr], y[tr]).predict(X[va]) == y[va]) for tr, va in folds])
            if acc > best[0]:
                best = (acc, depth, leaf)
    acc, depth, leaf = best
    pred = make(depth, leaf).fit(X, y).predict(X_test)
    return pred, {"max_depth": depth or 0, "min_samples_leaf": leaf, "cv_acc": acc}  # max_depth 0 = 제한 없음


def _boosting(make: Callable, predict_at: Callable, X, y, X_test, gap, n_splits, depths, lrs, trees):
    """부스팅 공통: (깊이, 학습률) 마다 fold 에서 최대 나무 수로 한 번 학습하고 앞쪽 나무 수별 정확도를 본다."""
    classes = np.unique(y)
    if len(classes) == 1:
        return np.full(len(X_test), classes[0]), {}
    yi = np.searchsorted(classes, y)  # 0 … K-1
    folds = _folds(yi, gap, n_splits, all_classes=True)
    best = (-np.inf, depths[0], lrs[0], trees[-1])
    for depth in depths:
        for lr in lrs:
            if not folds:
                break
            acc = np.zeros(len(trees))
            for tr, va in folds:
                model = make(depth, lr, max(trees)).fit(X[tr], yi[tr])
                acc += [np.mean(predict_at(model, X[va], t) == yi[va]) for t in trees]
            for t, a in zip(trees, acc / len(folds)):
                if a > best[0]:
                    best = (a, depth, lr, t)
    acc, depth, lr, t = best
    pred = classes[np.asarray(predict_at(make(depth, lr, t).fit(X, yi), X_test, t)).astype(int)]
    return pred, {"max_depth": depth, "learning_rate": lr, "n_estimators": t, "cv_acc": acc}


def fit_predict_xgb(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                    depths: Sequence[int] = GB_DEPTHS, lrs: Sequence[float] = GB_LRS,
                    trees: Sequence[int] = GB_TREES, subsample: float = 0.8, seed: int = 0,
                    device: str = "cpu", threads: int = 1) -> Tuple[np.ndarray, Dict]:
    """XGBoost (hist). ``device="cuda"`` 면 GPU 에서 학습한다."""
    from xgboost import XGBClassifier

    def make(depth, lr, n):
        return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=subsample,
                             tree_method="hist", device=device, n_jobs=threads, random_state=seed, verbosity=0)

    return _boosting(make, lambda m, x, t: m.predict(x, iteration_range=(0, t)), X, y, X_test, gap, n_splits,
                     depths, lrs, trees)


def fit_predict_lgbm(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                     depths: Sequence[int] = GB_DEPTHS, lrs: Sequence[float] = GB_LRS,
                     trees: Sequence[int] = GB_TREES, subsample: float = 0.8, seed: int = 0,
                     threads: int = 1) -> Tuple[np.ndarray, Dict]:
    """LightGBM (num_leaves = 2^max_depth)."""
    from lightgbm import LGBMClassifier

    def make(depth, lr, n):
        return LGBMClassifier(n_estimators=n, max_depth=depth, num_leaves=2 ** depth, learning_rate=lr,
                              subsample=subsample, subsample_freq=1, n_jobs=threads, random_state=seed, verbose=-1)

    return _boosting(make, lambda m, x, t: m.predict(x, num_iteration=t), X, y, X_test, gap, n_splits,
                     depths, lrs, trees)


# ---------------------------------------------------------------------------
# 계량 모형: 로지스틱 회귀 · ARIMA
# ---------------------------------------------------------------------------

def fit_predict_logit(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                      cs: Sequence[float] = LOGIT_CS) -> Tuple[np.ndarray, Dict]:
    """로지스틱 회귀 (L2, 3 클래스는 다항 로짓). C 를 purged k-fold 정확도로 고른다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}
    folds = _folds(y, gap, n_splits)
    best = (-np.inf, cs[0])
    for c in cs:
        acc = np.mean([np.mean(LogisticRegression(C=c, max_iter=1000).fit(X[tr], y[tr]).predict(X[va]) == y[va])
                       for tr, va in folds])
        if acc > best[0]:
            best = (acc, c)
    acc, c = best
    return LogisticRegression(C=c, max_iter=1000).fit(X, y).predict(X_test), {"C": c, "cv_acc": acc}


_ARMA_CACHE: Dict[str, Tuple] = {}


def select_arma(r: np.ndarray, orders: Sequence[Tuple[int, int]] = ARIMA_ORDERS,
                criterion: str = "aic") -> Tuple[int, int, float, np.ndarray, np.ndarray, float]:
    """r 에 ARMA(p, q) + 상수를 최우추정으로 맞추고 정보기준이 가장 작은 차수를 고른다.

    반환 (p, q, 평균 μ, AR 계수, MA 계수, 정보기준). 같은 학습 표본은 프로세스 안에서 다시 맞추지 않는다
    (그리드에서는 예측기간 · 클래스 수만 다른 작업이 같은 표본을 쓴다).
    """
    from statsmodels.tsa.arima.model import ARIMA

    key = hashlib.sha1(np.ascontiguousarray(r).tobytes() + repr((tuple(orders), criterion)).encode()).hexdigest()
    if key in _ARMA_CACHE:
        return _ARMA_CACHE[key]
    best = None
    for p, q in orders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(r, order=(p, 0, q), trend="c").fit()
        ic = float(getattr(res, criterion))
        if np.isfinite(ic) and (best is None or ic < best[-1]):
            mu = float(res.params[res.model.param_names.index("const")])  # statsmodels ARIMA 의 const 는 과정의 평균
            best = (p, q, mu, np.asarray(res.arparams, float), np.asarray(res.maparams, float), ic)
    _ARMA_CACHE[key] = best
    return best


def arma_cum_forecast(r: np.ndarray, mu: float, ar: np.ndarray, ma: np.ndarray, origins: np.ndarray,
                      horizon: int) -> np.ndarray:
    """계수를 고정한 ARMA 로 각 시점 t(``origins``) 까지의 r 만 써서 r_{t+1} + … + r_{t+s} 를 예측한다.

    잔차는 처음 값을 0 으로 둔 조건부 재귀 e = φ(L)/θ(L) (r - μ) 이고, 예측은 t 이후 잔차를 0 으로 두는 표준 재귀.
    """
    from scipy.signal import lfilter

    p, q = len(ar), len(ma)
    z = r - mu
    e = lfilter(np.r_[1.0, -ar], np.r_[1.0, ma], z)
    m = max(p, q)
    zb, eb = np.zeros((len(origins), m + horizon)), np.zeros((len(origins), m + horizon))
    for j in range(m):  # 열 j = 시점 t - (m - 1 - j)
        idx = origins - (m - 1 - j)
        ok = idx >= 0
        zb[ok, j], eb[ok, j] = z[idx[ok]], e[idx[ok]]
    for h in range(horizon):
        col = m + h
        zb[:, col] = sum(ar[i] * zb[:, col - 1 - i] for i in range(p)) + sum(ma[j] * eb[:, col - 1 - j] for j in range(q))
    return zb[:, m:].sum(axis=1) + horizon * mu


def fit_predict_arima(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, horizon: int, delta: float,
                      n_classes: int, positions: Optional[np.ndarray] = None,
                      orders: Sequence[Tuple[int, int]] = ARIMA_ORDERS, criterion: str = "aic") -> Tuple[np.ndarray, Dict]:
    """로그가격 ARIMA(p, 1, q). ``X`` = 학습 일간 로그수익률 (열 1개), ``X_test`` = 학습 끝 다음 날부터 마지막 평가일까지
    모든 거래일의 로그수익률, ``positions`` = 그중 평가일 위치. ``y`` 는 쓰지 않는다."""
    r_fit, path = X[:, 0], X_test[:, 0]
    positions = np.arange(len(path)) if positions is None else np.asarray(positions)
    p, q, mu, ar, ma, ic = select_arma(r_fit, orders, criterion)
    full = np.r_[r_fit, path]
    f = np.expm1(arma_cum_forecast(full, mu, ar, ma, len(r_fit) + positions, horizon))  # s일 단순수익률 예측
    if n_classes == 2:
        pred = np.where(f > 0, 1, -1)
    else:
        pred = np.where(f > delta, 1, np.where(f < -delta, -1, 0))
    return pred, {"p": p, "q": q, criterion: ic, "mu": mu, "forecast_mean": float(f.mean()),
                  "forecast_sd": float(f.std())}


FIT_PREDICT = {"svm": fit_predict_svm, "ann": fit_predict_ann, "knn": fit_predict_knn, "rf": fit_predict_rf,
               "xgb": fit_predict_xgb, "lgbm": fit_predict_lgbm, "logit": fit_predict_logit,
               "arima": fit_predict_arima}


def fit_predict(model: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, **kwargs):
    if model not in FIT_PREDICT:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return FIT_PREDICT[model](X, y, X_test, gap, **kwargs)
