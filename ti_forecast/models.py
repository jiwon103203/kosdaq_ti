"""SVM · ANN · kNN — 논문 Sec. 5.5 의 학습·검증 절차.

SVM  sigmoid 커널(LibSVM 과 같이 coef0 = 0). γ ∈ {2^-15, 2^-13, …, 2^3} × C ∈ {2^-5, 2^-3, …, 2^15} 격자를
     5-fold 교차검증 정확도로 고른 뒤 학습구간 전체로 다시 학습한다.
ANN  입력 10 – 은닉 10 (tanh) – softmax. 학습구간 앞 75% 로 학습하고 뒤 25% 의 검증 손실이 ``patience`` epoch 동안
     줄지 않으면 멈춰 가장 좋았던 가중치를 쓴다. 논문은 Matlab trainscg 지만 여기서는 Adam 이며, 표본이 한 지수뿐이라
     초기값 운을 줄이려고 시드 여러 개의 확률을 평균한다.
kNN  k ∈ {1, …, 10} 을 5-fold 교차검증 정확도로 고른다 (유클리드 거리).

논문과 달리 fold 를 시간 순서의 연속 블록으로 나누고, 레이블 구간 (t, t+s] 가 검증 블록과 겹치는 학습 표본을 버린다
(purged k-fold, ``gap = s - 1``). 무작위 fold 로 나누면 겹치는 s일 레이블 때문에 검증 정확도가 부풀어 과적합된
하이퍼파라미터가 뽑힌다. ANN 의 학습·검증 사이에도 같은 gap 을 둔다.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterator, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS = ("svm", "ann", "knn")
SVM_GAMMAS = tuple(2.0 ** np.arange(-15, 4, 2))
SVM_CS = tuple(2.0 ** np.arange(-5, 16, 2))
KNN_KS = tuple(range(1, 11))


def purged_kfold(n: int, n_splits: int = 5, gap: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """시간 순서 연속 블록 k-fold. 검증 블록 앞뒤 ``gap`` 개의 학습 표본을 버린다."""
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        train = np.r_[np.arange(0, max(a - gap, 0)), np.arange(min(b + gap, n), n)]
        yield train, np.arange(a, b)


def _constant_if_single_class(y: np.ndarray, n: int):
    classes = np.unique(y)
    return np.full(n, classes[0]) if len(classes) == 1 else None


def fit_predict_svm(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, n_splits: int = 5,
                    gammas: Sequence[float] = SVM_GAMMAS, cs: Sequence[float] = SVM_CS,
                    max_iter: int = 1_000_000) -> Tuple[np.ndarray, Dict]:
    """sigmoid 커널 SVM. 커널 행렬 tanh(γ x·x') 를 γ 마다 한 번만 만들어 C 와 fold 에 재사용한다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}
    gram = X @ X.T
    folds = [(tr, va) for tr, va in purged_kfold(len(X), n_splits, gap) if len(np.unique(y[tr])) > 1]
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


def fit_predict_ann(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, valid_frac: float = 0.25,
                    hidden: int = 10, seeds: Sequence[int] = (0, 1, 2, 3, 4), max_epochs: int = 500,
                    patience: int = 10, lr: float = 1e-3) -> Tuple[np.ndarray, Dict]:
    """은닉층 1개 MLP. 시드마다 검증 손실 early stopping 후 테스트 확률을 평균한다."""
    const = _constant_if_single_class(y, len(X_test))
    if const is not None:
        return const, {}
    classes = np.unique(y)
    n_val = int(round(len(X) * valid_frac))
    tr, va = slice(0, len(X) - n_val - gap), slice(len(X) - n_val, len(X))
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


def fit_predict(model: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, gap: int, **kwargs):
    if model == "svm":
        return fit_predict_svm(X, y, X_test, gap, **kwargs)
    if model == "ann":
        return fit_predict_ann(X, y, X_test, gap, **kwargs)
    if model == "knn":
        return fit_predict_knn(X, y, X_test, gap, **kwargs)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")
