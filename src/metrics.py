"""Métricas de clasificación binaria e inferencia estadística.

- Discriminación: ROC AUC y PR AUC (precisión promedio).
- Probabilísticas: log loss, Brier y ECE con contenedores de igual frecuencia.
- Umbral fijo: el umbral que maximiza F1 en validación se aplica a prueba.
- Utilidad de negocio: lift y ganancia acumulada por decil.
- Inferencia: bootstrap percentil, DeLong para dos AUC correlacionadas, McNemar,
  intervalos t entre semillas y corrección de Holm.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from . import config as C

EPS = 1e-7


def ece(y: np.ndarray, p: np.ndarray, bins: int = C.ECE_BINS) -> float:
    """Error de calibración esperado con contenedores de igual frecuencia."""
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(ix) / n * abs(y[ix].mean() - p[ix].mean()) for ix in chunks if len(ix)))


def calibration_points(y: np.ndarray, p: np.ndarray, bins: int = C.ECE_BINS) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p, kind="stable")
    chunks = [ix for ix in np.array_split(order, bins) if len(ix)]
    return np.array([p[ix].mean() for ix in chunks]), np.array([y[ix].mean() for ix in chunks])


def best_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Umbral que maximiza F1 en validación."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], EPS, None)
    return float(thr[int(np.nanargmax(f1))])


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    yhat = (p >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "base_rate": float(y.mean()), "mean_pred": float(p.mean()), "n": int(len(y)),
    }


def lift_gain(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Lift por decil y ganancia acumulada (fracción de clics capturados al contactar el top q %)."""
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    cum = np.cumsum(ys) / max(ys.sum(), 1)
    q = np.linspace(0, 1, 101)
    idx = np.clip((q * len(ys)).astype(int) - 1, 0, len(ys) - 1)
    gain = np.where(q == 0, 0.0, cum[idx])
    dec = np.array_split(ys, n_bins)
    lift = np.array([d.mean() / max(y.mean(), EPS) for d in dec])
    return {"q": q, "gain": gain, "lift_decil": lift}


# ------------------------------------------------------------------ inferencia
def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "roc_auc", n_boot: int = C.N_BOOT,
                 max_rows: int = C.BOOT_MAX_ROWS, seed: int = C.SAMPLE_SEED, alpha: float = C.ALPHA) -> dict:
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        sub = rng.choice(len(y), max_rows, replace=False)
        y, p = y[sub], p[sub]
    f = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
         "log_loss": lambda a, b: log_loss(a, np.clip(b, EPS, 1 - EPS), labels=[0, 1]),
         "brier": brier_score_loss}[metric]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        vals[b] = f(y[ix], p[ix])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "estimate": float(f(y, p)), "lo": float(lo), "hi": float(hi), "n_boot": n_boot,
            "n_rows": int(len(y))}


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Prueba de DeLong para dos AUC medidas sobre las mismas observaciones (Sun y Xu, 2014)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty = _midrank(p[pos]), _midrank(p[neg])
        tz = _midrank(np.concatenate([p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(max(var, 1e-18))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff),
            "se": float(np.sqrt(max(var, 0))), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def mcnemar_test(y: np.ndarray, yhat1: np.ndarray, yhat2: np.ndarray) -> dict:
    """McNemar con corrección de continuidad sobre los aciertos de dos clasificadores."""
    c1, c2 = yhat1 == y, yhat2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    chi2 = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return {"b_solo_1_acierta": b, "c_solo_2_acierta": c, "chi2": float(chi2), "p": float(stats.chi2.sf(chi2, 1))}


def mean_ci(x: np.ndarray, alpha: float = C.ALPHA) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, s = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(1 - alpha / 2, n - 1) * s / np.sqrt(n)) if n > 1 else float("nan")
    return {"media": m, "de": s, "n": n, "lo": m - h, "hi": m + h}


def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return {"diff": float(d.mean()), "t": float("nan"), "p": float("nan"), "n": len(d)}
    t, p = stats.ttest_rel(a, b)
    ci = mean_ci(d)
    return {"diff": ci["media"], "lo": ci["lo"], "hi": ci["hi"], "t": float(t), "p": float(p), "n": len(d)}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
