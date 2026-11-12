"""Interpretabilidad local con LIME y contraste con SHAP e importancia por permutación.

Con hashing las columnas del modelo no tienen nombre legible. Por eso los explicadores
trabajan en un espacio reducido e interpretable: las 24 variables categóricas originales,
cada una con sus `LIME_TOP_K_CATEGORIES` categorías más frecuentes en entrenamiento y el
token `__RARE__` para el resto. La función de predicción decodifica cada fila de ese espacio
a texto, aplica el mismo preprocesamiento del modelo y devuelve la probabilidad de clic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from . import config as C


class InterpretableSpace:
    def __init__(self, train: pl.DataFrame, top_k: int = C.LIME_TOP_K_CATEGORIES, cols: list[str] = C.CAT_COLS):
        self.cols = cols
        self.levels = {}
        for c in cols:
            vc = train.group_by(c).len().sort(["len", c], descending=[True, False])
            top = [v for v in vc[c].head(top_k).to_list() if v != C.RARE_TOKEN]
            self.levels[c] = top + [C.RARE_TOKEN]

    @property
    def categorical_names(self) -> dict[int, list[str]]:
        return {j: list(self.levels[c]) for j, c in enumerate(self.cols)}

    def encode(self, df: pl.DataFrame) -> np.ndarray:
        out = np.empty((df.height, len(self.cols)), dtype=np.int32)
        for j, c in enumerate(self.cols):
            lv = self.levels[c]
            m = {v: i for i, v in enumerate(lv)}
            rare = len(lv) - 1
            out[:, j] = np.fromiter((m.get(v, rare) for v in df[c].to_list()), dtype=np.int32, count=df.height)
        return out

    def decode(self, Z: np.ndarray) -> pl.DataFrame:
        Z = np.asarray(Z).round().astype(int)
        return pl.DataFrame({c: np.asarray(self.levels[c], dtype=object)[np.clip(Z[:, j], 0, len(self.levels[c]) - 1)]
                             for j, c in enumerate(self.cols)}, schema={c: pl.String for c in self.cols})


def predict_fn(prep, model, space: InterpretableSpace):
    def f(Z: np.ndarray) -> np.ndarray:
        p1 = model.predict_proba(prep.transform(space.decode(Z)))[:, 1]
        return np.column_stack([1 - p1, p1])
    return f


def select_cases(y: np.ndarray, p: np.ndarray, thr: float, n: int = C.LIME_N_CASES) -> pd.DataFrame:
    """Falsos positivos con mayor probabilidad y falsos negativos con menor probabilidad."""
    idx = np.arange(len(y))
    fp = idx[(y == 0) & (p >= thr)]
    fn = idx[(y == 1) & (p < thr)]
    fp = fp[np.argsort(-p[fp], kind="stable")][:n]
    fn = fn[np.argsort(p[fn], kind="stable")][:n]
    return pd.DataFrame({"fila": np.r_[fp, fn], "tipo": ["FP"] * len(fp) + ["FN"] * len(fn),
                         "y": y[np.r_[fp, fn]], "p": p[np.r_[fp, fn]]})


def make_lime(space: InterpretableSpace, Z_train: np.ndarray, kernel_factor: float, seed: int):
    from lime.lime_tabular import LimeTabularExplainer
    p = len(space.cols)
    return LimeTabularExplainer(Z_train, mode="classification", feature_names=space.cols,
                                categorical_features=list(range(p)), categorical_names=space.categorical_names,
                                class_names=["no clic", "clic"], discretize_continuous=False,
                                kernel_width=kernel_factor * np.sqrt(p), random_state=seed)


def lime_one(explainer, z: np.ndarray, f, num_samples: int) -> dict:
    exp = explainer.explain_instance(z, f, labels=(1,), num_features=len(z), num_samples=num_samples)
    w = dict(exp.as_map()[1])
    return {"pesos": w, "r2": float(exp.score), "pred_local": float(np.ravel(exp.local_pred)[0]),
            "intercepto": float(exp.intercept[1])}


def top_k(w: dict, k: int = C.LIME_TOP_FEATURES) -> set:
    return set(sorted(w, key=lambda j: -abs(w[j]))[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def mean_pairwise_jaccard(sets: list[set]) -> float:
    vals = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else float("nan")


def shap_values(f, Z_bg: np.ndarray, Z_cases: np.ndarray, seed: int = 0) -> np.ndarray:
    import shap
    f1 = lambda Z: f(Z)[:, 1]   # noqa: E731
    masker = shap.maskers.Independent(Z_bg, max_samples=len(Z_bg))
    ex = shap.explainers.Permutation(f1, masker, seed=seed)
    return ex(Z_cases, max_evals=max(500, 2 * Z_cases.shape[1] + 1)).values


def permutation_importance(f, Z: np.ndarray, y: np.ndarray, cols: list[str], n_repeats: int = 5,
                           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, f(Z)[:, 1])
    rows = []
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Zp = Z.copy()
            Zp[:, j] = rng.permutation(Zp[:, j])
            drops.append(base - roc_auc_score(y, f(Zp)[:, 1]))
        rows.append({"variable": c, "caida_auc": float(np.mean(drops)), "de": float(np.std(drops, ddof=1))})
    return pd.DataFrame(rows).sort_values("caida_auc", ascending=False).reset_index(drop=True)
