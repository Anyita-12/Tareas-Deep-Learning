"""Funciones de apoyo para los capítulos del informe: tablas resumen, formatos y predicciones."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C, data as D
from .metrics import mean_ci
from .persist import collect

METRIC_LABELS = {"test.roc_auc": "ROC AUC", "test.pr_auc": "PR AUC", "test.log_loss": "Log loss",
                 "test.brier": "Brier", "test.ece": "ECE", "test.accuracy": "Accuracy",
                 "test.precision": "Precision", "test.recall": "Recall", "test.f1": "F1"}
ENV_DIR = {"scikit-learn": "sklearn", "PySpark": "pyspark"}


# Columnas que los cuadernos consultan. Si un experimento no se ejecutó (por ejemplo, por el límite de
# tiempo) o no tiene corridas omitidas, la tabla las trae vacías en lugar de fallar con KeyError.
BASE_COLS = ["env", "exp", "model", "encoding", "size", "n_train", "seed", "cores", "max_iter", "weighted", "crosses",
             "omitido", "motivo", "fit_s_estimado", "n_iter", "run_dir", "val.roc_auc", "test.roc_auc",
             "bench.fit_s", "bench.fit_s_per_iter", "bench.predict_s", "params.hidden_layer_sizes"]


def runs(env: str, exp: str | None = None, include_skipped: bool = False) -> pd.DataFrame:
    df = collect(ENV_DIR.get(env, env), exp)
    for c in BASE_COLS:
        if c not in df:
            df[c] = pd.Series(dtype="float64" if c not in ("env", "exp", "model", "encoding", "size", "motivo",
                                                            "run_dir", "params.hidden_layer_sizes") else "object")
    if df.empty:
        df["omitido"] = df["omitido"].astype(bool)
        return df
    df["omitido"] = df["omitido"].map(lambda v: v is True)
    return df if include_skipped else df[~df["omitido"]].reset_index(drop=True)


SIN_DATOS = ("*Sin datos: este experimento no se ejecutó dentro del límite de tiempo del cómputo "
             "(ver la sección de limitaciones y el anexo de reproducibilidad).*")


def tabla(df: pd.DataFrame, index, nd: int = 4) -> None:
    """Muestra una tabla redondeada e indexada; si está vacía, deja una nota en lugar de fallar."""
    from IPython.display import Markdown, display
    if df is None or len(df) == 0:
        display(Markdown(SIN_DATOS))
        return
    display(df.round(nd).set_index(index))


def label(row) -> str:
    name = C.MODEL_NAMES.get(str(row["model"]).replace("_lbfgs", ""), row["model"])
    if str(row["model"]).endswith("_lbfgs"):
        name += " (L-BFGS)"
    return f"{name} · {C.ENCODING_LABELS.get(row['encoding'], row['encoding'])}"


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["semillas"] = int(g["seed"].nunique()) if "seed" in g else len(g)
        for m in metrics:
            ci = mean_ci(g[m].to_numpy())
            r[f"{m}|media"], r[f"{m}|de"], r[f"{m}|lo"], r[f"{m}|hi"] = ci["media"], ci["de"], ci["lo"], ci["hi"]
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(m: float, s: float, lo: float | None = None, hi: float | None = None, d: int = 4) -> str:
    out = f"{m:.{d}f} ± {s:.{d}f}"
    if lo is not None and np.isfinite(lo):
        out += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return out


def pretty(summary: pd.DataFrame, by: list[str], metrics: list[str], d: int = 4, ci: bool = True) -> pd.DataFrame:
    out = summary[by + (["semillas"] if "semillas" in summary else [])].copy()
    for m in metrics:
        out[METRIC_LABELS.get(m, m)] = [fmt(r[f"{m}|media"], r[f"{m}|de"], r[f"{m}|lo"] if ci else None,
                                            r[f"{m}|hi"] if ci else None, d) for _, r in summary.iterrows()]
    return out


@lru_cache(maxsize=2)
def y_split(which: str) -> np.ndarray:
    if which == "test":
        D.verify_test_frozen()
        df = D.load([C.TEST_DAY], C.TEST_ROWS, columns=[])
    else:
        df = D.load([C.VAL_DAY], C.VAL_ROWS, columns=[])
    return df[C.TARGET].to_numpy().astype(np.int8)


def preds(row: pd.Series, which: str = "test") -> np.ndarray | None:
    """Predicciones guardadas de una corrida (None si no se guardaron)."""
    p = C.RUNS / str(row["run_dir"]) / f"p_{which}.npy"
    return np.load(p) if p.exists() else None


def median_seed(df: pd.DataFrame, metric: str = "test.roc_auc") -> pd.Series:
    g = df.sort_values(metric).reset_index(drop=True)
    return g.iloc[len(g) // 2]


def findings(texto: str, decisiones: dict[str, str] | None = None, titulo: str = "Hallazgos y decisiones") -> None:
    """Muestra la sección de hallazgos con un párrafo y viñetas en negrita (mismo formato del Entregable 1)."""
    from IPython.display import Markdown, display
    md = f"### {titulo}\n\n{texto.strip()}\n"
    if decisiones:
        md += "\n" + "\n".join(f"- **{k}:** {v}" for k, v in decisiones.items()) + "\n"
    display(Markdown(md))


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %"


def num(x: float) -> str:
    """Entero con separador de miles por espacio fino, como en el Entregable 1 (p. ej. 40 428 967)."""
    return f"{int(round(x)):,}".replace(",", " ")


def secs(s: float) -> str:
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"
