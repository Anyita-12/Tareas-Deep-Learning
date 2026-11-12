"""Cálculos del análisis exploratorio con DuckDB sobre el Parquet completo (lectura fuera de memoria).

Las consultas de la ventana de entrenamiento usan los días 21 a 28. El día 29 aparece solo
en la tasa de clics por día. Del día 30 (prueba) se usan únicamente las distribuciones de las
variables de entrada para medir la deriva (PSI); su etiqueta no se consulta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _duck

TRAIN = ",".join(map(str, C.TRAIN_DAYS))


def _glob() -> str:
    return (C.PARQUET_DIR / "day=*" / "*.parquet").as_posix()


def con():
    c = _duck()
    c.execute(f"CREATE OR REPLACE VIEW avazu AS SELECT * FROM read_parquet('{_glob()}', hive_partitioning=true)")
    c.execute(f"CREATE OR REPLACE VIEW train AS SELECT * FROM avazu WHERE day IN ({TRAIN})")
    c.execute(f"CREATE OR REPLACE VIEW test_x AS SELECT * EXCLUDE (click) FROM avazu WHERE day = {C.TEST_DAY}")
    return c


def schema_and_missing(c) -> pd.DataFrame:
    tipos = c.execute("DESCRIBE avazu").df()[["column_name", "column_type"]]
    nulls = c.execute("SELECT " + ", ".join(f"count(*) - count({col}) AS {col}" for col in tipos.column_name)
                      + " FROM avazu").df().T.reset_index()
    nulls.columns = ["column_name", "nulos"]
    return tipos.merge(nulls, on="column_name").rename(columns={"column_name": "variable", "column_type": "tipo"})


def ctr_by(c, col: str, days: str = TRAIN) -> pd.DataFrame:
    return c.execute(f"SELECT {col}, count(*) AS n, avg(click) AS ctr FROM avazu WHERE day IN ({days}) "
                     f"GROUP BY {col} ORDER BY {col}").df()


def column_profile(c, col: str) -> dict:
    """Cardinalidad, cola larga y concentración de una variable en entrenamiento."""
    counts = c.execute(f"SELECT count(*) AS n FROM train GROUP BY {col} ORDER BY n DESC").df()["n"].to_numpy()
    top1 = c.execute(f"SELECT {col} AS v FROM train GROUP BY {col} ORDER BY count(*) DESC LIMIT 1").fetchone()[0]
    total = counts.sum()
    return {"variable": col, "categorias": int(len(counts)), "vistas_una_vez": float((counts == 1).mean()),
            "categorias_lt_umbral": float((counts < C.RARE_MIN_COUNT).mean()),
            "filas_a_rare": float(counts[counts < C.RARE_MIN_COUNT].sum() / total),
            "valor_mas_frecuente": str(top1), "cuota_top1": float(counts[0] / total),
            "card_tras_raras": int((counts >= C.RARE_MIN_COUNT).sum() + int((counts < C.RARE_MIN_COUNT).any())),
            "_curva": np.cumsum(counts) / total}


def long_tail_curve(curve: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Fracción de categorías (ordenadas por frecuencia) vs fracción acumulada de filas."""
    x = np.linspace(0, 1, points)
    idx = np.clip((x * len(curve)).astype(int) - 1, 0, len(curve) - 1)
    return x, np.where(x == 0, 0.0, curve[idx])


def _grouped(col: str, table: str = "train") -> str:
    return (f"(SELECT CASE WHEN f.v IS NULL THEN '{C.RARE_TOKEN}' ELSE t.{col} END AS v, t.* EXCLUDE ({col}) "
            f"FROM {table} t LEFT JOIN (SELECT {col} AS v FROM train GROUP BY {col} "
            f"HAVING count(*) >= {C.RARE_MIN_COUNT}) f ON t.{col} = f.v)")


def association(c, col: str) -> dict:
    """Información mutua (bits) y V de Cramér entre la variable agrupada y `click` en entrenamiento."""
    t = c.execute(f"SELECT v, count(*) AS n, sum(click) AS k FROM {_grouped(col)} GROUP BY v").df()
    n, k = t["n"].to_numpy(float), t["k"].to_numpy(float)
    N, K = n.sum(), k.sum()
    py1 = K / N
    joint = np.stack([(n - k) / N, k / N])                 # (2, V)
    pv = n / N
    py = np.array([1 - py1, py1])[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(joint * np.log2(joint / (py * pv[None, :])))
    exp1 = n * py1
    chi2 = np.sum((k - exp1) ** 2 / np.clip(exp1, 1e-12, None) + ((n - k) - (n - exp1)) ** 2 / np.clip(n - exp1, 1e-12, None))
    hy = -(py1 * np.log2(py1) + (1 - py1) * np.log2(1 - py1))
    return {"variable": col, "mi_bits": float(mi), "mi_normalizada": float(mi / hy), "v_cramer": float(np.sqrt(chi2 / N)),
            "categorias": int(len(t))}


def psi(c, col: str, top: int = 20, eps: float = 1e-4) -> dict:
    """PSI entre entrenamiento (esperado) y prueba (observado) con las `top` categorías de entrenamiento."""
    q = (f"WITH tr AS (SELECT v, count(*) AS n FROM {_grouped(col)} GROUP BY v), "
         f"topv AS (SELECT v FROM tr ORDER BY n DESC LIMIT {top}), "
         f"te AS (SELECT v, count(*) AS n FROM {_grouped(col, 'test_x')} GROUP BY v) "
         f"SELECT coalesce(tp.v, 'resto') AS b, sum(tr.n) AS e FROM tr LEFT JOIN topv tp ON tr.v = tp.v GROUP BY b")
    e = c.execute(q).df().set_index("b")["e"]
    q2 = (f"WITH tr AS (SELECT v, count(*) AS n FROM {_grouped(col)} GROUP BY v), "
          f"topv AS (SELECT v FROM tr ORDER BY n DESC LIMIT {top}), "
          f"te AS (SELECT v, count(*) AS n FROM {_grouped(col, 'test_x')} GROUP BY v) "
          f"SELECT coalesce(tp.v, 'resto') AS b, sum(te.n) AS o FROM te LEFT JOIN topv tp ON te.v = tp.v GROUP BY b")
    o = c.execute(q2).df().set_index("b")["o"]
    j = pd.concat([e, o], axis=1).fillna(0)
    pe = np.clip(j["e"] / j["e"].sum(), eps, None)
    po = np.clip(j["o"] / j["o"].sum(), eps, None)
    return {"variable": col, "psi": float(np.sum((po - pe) * np.log(po / pe)))}


def column_groups(c) -> dict:
    card = {}
    for col in C.CAT_COLS:
        keep, rare = c.execute(
            f"SELECT count(*) FILTER (WHERE n >= {C.RARE_MIN_COUNT}), count(*) FILTER (WHERE n < {C.RARE_MIN_COUNT}) "
            f"FROM (SELECT count(*) AS n FROM train GROUP BY {col})").fetchone()
        card[col] = int(keep) + int(rare > 0)
    low = [col for col in C.CAT_COLS if card[col] <= C.LOW_CARD_MAX]
    return {"baja": low, "alta": [col for col in C.CAT_COLS if col not in low], "cardinalidad_tras_raras": card,
            "umbral_baja": C.LOW_CARD_MAX, "rare_min_count": C.RARE_MIN_COUNT}


def psi_label(v: float) -> str:
    return "estable" if v < 0.1 else ("moderada" if v < 0.25 else "alta")
