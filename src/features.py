"""Preprocesamiento de variables categóricas para scikit-learn.

Todas las transformaciones se ajustan solo con la ventana de entrenamiento:
- agrupación de categorías raras en `__RARE__` (umbral de conteo en entrenamiento);
- codificación como factor experimental: hashing (2^12, 2^14, 2^16), frecuencia y target
  encoding fuera de pliegue (out-of-fold);
- en las estrategias densas, one-hot para las variables de baja cardinalidad,
  log(1 + conteo) para la frecuencia y StandardScaler sobre las columnas numéricas.

El hashing replica la lógica de `pyspark.ml.feature.FeatureHasher`: cada par
"variable=valor" se asigna a una columna `hash mod d` con valor 1 y las colisiones se suman.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import config as C

HASH_SEED = 42


def add_crosses(df: pl.DataFrame, crosses=C.CROSSES) -> pl.DataFrame:
    return df.with_columns([pl.concat_str([pl.col(a), pl.col(b)], separator="|").alias(f"{a}_x_{b}")
                            for a, b in crosses])


def cross_names(crosses=C.CROSSES) -> list[str]:
    return [f"{a}_x_{b}" for a, b in crosses]


class RareGrouper:
    """Reemplaza por `__RARE__` las categorías con menos de `min_count` apariciones en entrenamiento."""

    def __init__(self, cols: list[str], min_count: int = C.RARE_MIN_COUNT):
        self.cols, self.min_count = cols, min_count

    def fit(self, df: pl.DataFrame) -> "RareGrouper":
        self.vocab_ = {}
        for c in self.cols:
            vc = df.group_by(c).len()
            self.vocab_[c] = vc.filter(pl.col("len") >= self.min_count)[c]
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.when(pl.col(c).is_in(self.vocab_[c].implode())).then(pl.col(c)).otherwise(pl.lit(C.RARE_TOKEN)).alias(c)
            for c in self.cols])

    def rare_share(self, df: pl.DataFrame) -> dict[str, float]:
        g = self.transform(df.select(self.cols))
        return {c: float((g[c] == C.RARE_TOKEN).mean()) for c in self.cols}


def hash_matrix(df: pl.DataFrame, cols: list[str], dim: int) -> sp.csr_matrix:
    """Matriz dispersa (n, dim) con un 1 por variable en la columna hash("variable=valor") mod dim."""
    n, k = df.height, len(cols)
    idx = df.select([
        (pl.concat_str([pl.lit(f"{c}="), pl.col(c)]).hash(seed=HASH_SEED) % dim).cast(pl.Int32).alias(c)
        for c in cols]).to_numpy()                                     # (n, k) int32
    X = sp.csr_matrix((np.ones(n * k, dtype=np.float32), idx.ravel(), np.arange(0, n * k + 1, k)),
                      shape=(n, dim))
    X.sum_duplicates()
    return X


class CTRPreprocessor:
    """Preprocesador común: `encoding` en {hash_12, hash_14, hash_16, freq, target}."""

    def __init__(self, encoding: str, groups: dict | None = None, crosses: bool = False):
        self.encoding = encoding
        self.groups = groups or C.column_groups()
        self.crosses = crosses
        self.cols = list(C.CAT_COLS) + (cross_names() if crosses else [])
        self.low = [c for c in self.cols if c in self.groups["baja"]]
        self.high = [c for c in self.cols if c not in self.low]

    # -------------------------------------------------------------- utilidades
    def _prep(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.crosses:
            df = add_crosses(df)
        return self.rare_.transform(df.select(self.cols))

    @property
    def is_hash(self) -> bool:
        return self.encoding.startswith("hash")

    # -------------------------------------------------------------- ajuste
    def fit(self, df: pl.DataFrame, y: np.ndarray | None = None) -> "CTRPreprocessor":
        self.fit_transform(df, y)
        return self

    def fit_transform(self, df: pl.DataFrame, y: np.ndarray | None = None):
        src = add_crosses(df) if self.crosses else df
        self.rare_ = RareGrouper(self.cols).fit(src)
        g = self.rare_.transform(src.select(self.cols))
        if self.is_hash:
            self.dim_ = C.HASH_DIMS[self.encoding]
            return hash_matrix(g, self.cols, self.dim_)
        # one-hot de baja cardinalidad
        self.levels_ = {c: g[c].unique().sort().to_list() for c in self.low}
        if self.encoding == "freq":
            self.counts_ = {c: g.group_by(c).len().rename({"len": "n"}) for c in self.high}
            num = self._freq(g)
        elif self.encoding == "target":
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import TargetEncoder
            if y is None:
                raise ValueError("target encoding requiere y")
            # Pliegues contiguos en el tiempo (las filas llegan ordenadas por hora), sin barajar.
            self.te_ = TargetEncoder(target_type="binary", cv=KFold(C.TARGET_ENCODER_CV, shuffle=False))
            num = self.te_.fit_transform(g.select(self.high).to_pandas(), y).astype(np.float64)   # fuera de pliegue
        else:
            raise ValueError(self.encoding)
        self.mu_, self.sd_ = num.mean(0), num.std(0) + 1e-9
        return self._dense(g, num)

    def transform(self, df: pl.DataFrame):
        g = self._prep(df)
        if self.is_hash:
            return hash_matrix(g, self.cols, self.dim_)
        if self.encoding == "freq":
            num = self._freq(g)
        else:
            num = self.te_.transform(g.select(self.high).to_pandas()).astype(np.float64)
        return self._dense(g, num)

    # -------------------------------------------------------------- piezas densas
    def _freq(self, g: pl.DataFrame) -> np.ndarray:
        out = []
        for c in self.high:
            n = g.select(c).join(self.counts_[c], on=c, how="left", maintain_order="left")["n"].fill_null(0)
            out.append(np.log1p(n.to_numpy().astype(np.float64)))
        return np.column_stack(out) if out else np.zeros((g.height, 0))

    def _dense(self, g: pl.DataFrame, num: np.ndarray) -> np.ndarray:
        z = (num - self.mu_) / self.sd_
        oh = []
        for c in self.low:
            v = g[c].to_numpy()
            oh.append(np.column_stack([(v == lv) for lv in self.levels_[c]]).astype(np.float32))
        return np.hstack([z.astype(np.float32), *oh]) if oh else z.astype(np.float32)

    def feature_names(self) -> list[str]:
        if self.is_hash:
            return [f"h{i}" for i in range(self.dim_)]
        pre = "logfreq" if self.encoding == "freq" else "te"
        return [f"{pre}_{c}" for c in self.high] + [f"{c}={lv}" for c in self.low for lv in self.levels_[c]]

    def n_features(self) -> int:
        return self.dim_ if self.is_hash else len(self.feature_names())


def matrix_bytes(X) -> int:
    if sp.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(X.nbytes)


def compute_column_groups(df: pl.DataFrame, low_card_max: int = C.LOW_CARD_MAX) -> dict:
    """Clasifica las variables por cardinalidad tras agrupar las raras, en la ventana de entrenamiento."""
    rg = RareGrouper(C.CAT_COLS).fit(df)
    card = {c: int(len(rg.vocab_[c]) + 1) for c in C.CAT_COLS}
    low = [c for c in C.CAT_COLS if card[c] <= low_card_max]
    return {"baja": low, "alta": [c for c in C.CAT_COLS if c not in low], "cardinalidad_tras_raras": card,
            "umbral_baja": low_card_max, "rare_min_count": C.RARE_MIN_COUNT}
