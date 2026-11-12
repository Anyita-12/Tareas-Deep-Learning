"""Datos sintéticos con el esquema de Avazu, solo para la prueba de humo (AVAZU_SMOKE=1).

Reproducen la estructura del archivo original (columnas, formato de `hour`, cardinalidades
con cola larga y una tasa de clics cercana al 17 %) para verificar que el flujo completo
se ejecuta. No se usan en ningún resultado del informe.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from . import config as C


def _zipf_ids(rng, n, card, a, prefix):
    ranks = np.minimum(rng.zipf(a, n), card) - 1
    return np.array([f"{prefix}{r:06x}" for r in range(card)])[ranks], ranks


def make(path: Path, rows_per_day: int = 12_000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    days = list(range(21, 31))
    n = rows_per_day * len(days)
    day = np.repeat(days, rows_per_day)
    hh = np.sort(rng.integers(0, 24, n).reshape(len(days), -1), axis=1).ravel()
    hour = 14_10_00_00 + day * 100 + hh
    cols = {}
    eff = np.zeros(n)
    specs = {"site_id": (400, 1.6), "site_domain": (500, 1.6), "site_category": (20, 2.0), "app_id": (600, 1.5),
             "app_domain": (80, 1.8), "app_category": (25, 2.0), "device_id": (20_000, 1.3),
             "device_ip": (40_000, 1.2), "device_model": (900, 1.5), "C14": (700, 1.5), "C17": (300, 1.6),
             "C19": (40, 1.9), "C20": (120, 1.7), "C21": (45, 1.9)}
    for c, (card, a) in specs.items():
        vals, ranks = _zipf_ids(rng, n, card, a, c[:2].lower())
        cols[c] = vals
        if c in ("site_id", "app_id", "C14", "C17", "device_model"):
            w = rng.normal(0, 0.6, card)
            eff += w[ranks]
    small = {"C1": (["1005", "1002", "1010", "1012"], [0.9, 0.05, 0.03, 0.02]),
             "banner_pos": (["0", "1", "7"], [0.72, 0.27, 0.01]),
             "device_type": (["1", "0", "4", "5"], [0.92, 0.05, 0.02, 0.01]),
             "device_conn_type": (["0", "2", "3", "5"], [0.86, 0.08, 0.05, 0.01]),
             "C15": (["320", "300", "216", "728"], [0.93, 0.05, 0.01, 0.01]),
             "C16": (["50", "250", "36", "480"], [0.94, 0.04, 0.01, 0.01]),
             "C18": (["0", "3", "2", "1"], [0.42, 0.34, 0.17, 0.07])}
    for c, (vals, p) in small.items():
        idx = rng.choice(len(vals), n, p=p)
        cols[c] = np.array(vals)[idx]
        eff += rng.normal(0, 0.3, len(vals))[idx]
    eff += 0.25 * np.sin(2 * np.pi * hh / 24) - 0.05 * (day - 25)
    logit = -1.75 + eff + rng.normal(0, 0.5, n)
    click = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    ids = rng.integers(0, 2**63 - 1, n, dtype=np.int64)
    header = ["id", "click", "hour", *C.ORIGINAL_CAT]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for i in range(n):
            row = [str(ids[i]), str(click[i]), str(hour[i])] + [str(cols[c][i]) for c in C.ORIGINAL_CAT]
            f.write(",".join(row) + "\n")
    return path
