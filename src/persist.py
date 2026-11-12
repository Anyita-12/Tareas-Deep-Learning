"""Guardado atómico de resultados (JSON y arreglos NumPy) y lectura de corridas."""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class SinTiempo(Exception):
    """Se alcanzó el límite de tiempo de la etapa (AVAZU_LIMITE): no se inician corridas nuevas."""


def comprobar_tiempo() -> None:
    from datetime import datetime
    if C.LIMITE is not None and datetime.now() >= C.LIMITE:
        raise SinTiempo(f"límite de tiempo {C.LIMITE:%Y-%m-%d %H:%M} alcanzado")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _atomic_write(path: Path, writer, suffix: str = ".tmp") -> None:
    """Escribe en un temporal, fuerza el volcado a disco (fsync) y lo renombra: si se va la luz a mitad
    de la escritura queda el archivo anterior intacto o el nuevo completo, nunca uno a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + suffix)
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(obj, path: Path) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False, default=_default).encode("utf-8")
    _atomic_write(path, lambda f: f.write(data))


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_array(a: np.ndarray, path: Path) -> None:
    _atomic_write(path, lambda f: np.save(f, a))


def save_pickle(obj, path: Path) -> None:
    """Pickle atómico (con fsync). Úsalo para modelos y objetos ajustados."""
    _atomic_write(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_safe(obj, path: Path, que: str = "objeto") -> bool:
    """Guarda un pickle sin tumbar el entrenamiento si falla; avisa en voz alta y devuelve si se guardó."""
    try:
        save_pickle(obj, path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[AVISO] No se pudo guardar {que} en {path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return False


def run_dir(env: str, exp: str, key: str) -> Path:
    return C.RUNS / env / exp / key


def is_done(d: Path) -> bool:
    return (d / "result.json").exists()


def collect(env: str | None = None, exp: str | None = None) -> pd.DataFrame:
    """Une todos los result.json en una tabla (una fila por corrida)."""
    pat = f"{env or '*'}/{exp or '*'}/*/result.json"
    rows = [load_json(p) for p in sorted(C.RUNS.glob(pat))]
    return pd.json_normalize(rows) if rows else pd.DataFrame()
