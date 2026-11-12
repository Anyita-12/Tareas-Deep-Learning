"""Datos de Avazu: verificación, conversión a Parquet particionado por día y muestreo.

La conversión usa DuckDB, que lee el CSV comprimido por bloques y escribe Parquet sin
cargar el archivo completo en memoria. Cada fila recibe una clave uniforme `u` en [0, 1),
calculada como el hash de su `id`. Las muestras de n filas se definen como las n filas
con menor `u` dentro de la ventana: son reproducibles, idénticas en scikit-learn y en
PySpark, y anidadas (la muestra de 100 mil filas está contenida en la de 1 millón).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import config as C

RAW_COLUMNS = ["id", "click", "hour", *C.ORIGINAL_CAT]


# ------------------------------------------------------------------ huellas
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_sums(path: Path = C.SUMS) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split(maxsplit=1)
            out[rel.strip()] = sha
    return out


def register_checksum(rel: str, sha: str, path: Path = C.SUMS) -> str:
    """Registra la huella si no existe. Devuelve NUEVO, IGUAL o DIFERENTE."""
    sums = read_sums(path)
    if rel in sums:
        return "IGUAL" if sums[rel] == sha else "DIFERENTE"
    sums[rel] = sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())), encoding="utf-8")
    return "NUEVO"


def find_raw() -> Path:
    for name in C.RAW_CANDIDATES:
        p = C.RAW_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No encuentro el archivo de entrenamiento de Avazu en {C.RAW_DIR}. "
        "Descargue train.gz desde Kaggle (competencia avazu-ctr-prediction) y colóquelo allí.")


# ------------------------------------------------------------------ conversión
def _duck(memory_limit: str = "12GB"):
    import duckdb
    tmp = C.INTERIM / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=true")
    return con


def csv_to_parquet(raw: Path | None = None, overwrite: bool = False) -> dict:
    """CSV (o CSV.gz) de Avazu -> Parquet particionado por día con variables temporales derivadas."""
    raw = raw or find_raw()
    out = C.PARQUET_DIR
    if out.exists() and any(out.glob("day=*/*.parquet")) and not overwrite:
        return {"estado": "existente", "ruta": str(out)}
    # Se escribe en una carpeta temporal y se renombra al final: si la corrida se interrumpe,
    # la próxima ejecución no confunde un Parquet incompleto con uno terminado.
    final = out
    out = final.with_name(final.name + "_tmp")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    types = {"id": "VARCHAR", "click": "TINYINT", "hour": "INTEGER", **{c: "VARCHAR" for c in C.ORIGINAL_CAT}}
    cols_sql = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    franja_sql = " ".join(f"WHEN hora_n BETWEEN {min(r)} AND {max(r)} THEN '{k}'" for k, r in C.FRANJAS.items())
    cats = ", ".join(C.ORIGINAL_CAT)
    sql = f"""
    COPY (
      SELECT click, hour, {cats},
             CAST(hora_n AS VARCHAR) AS hora,
             CAST(dayofweek(ts) AS VARCHAR) AS dia_semana,
             CASE {franja_sql} END AS franja,
             CAST(hash(id) AS DOUBLE) / 18446744073709551616.0 AS u,
             day
      FROM (
        SELECT *, strptime(CAST(hour AS VARCHAR), '%y%m%d%H') AS ts,
               CAST(hour % 100 AS INTEGER) AS hora_n,
               CAST((hour // 100) % 100 AS INTEGER) AS day
        FROM read_csv('{raw.as_posix()}', header=true, columns={{{cols_sql}}})
      )
    ) TO '{out.as_posix()}' (FORMAT parquet, PARTITION_BY (day), COMPRESSION zstd, OVERWRITE_OR_IGNORE true)
    """
    t0 = time.perf_counter()
    con = _duck()
    con.execute(sql)
    con.close()
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    return {"estado": "creado", "ruta": str(final), "segundos": round(time.perf_counter() - t0, 1)}


def parquet_files(day: int | None = None) -> list[Path]:
    pat = f"day={day}/*.parquet" if day is not None else "day=*/*.parquet"
    return sorted(C.PARQUET_DIR.glob(pat))


def build_manifest(raw: Path | None = None) -> dict:
    """Filas por día, tamaño en disco y huella SHA-256 de cada archivo Parquet.

    No calcula la tasa de clics del día de prueba: el día 30 no se consulta antes de evaluar.
    """
    raw = raw or find_raw()
    con = _duck()
    glob = (C.PARQUET_DIR / "day=*" / "*.parquet").as_posix()
    rows = con.execute(
        f"SELECT day, count(*) AS n FROM read_parquet('{glob}', hive_partitioning=true) GROUP BY day ORDER BY day"
    ).df()
    con.close()
    files = []
    for p in parquet_files():
        files.append({"archivo": p.relative_to(C.DATA).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    man = {
        "raw": {"archivo": raw.name, "bytes": raw.stat().st_size},
        "filas_por_dia": {int(r.day): int(r.n) for r in rows.itertuples()},
        "archivos_parquet": files,
        "test_day": C.TEST_DAY,
        "test_sha256": hashlib.sha256("".join(f["sha256"] for f in files
                                              if f["archivo"].startswith(f"parquet/avazu/day={C.TEST_DAY}/")).encode()).hexdigest(),
    }
    C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    C.MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    return man


def load_manifest() -> dict:
    return json.loads(C.MANIFEST.read_text(encoding="utf-8"))


def verify_test_frozen() -> None:
    """Comprueba que el Parquet del día de prueba no cambió desde que se registró el manifiesto."""
    man = load_manifest()
    files = [p for p in parquet_files(C.TEST_DAY)]
    sha = hashlib.sha256("".join(sha256_file(p) for p in files).encode()).hexdigest()
    if sha != man["test_sha256"]:
        raise RuntimeError("El conjunto de prueba (día 30) no coincide con el manifiesto. No continuar.")


# ------------------------------------------------------------------ lectura
def scan() -> pl.LazyFrame:
    return pl.scan_parquet(C.PARQUET_DIR / "**" / "*.parquet", hive_partitioning=True)


def window_rows(days: list[int]) -> int:
    fpd = load_manifest()["filas_por_dia"]
    return int(sum(fpd[str(d)] if str(d) in fpd else fpd[d] for d in days))


def sample_threshold(days: list[int], n: int | None) -> float:
    """Valor de `u` de la n-ésima fila con menor `u` en la ventana. Se calcula una vez y se guarda."""
    if n is None:
        return 1.0
    path = C.COMMON / "sample_thresholds.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{days[0]}-{days[-1]}:{n}"
    if key not in table:
        if n >= window_rows(days):
            thr = 1.0
        else:
            con = _duck()
            glob = (C.PARQUET_DIR / "day=*" / "*.parquet").as_posix()
            dl = ",".join(map(str, days))
            thr = float(con.execute(
                f"SELECT max(u) FROM (SELECT u FROM read_parquet('{glob}', hive_partitioning=true) "
                f"WHERE day IN ({dl}) ORDER BY u LIMIT {int(n)})").fetchone()[0])
            con.close()
        table[key] = thr
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return float(table[key])


def load(days: list[int], n: int | None = None, columns: list[str] | None = None,
         with_target: bool = True) -> pl.DataFrame:
    """Carga las n filas de menor `u` de los días indicados, en orden temporal."""
    cols = list(dict.fromkeys((columns or C.CAT_COLS) + ([C.TARGET] if with_target else []) + ["hour", "day", C.SAMPLE_KEY]))
    thr = sample_threshold(days, n)
    lf = scan().filter(pl.col("day").is_in(days))
    if thr < 1.0:
        lf = lf.filter(pl.col(C.SAMPLE_KEY) <= thr)
    df = lf.select(cols).collect()
    # Orden total y reproducible: cronológico y, dentro de cada hora, por la clave `u`.
    # PySpark usa el mismo orden, por lo que las predicciones de ambos entornos quedan alineadas.
    return df.sort(["hour", C.SAMPLE_KEY])


def load_train(n: int | None) -> pl.DataFrame:
    return load(C.TRAIN_DAYS, n)


def load_val(n: int | None = None) -> pl.DataFrame:
    return load([C.VAL_DAY], n if n is not None else C.VAL_ROWS)


def load_test(n: int | None = C.TEST_ROWS) -> pl.DataFrame:
    verify_test_frozen()
    return load([C.TEST_DAY], n)


def y_of(df: pl.DataFrame) -> np.ndarray:
    return df[C.TARGET].to_numpy().astype(np.int8)


def to_pandas_cats(df: pl.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    return df.select(cols or C.CAT_COLS).to_pandas()
