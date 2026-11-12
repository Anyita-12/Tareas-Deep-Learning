"""Fusiona en `results/` los resultados de PySpark y del benchmark calculados en un segundo equipo.

    python -m src.fusionar "C:\\ruta\\a\\results_companero"            elige automáticamente
    python -m src.fusionar "C:\\ruta\\a\\results_companero" --usar B   fuerza los del compañero
    python -m src.fusionar "C:\\ruta\\a\\results_companero" --usar A   conserva los propios

Regla: todo lo que depende del hardware (PySpark, escalado de scikit-learn, escalamiento fuerte y débil y
lectura) se toma como un bloque de UN solo equipo, para que los tiempos sean comparables entre sí. Gana el
equipo con más corridas de PySpark terminadas; en empate, el propio (A). Antes de reemplazar, se hace una copia
de respaldo en results/_respaldo_A_<fecha>.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from . import config as C
from .persist import save_json

BLOQUE_RUNS = [Path("pyspark"), Path("sklearn") / "scale"]
BLOQUE_COMMON = ["spark_grid_mlp.json", "spark_grid_mlp_parcial.json", "spark_info.json", "spark_mem_checks.json",
                 "lectura.json"]
ETAPAS_BLOQUE = ("spark", "escala_sklearn", "bench")


def terminadas(results: Path) -> int:
    n = 0
    for p in (results / "runs" / "pyspark").glob("*/*/result.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("omitido") is not True:
                n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultados_b", help="carpeta results del compañero (descomprimida)")
    ap.add_argument("--usar", choices=["auto", "A", "B"], default="auto")
    a = ap.parse_args()
    B = Path(a.resultados_b).resolve()
    A = C.RESULTS.resolve()
    if not (B / "runs").exists():
        raise SystemExit(f"{B} no parece una carpeta results (falta runs/).")
    na, nb = terminadas(A), terminadas(B)
    print(f"Corridas de PySpark terminadas: equipo A = {na}, equipo B = {nb}")
    usar = a.usar if a.usar != "auto" else ("B" if nb > na else "A")
    if usar == "A":
        print("Se conservan los resultados del equipo A; no se copia nada.")
        save_json({"fuente_bloque_hardware": "A", "pyspark_A": na, "pyspark_B": nb,
                   "fecha": datetime.now().isoformat(timespec="seconds")}, C.COMMON / "fusion.json")
        return
    resp = A / f"_respaldo_A_{datetime.now():%Y%m%d_%H%M}"
    resp.mkdir(parents=True)
    for rel in BLOQUE_RUNS:
        src, dst = B / "runs" / rel, A / "runs" / rel
        if dst.exists():
            shutil.move(str(dst), str(resp / "runs" / rel))
        if src.exists():
            shutil.copytree(src, dst)
            print(f"Copiado runs/{rel.as_posix()}: {sum(1 for _ in dst.glob('**/result.json'))} corridas")
        else:
            print(f"AVISO: el equipo B no tiene runs/{rel.as_posix()}")
    for f in BLOQUE_COMMON:
        if (A / "common" / f).exists():
            (resp / "common").mkdir(parents=True, exist_ok=True)
            shutil.copy2(A / "common" / f, resp / "common" / f)
        if (B / "common" / f).exists():
            shutil.copy2(B / "common" / f, A / "common" / f)
    if (B / "common" / "hardware.json").exists():
        shutil.copy2(B / "common" / "hardware.json", A / "common" / "hardware_benchmark.json")
    lim_a = A / "common" / "limite_tiempo.json"
    lim = json.loads(lim_a.read_text(encoding="utf-8")) if lim_a.exists() else {}
    lim = {k: v for k, v in lim.items() if k not in ETAPAS_BLOQUE}
    lim_b = B / "common" / "limite_tiempo.json"
    if lim_b.exists():
        lim.update({k: v for k, v in json.loads(lim_b.read_text(encoding="utf-8")).items() if k in ETAPAS_BLOQUE})
    if lim:
        save_json(lim, lim_a)
    elif lim_a.exists():
        lim_a.unlink()
    save_json({"fuente_bloque_hardware": "B", "pyspark_A": na, "pyspark_B": nb, "respaldo_A": resp.name,
               "fecha": datetime.now().isoformat(timespec="seconds")}, C.COMMON / "fusion.json")
    print(f"Listo. Respaldo de lo reemplazado en {resp}")


if __name__ == "__main__":
    main()
