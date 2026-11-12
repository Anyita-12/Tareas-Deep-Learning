"""Ejecución completa sin supervisión: etapas, cuadernos y compilación del informe.

    python -m src.todo
    python -m src.todo --plan "sklearn=2026-11-11 02:00;spark=2026-11-11 07:00;ablaciones=2026-11-11 09:00"

Con `--plan`, cada etapa deja de iniciar corridas nuevas a su hora límite (lo ya terminado se conserva y
lo pendiente queda sin ejecutar), y las etapas pasan en este orden: prepare, sklearn, spark, bench, explain,
ablaciones. Las horas pueden ser absolutas ("AAAA-MM-DD HH:MM") o relativas ("+90s", "+10m", "+2h").

1. Verifica requisitos (Java, datos, disco) y guarda las versiones de las librerías.
2. Ejecuta `python -m src.runner all` y lo reintenta hasta 5 veces si falla (todo es reanudable).
3. Ejecuta los cuadernos docs/01 a docs/11 en orden y guarda cada uno con sus salidas.
4. Compila el informe con Jupyter Book (docs/_build/html).
5. Escribe `results/logs/TERMINADO.txt` (o `FALLO.txt`) con el resumen.

No publica en GitHub: eso se hace después de revisar el resultado. Registro en results/logs/todo.log.
Si se interrumpe (corte de luz), se vuelve a ejecutar el mismo comando: continúa desde lo guardado.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import config as C

MAX_INTENTOS = 5
DOCS = C.ROOT / "docs"
NOTEBOOKS = sorted(DOCS.glob("[0-1][0-9]_*.ipynb"))


def log(msg: str) -> None:
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with open(C.LOGS / "todo.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")
        f.flush()
        os.fsync(f.fileno())


def no_suspender() -> None:
    """Pide a Windows que no suspenda el equipo mientras este proceso viva (no impide cerrar la tapa)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def finalizar(ok: bool, detalle: str) -> None:
    for nombre in ("TERMINADO.txt", "FALLO.txt"):
        (C.LOGS / nombre).unlink(missing_ok=True)
    (C.LOGS / ("TERMINADO.txt" if ok else "FALLO.txt")).write_text(
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n{detalle}\n", encoding="utf-8")
    log(("TERMINADO: " if ok else "FALLO: ") + detalle.splitlines()[0])


# ------------------------------------------------------------------ 1. requisitos
def requisitos() -> None:
    log(f"Modo: {'PRUEBA DE HUMO' if C.SMOKE else 'DATOS REALES'} | resultados en {C.RESULTS}")
    java = subprocess.run(["java", "-version"], capture_output=True, text=True)
    if java.returncode != 0:
        raise SystemExit("Java no está disponible: cierra y abre la terminal, o reinstálalo (winget install EclipseAdoptium.Temurin.17.JDK).")
    if not C.SMOKE and not (C.RAW_DIR / "train.gz").exists():
        raise SystemExit(f"Falta {C.RAW_DIR / 'train.gz'} (descárgalo de Kaggle sin descomprimir).")
    if os.name == "nt" and not Path(os.environ.get("HADOOP_HOME", "C:/hadoop")).joinpath("bin", "winutils.exe").exists():
        raise SystemExit(r"Falta C:\hadoop\bin\winutils.exe (ver README, sección Hadoop para Windows).")
    libre = shutil.disk_usage(C.ROOT).free / 1024 ** 3
    log(f"Disco libre: {libre:.0f} GB")
    if not C.SMOKE and libre < 25:
        raise SystemExit(f"Disco libre insuficiente ({libre:.0f} GB); se recomiendan al menos 25 GB.")
    C.ensure_dirs()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (C.RESULTS / "requirements_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    log("Versiones de las librerías guardadas en results/requirements_freeze.txt")


# ------------------------------------------------------------------ 2. etapas
# La interpretabilidad va justo después de scikit-learn (solo necesita P2) para no quedar detrás de Spark.
ORDEN = ["prepare", "sklearn", "explain", "spark", "escala_sklearn", "bench", "ablaciones"]


def parse_plan(txt: str, t0: datetime) -> dict[str, datetime]:
    from datetime import timedelta
    plan: dict[str, datetime] = {}
    for parte in filter(None, (x.strip() for x in txt.split(";"))):
        etapa, _, v = parte.partition("=")
        etapa, v = etapa.strip(), v.strip()
        if etapa not in ORDEN:
            raise SystemExit(f"Etapa desconocida en --plan: {etapa} (válidas: {', '.join(ORDEN)})")
        if v.startswith("+"):
            n, u = float(v[1:-1]), v[-1]
            plan[etapa] = t0 + timedelta(seconds=n * {"s": 1, "m": 60, "h": 3600}[u])
        else:
            plan[etapa] = datetime.fromisoformat(v)
    return plan


def etapas(plan: dict[str, datetime], elegidas: list[str]) -> list[str]:
    """Ejecuta cada etapa en su propio proceso (hasta 3 intentos) con su límite de tiempo, si lo tiene.
    Una etapa que falla no impide las siguientes: se registra y el informe se compila con lo que haya."""
    fallidas: list[str] = []
    for st in [e for e in ORDEN if e in elegidas]:
        env = {k: v for k, v in os.environ.items() if k != "AVAZU_LIMITE"}
        if st in plan:
            env["AVAZU_LIMITE"] = plan[st].isoformat(timespec="seconds")
            log(f"Etapa {st}: límite {plan[st]:%Y-%m-%d %H:%M}")
        for intento in range(1, 4):
            t0 = time.perf_counter()
            rc = subprocess.run([sys.executable, "-m", "src.runner", st], cwd=C.ROOT, env=env).returncode
            log(f"Etapa {st}: código {rc} en {(time.perf_counter() - t0) / 60:.1f} min (intento {intento} de 3)")
            if rc == 0:
                break
            time.sleep(30)
        else:
            fallidas.append(st)
    return fallidas


# ------------------------------------------------------------------ 3. cuadernos
def cuadernos() -> list[str]:
    """Ejecuta los cuadernos en orden. Una celda que falla (por ejemplo, porque su experimento no alcanzó a
    ejecutarse antes del límite de tiempo) se reemplaza por una nota en el informe y se registra en
    results/logs/celdas_sin_datos.json para revisarla; el resto del cuaderno sigue ejecutándose."""
    import nbformat
    from nbclient import NotebookClient
    from .persist import save_json
    from .report import SIN_DATOS
    fallos: list[str] = []
    registro: dict = {}
    for p in NOTEBOOKS:
        for intento in (1, 2):
            t0 = time.perf_counter()
            nb = nbformat.read(p, as_version=4)
            try:
                NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True,
                               resources={"metadata": {"path": str(DOCS)}}).execute()
            except Exception as e:  # noqa: BLE001   (el núcleo murió o no arrancó)
                nbformat.write(nb, p)
                log(f"Cuaderno {p.name}: error en el intento {intento}: {str(e)[-800:]}")
                if intento == 2:
                    fallos.append(p.name)
                continue
            errores = []
            for i, c in enumerate(nb.cells):
                if c.cell_type != "code":
                    continue
                err = [o for o in c.outputs if o.output_type == "error"]
                if err:
                    errores.append({"celda": i, "error": f"{err[0].ename}: {err[0].evalue}"[:300],
                                    "codigo": c.source[:200]})
                    c.outputs = [o for o in c.outputs if o.output_type != "error"] + [
                        nbformat.v4.new_output("display_data", data={"text/markdown": SIN_DATOS, "text/plain": SIN_DATOS})]
            nbformat.write(nb, p)
            registro[p.name] = errores
            log(f"Cuaderno {p.name}: OK en {time.perf_counter() - t0:.0f} s"
                + (f" ({len(errores)} celdas sin datos)" if errores else ""))
            break
    save_json(registro, C.LOGS / "celdas_sin_datos.json")
    return fallos


# ------------------------------------------------------------------ 4. informe
def informe() -> None:
    exe = Path(sys.executable).parent / ("jupyter-book.exe" if os.name == "nt" else "jupyter-book")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "jupyter_book"]
    log("Compilando el informe (jupyter-book build docs --all)")
    rc = subprocess.run([*cmd, "build", str(DOCS), "--all"], cwd=C.ROOT).returncode
    if rc != 0:
        raise RuntimeError(f"jupyter-book terminó con código {rc}")
    if not (DOCS / "_build" / "html" / "index.html").exists():
        raise RuntimeError("no se generó docs/_build/html/index.html")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="", help='límites por etapa, p. ej. "sklearn=2026-11-11 02:00;spark=+7h"')
    ap.add_argument("--etapas", default=",".join(ORDEN),
                    help='etapas a ejecutar, separadas por comas (orden fijo), o "ninguna" para solo cuadernos e informe')
    ap.add_argument("--sin-informe", action="store_true", help="no ejecuta los cuadernos ni compila el libro")
    args = ap.parse_args()
    plan = parse_plan(args.plan, datetime.now())
    elegidas = [] if args.etapas.strip() == "ninguna" else [e.strip() for e in args.etapas.split(",") if e.strip()]
    malas = [e for e in elegidas if e not in ORDEN]
    if malas:
        raise SystemExit(f"Etapas desconocidas: {', '.join(malas)} (válidas: {', '.join(ORDEN)})")
    no_suspender()
    t0 = time.perf_counter()
    try:
        requisitos()
        fallidas = etapas(plan, elegidas)
        if args.sin_informe:
            dur = (time.perf_counter() - t0) / 3600
            finalizar(not fallidas, f"Etapas {', '.join(elegidas)} terminadas en {dur:.1f} h (sin informe)."
                      + (f" Fallaron: {', '.join(fallidas)}" if fallidas else ""))
            return
        fallos = cuadernos()
        informe()
    except SystemExit as e:
        finalizar(False, f"Requisito no cumplido: {e}")
        raise
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        finalizar(False, f"{type(e).__name__}: {e}")
        raise SystemExit(1)
    dur = (time.perf_counter() - t0) / 3600
    if fallos or fallidas:
        finalizar(False, f"El informe se compiló, pero fallaron etapas: {', '.join(fallidas) or 'ninguna'}; "
                         f"cuadernos: {', '.join(fallos) or 'ninguno'} (duración {dur:.1f} h)")
        raise SystemExit(1)
    cortes = C.COMMON / "limite_tiempo.json"
    if cortes.exists():
        log(f"Etapas detenidas por límite de tiempo: {', '.join(json.loads(cortes.read_text(encoding='utf-8')))}")
    finalizar(True, f"Etapas, {len(NOTEBOOKS)} cuadernos e informe listos en {dur:.1f} h. "
                    "Siguiente: revisar docs/_build/html/index.html y publicar con ghp-import.")


if __name__ == "__main__":
    main()
