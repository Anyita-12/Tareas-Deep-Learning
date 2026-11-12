"""Medición de tiempo de pared y memoria pico por etapa.

`Stage` mide una etapa (lectura, preprocesamiento, entrenamiento o predicción). La memoria
se muestrea cada 0.1 s como la memoria residente (RSS) del proceso de Python más la de sus
procesos hijos; en PySpark el hijo es la máquina virtual de Java que ejecuta el driver.
"""
from __future__ import annotations

import os
import platform
import threading
import time
from contextlib import contextmanager

import psutil

MB = 1024 ** 2


def _rss_tree(proc: psutil.Process) -> int:
    total = 0
    try:
        total += proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try:
                total += ch.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total


class PeakMemory:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()

    def __enter__(self):
        self.base = _rss_tree(self.proc)
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_tree(self.proc))
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_tree(self.proc))


@contextmanager
def stage(record: dict, name: str, rows: int | None = None):
    """Agrega a `record` {name}_s, {name}_peak_mb, {name}_delta_mb y {name}_rows_s."""
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
    record[f"{name}_s"] = dt
    record[f"{name}_peak_mb"] = pm.peak / MB
    record[f"{name}_delta_mb"] = (pm.peak - pm.base) / MB
    if rows:
        record[f"{name}_rows_s"] = rows / max(dt, 1e-9)


def hardware() -> dict:
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "sistema": f"{platform.system()} {platform.release()}",
        "procesador": platform.processor() or platform.machine(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "frecuencia_max_mhz": round(freq.max, 0) if freq else None,
        "ram_gb": round(vm.total / 1024 ** 3, 1),
        "python": platform.python_version(),
    }
