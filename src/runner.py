"""Orquestador del experimento. Cada etapa es reanudable: omite las corridas terminadas.

Uso (desde la raíz del repositorio, con el entorno activado):
    python -m src.runner prepare        conversión a Parquet, manifiesto y grupos de variables
    python -m src.runner sklearn        Optuna, codificaciones y líneas base
    python -m src.runner escala_sklearn escalado por tamaño en scikit-learn (P1)
    python -m src.runner spark          rejilla, modelos, escalado, escalamiento fuerte y débil
    python -m src.runner bench          lectura del CSV y del Parquet con cada motor
    python -m src.runner explain        LIME, SHAP e importancia por permutación
    python -m src.runner ablaciones     ablaciones del MLP (al final: son lo menos esencial)
    python -m src.runner all            todas las etapas en orden
    python -m src.runner status         avance
    python -m src.runner smoke          prueba de humo completa con datos sintéticos (minutos)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np

from . import config as C


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with open(C.LOGS / "runner.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ preparación
def stage_prepare() -> None:
    from . import bench, data as D, eda as E
    from .persist import save_json
    C.ensure_dirs()
    if C.SMOKE and not any((C.RAW_DIR / n).exists() for n in C.RAW_CANDIDATES):
        from . import synthetic
        synthetic.make(C.RAW_DIR / "train.gz")
    raw = D.find_raw()
    rel = raw.relative_to(C.DATA).as_posix()
    if not (C.COMMON / "raw_sha256.json").exists():
        log(f"SHA-256 de {raw.name} (puede tardar unos minutos)")
        sha = D.sha256_file(raw)
        st = D.register_checksum(rel, sha)
        save_json({"archivo": rel, "sha256": sha, "estado": st}, C.COMMON / "raw_sha256.json")
        if st == "DIFERENTE":
            raise RuntimeError("El archivo de Avazu no coincide con la huella registrada en SHA256SUMS.txt.")
    rec: dict = {}
    with bench.stage(rec, "conversion"):
        info = D.csv_to_parquet(raw)
    log(f"Parquet: {info}")
    if info["estado"] == "creado":
        save_json({**info, **rec}, C.COMMON / "conversion_parquet.json")
    if not C.MANIFEST.exists():
        D.build_manifest(raw)
    man = D.load_manifest()
    for f in man["archivos_parquet"]:
        D.register_checksum(f["archivo"], f["sha256"])
    if not (C.COMMON / "column_groups.json").exists():
        c = E.con()
        save_json(E.column_groups(c), C.COMMON / "column_groups.json")
        c.close()
    save_json({"hardware": bench.hardware(), "fecha": datetime.now().isoformat(timespec="seconds")},
              C.COMMON / "hardware.json")
    log("Preparación lista")


# ------------------------------------------------------------------ scikit-learn
def _best_encoding() -> str:
    from .persist import collect
    df = collect("sklearn", "encoding")
    if df.empty:
        return C.MLP_TUNE_ENCODING
    return str(df.groupby("encoding")["val.roc_auc"].mean().idxmax())


def stage_sklearn() -> None:
    from . import sk_models as S
    log("Optuna: MLP")
    S.tune("mlp", C.MLP_TUNE_ENCODING, C.OPTUNA_TRIALS)
    log("Optuna: líneas base")
    S.tune("lr", C.MLP_TUNE_ENCODING, C.OPTUNA_TRIALS_BASELINES)
    S.tune("lr", "freq", C.OPTUNA_TRIALS_BASELINES)
    S.tune("hgb", "freq", C.OPTUNA_TRIALS_BASELINES)
    mlp = S.best_params("mlp", C.MLP_TUNE_ENCODING)
    size = C.MAIN_SIZE
    for enc in C.ENCODINGS:                               # P2
        for s in C.SEEDS:
            r = S.run("encoding", "mlp", enc, size, s, mlp)
            log(f"P2 mlp {enc} s{s}: AUC={r['test']['roc_auc']:.4f} fit={r['bench']['fit_s']:.1f}s")
    for kind, enc in [("lr", C.MLP_TUNE_ENCODING), ("lr", "freq"), ("hgb", "freq")]:   # P3
        for s in C.SEEDS:
            r = S.run("baselines", kind, enc, size, s, S.best_params(kind, enc))
            log(f"P3 {kind} {enc} s{s}: AUC={r['test']['roc_auc']:.4f}")
    log(f"Mejor codificación en validación: {_best_encoding()}")


def stage_escala_sklearn() -> None:
    """P1, lado scikit-learn: tiempo de entrenamiento frente al tamaño de la muestra. Es benchmark de hardware,
    así que debe correr en el mismo equipo que la etapa spark para que el punto de cruce sea comparable."""
    from . import sk_models as S
    for kind in ("lr_lbfgs", "mlp_lbfgs"):                # P1, lado scikit-learn
        prev = None
        for size_lbl, n in C.SIZES.items():
            for s in C.SEEDS[: C.SEEDS_BY_SIZE[size_lbl]]:
                if prev is not None:
                    est = prev[1] * n / prev[0]
                    if est > C.FIT_BUDGET_MIN * 60:
                        _skip_sklearn("scale", kind, C.P1_ENCODING, size_lbl, s, est)
                        continue
                r = S.run("scale", kind, C.P1_ENCODING, size_lbl, s, save_preds=(size_lbl == C.SPARK_MLP_SIZE))
                if not r.get("omitido"):
                    prev = (r["n_train"], r["bench"]["fit_s"])
                log(f"P1 sklearn {kind} {size_lbl} s{s}: fit={r.get('bench', {}).get('fit_s', float('nan')):.1f}s")


def stage_ablaciones() -> None:
    """Ablaciones del MLP (peso de clases, cruces, profundidad). Van al final: son lo menos esencial
    de la comparación principal y, con límite de tiempo, se ejecutan solo si sobra."""
    from . import sk_models as S
    mlp = S.best_params("mlp", C.MLP_TUNE_ENCODING)
    size = C.MAIN_SIZE
    best = _best_encoding()
    for s in C.SEEDS:                                     # ablaciones
        S.run("ablation_weight", "mlp", best, size, s, mlp, weighted=True)
        if best.startswith("hash"):
            S.run("ablation_cross", "mlp", best, size, s, mlp, crosses=True)
        else:
            S.run("ablation_cross", "mlp", C.MLP_TUNE_ENCODING, size, s, mlp, crosses=True)
        for hl in C.DEPTH_ABLATION:
            lbl = "-".join(map(str, hl))
            S.run("ablation_depth", "mlp", best, size, s, {**mlp, "hidden_layer_sizes": lbl}, tag=f"_h{lbl}")
    log("Ablaciones listas")


def _skip_sklearn(exp, kind, enc, size, seed, est):
    from .persist import is_done, run_dir, save_json
    d = run_dir("sklearn", exp, f"{kind}_{enc}_{size}_s{seed:02d}")
    if not is_done(d):
        save_json({"env": "scikit-learn", "exp": exp, "model": kind, "encoding": enc, "size": size,
                   "n_train": C.SIZES[size], "seed": seed, "omitido": True, "fit_s_estimado": est,
                   "motivo": f"tiempo estimado {est / 60:.0f} min > presupuesto de {C.FIT_BUDGET_MIN} min"},
                  d / "result.json")


# ------------------------------------------------------------------ PySpark
def stage_spark() -> None:
    from . import spark_models as SP
    from .persist import save_json
    cores = C.SPARK_CORES_MAIN
    spark = SP.get_spark(cores)
    save_json(SP.spark_info(spark), C.COMMON / "spark_info.json")
    nueva = not (C.COMMON / "spark_grid_mlp.json").exists()
    if nueva:
        log("Rejilla del MLP de Spark")
    grid = SP.grid_tune(cores)
    mlp = grid["best_params"]
    if nueva:
        log(f"Mejor MLP de Spark: {mlp} (AUC val {grid['best_auc_val']:.4f})")
    # Modelos de Spark con 5 semillas. Las corridas se agrupan por codificación y tamaño para
    # preprocesar una sola vez cada combinación.
    best_row = next(r for r in grid["resultados"] if not r["omitido"] and r["auc_val"] == grid["best_auc_val"])
    n_mlp = C.SIZES[C.SPARK_MLP_SIZE]
    est_mlp = best_row["fit_s"] * n_mlp / grid["n_train"]          # extrapolación lineal desde la rejilla
    for s in C.SEEDS:
        SP.run("main", "lr", C.P1_ENCODING, C.SPARK_MLP_SIZE, s, {"maxIter": C.SPARK_LR["maxIter"]}, cores)
        if est_mlp > C.FIT_BUDGET_MIN * 60:
            if not SP.is_done(SP.run_dir("pyspark", "main", f"mlp_{C.P1_ENCODING}_{C.SPARK_MLP_SIZE}_s{s:02d}_c{cores}")):
                SP.skipped("main", "mlp", C.P1_ENCODING, C.SPARK_MLP_SIZE, s, cores, est_mlp)
                log(f"Spark main s{s}: MLP omitido (estimado {est_mlp / 60:.0f} min)")
            continue
        r = SP.run("main", "mlp", C.P1_ENCODING, C.SPARK_MLP_SIZE, s, mlp, cores)
        if r.get("_nuevo"):                # solo se registra lo calculado en este proceso
            log(f"Spark main s{s}: MLP omitido ({r['motivo']})" if r.get("omitido") else
                f"Spark main s{s}: MLP AUC={r['test']['roc_auc']:.4f} fit={r['bench']['fit_s']:.1f}s")
    for s in C.SEEDS:
        SP.run("main", "gbt", "freq", C.SPARK_MLP_SIZE, s, C.SPARK_GBT, cores)
    for s in C.SEEDS:
        SP.run("main", "lr", C.P1_ENCODING, C.MAIN_SIZE, s, {"maxIter": C.SPARK_LR["maxIter"]}, cores)
    for s in C.SEEDS:
        SP.run("main", "gbt", "freq", C.MAIN_SIZE, s, C.SPARK_GBT, cores)
    for kind in ("lr_lbfgs", "mlp_lbfgs"):                # P1, lado PySpark
        prev = None
        params = {"maxIter": C.P1_MAX_ITER, "layers": [C.P1_HIDDEN], "solver": "l-bfgs"}
        for size_lbl, n in C.SIZES.items():
            for s in C.SEEDS[: C.SEEDS_BY_SIZE[size_lbl]]:
                est = prev[1] * n / prev[0] if prev is not None else None
                if est is not None and est > C.FIT_BUDGET_MIN * 60:
                    SP.skipped("scale", kind, C.P1_ENCODING, size_lbl, s, cores, est)
                    continue
                r = SP.run("scale", kind, C.P1_ENCODING, size_lbl, s, params, cores,
                           save_preds=(size_lbl == C.SPARK_MLP_SIZE))
                if not r.get("omitido"):
                    prev = (r["n_train"], r["bench"]["fit_s"])
                if r.get("_nuevo"):
                    log(f"P1 Spark {kind} {size_lbl} s{s}: omitido ({r['motivo']})" if r.get("omitido") else
                        f"P1 Spark {kind} {size_lbl} s{s}: fit={r['bench']['fit_s']:.1f}s")
    plan = {"lr": (C.SCALING_ROWS, C.SPARK_LR["maxIter"] // 2), "mlp_lbfgs": (C.SPARK_TUNE_ROWS * 5, 10)}

    def _escala(kind, n, p, iters, exp):
        r = SP.scaling_run(kind, n, p, iters, exp)
        if not r.get("_nuevo"):
            return
        etiqueta = "Fuerte" if exp == "strong" else "Débil"
        log(f"{etiqueta} {kind} n={n} p={p}: omitido ({r['motivo']})" if r.get("omitido") else
            f"{etiqueta} {kind} n={n} p={p}: {r['bench']['fit_s']:.1f}s")

    for kind, (n, iters) in plan.items():                 # escalamiento fuerte (mismo n para todos los p)
        for p in C.SPARK_CORES_SCALING:
            _escala(kind, n, p, iters, "strong")
    for kind, (n, iters) in plan.items():                 # escalamiento débil (n = n0 * p)
        base = C.WEAK_BASE_ROWS if kind == "lr" else C.SPARK_TUNE_ROWS // 2
        for p in C.SPARK_CORES_SCALING:
            _escala(kind, base * p, p, iters, "weak")
    SP.stop_spark()


# ------------------------------------------------------------------ lectura
def stage_bench() -> None:
    """Lectura completa (todas las columnas) del CSV y del Parquet de entrenamiento con cada motor.

    Todas las lecturas son por flujo o por bloques, para no cargar el archivo completo en memoria.
    """
    import duckdb
    import pandas as pd
    import polars as pl
    import pyarrow.dataset as ds
    from . import bench, data as D, spark_models as SP
    from .persist import save_json
    out = C.COMMON / "lectura.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    raw = D.find_raw()

    def measure(name, fn):
        if name in res and "error" not in res[name]:
            return
        rec: dict = {}
        try:
            with bench.stage(rec, "lectura"):
                extra = fn()
            res[name] = {**rec, **(extra or {})}
        except Exception as e:   # noqa: BLE001
            res[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        save_json(res, out)
        log(f"Lectura {name}: {res[name]}")

    pq = (C.PARQUET_DIR / "day=*" / "*.parquet").as_posix()
    days = ",".join(map(str, C.TRAIN_DAYS))
    cols = C.CAT_COLS + [C.TARGET]
    allc = ", ".join(f"count({c})" for c in cols)
    raw_allc = ", ".join(f"count({c})" for c in D.RAW_COLUMNS)

    def duck(q):
        con = duckdb.connect()
        try:
            return con.execute(q).fetchone()
        finally:
            con.close()

    def pandas_csv():
        return {"filas": int(sum(len(ch) for ch in pd.read_csv(raw, chunksize=1_000_000, dtype=str)))}

    def pandas_pq():
        d = ds.dataset(C.PARQUET_DIR, format="parquet", partitioning="hive")
        n = 0
        for b in d.to_batches(columns=cols, filter=ds.field("day").isin(C.TRAIN_DAYS), batch_size=1_000_000):
            n += len(b.to_pandas())
        return {"filas": n}

    measure("CSV | pandas (bloques de 1 M)", pandas_csv)
    measure("CSV | DuckDB", lambda: {"filas": int(duck(
        f"SELECT count(*), {raw_allc} FROM read_csv('{raw.as_posix()}', header=true, all_varchar=true)")[0])})
    measure("CSV | Polars (streaming)", lambda: {"filas": int(pl.scan_csv(raw, infer_schema=False)
                                                                .select(pl.all().count()).collect(engine="streaming")[0, 0])})
    measure("CSV | PySpark", lambda: SP.read_csv_count(raw))
    measure("Parquet días 21-28 | pandas (lotes de pyarrow)", pandas_pq)
    measure("Parquet días 21-28 | DuckDB", lambda: {"filas": int(duck(
        f"SELECT count(*), {allc} FROM read_parquet('{pq}', hive_partitioning=true) WHERE day IN ({days})")[0])})
    measure("Parquet días 21-28 | Polars (streaming)", lambda: {"filas": int(
        D.scan().filter(pl.col("day").is_in(C.TRAIN_DAYS)).select(pl.col(cols).count()).collect(engine="streaming")[0, 0])})
    measure("Parquet días 21-28 | PySpark", lambda: SP.read_parquet_count(C.TRAIN_DAYS))
    SP.stop_spark()


# ------------------------------------------------------------------ interpretabilidad
def stage_explain() -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from . import data as D, explain as X, features as F, metrics as M, sk_models as S
    from .persist import collect, save_json
    out = C.SUMMARY / "explain.json"
    if out.exists():
        log("Interpretabilidad ya calculada")
        return
    enc = _best_encoding()
    runs = collect("sklearn", "encoding")
    runs = runs[runs["encoding"] == enc]
    med_seed = int(runs.sort_values("test.roc_auc").iloc[len(runs) // 2]["seed"])
    params = S.best_params("mlp", C.MLP_TUNE_ENCODING)
    log(f"Explicando el MLP con {enc}, semilla mediana {med_seed}")
    tr = D.load_train(C.SIZES[C.MAIN_SIZE])
    va = D.load_val()
    te = D.load_test()
    ytr, yva, yte = D.y_of(tr), D.y_of(va), D.y_of(te)
    # Se reutilizan el modelo y el preprocesador guardados por la corrida de P2 (mismos datos, parámetros y semilla);
    # si no existen, se reentrena.
    from .persist import load_pickle
    fila = runs[runs["seed"] == med_seed].iloc[0]
    mp = C.RUNS / str(fila["run_dir"]) / "model.pkl"
    pp = C.RESULTS / "models" / "sklearn_prep" / f"{enc}_{C.MAIN_SIZE}.pkl"
    try:
        model, prep = load_pickle(mp), load_pickle(pp)
        log(f"Modelo reutilizado de {mp.parent.name} (sin reentrenar)")
    except Exception:  # noqa: BLE001
        log("No hay modelo guardado para esa corrida: se reentrena")
        prep = F.CTRPreprocessor(enc)
        model = S.make_model("mlp", params, med_seed).fit(prep.fit_transform(tr, ytr), ytr)
    p_va = model.predict_proba(prep.transform(va))[:, 1]
    p_te = model.predict_proba(prep.transform(te))[:, 1]
    thr = M.best_threshold(yva, p_va)
    space = X.InterpretableSpace(prep.rare_.transform(tr.select(C.CAT_COLS)))
    f = X.predict_fn(prep, model, space)
    rng = np.random.default_rng(C.SAMPLE_SEED)
    Ztr = space.encode(tr.select(C.CAT_COLS)[rng.choice(tr.height, min(20_000, tr.height), replace=False)])
    n_chk = min(50_000, va.height)
    ix = rng.choice(va.height, n_chk, replace=False)
    Zva = space.encode(va.select(C.CAT_COLS)[ix])
    fid_space = {"auc_original": float(roc_auc_score(yva[ix], p_va[ix])),
                 "auc_espacio_interpretable": float(roc_auc_score(yva[ix], f(Zva)[:, 1])),
                 "corr_pearson_prob": float(np.corrcoef(p_va[ix], f(Zva)[:, 1])[0, 1])}
    cases = X.select_cases(yte, p_te, thr)
    Zc = space.encode(te.select(C.CAT_COLS)[cases["fila"].to_numpy()])
    log(f"LIME sobre {len(cases)} casos")
    lime_rows = []
    for kf in C.LIME_KERNEL_FACTORS:
        for ns in C.LIME_NUM_SAMPLES:
            for sd in C.LIME_SEEDS:
                ex = X.make_lime(space, Ztr, kf, sd)
                for i, z in enumerate(Zc):
                    r = X.lime_one(ex, z, f, ns)
                    lime_rows.append({"caso": i, "tipo": cases["tipo"].iat[i], "kernel": kf, "num_samples": ns, "semilla": sd,
                                      "r2": r["r2"], "pred_local": r["pred_local"],
                                      "pesos": {space.cols[j]: w for j, w in r["pesos"].items()}})
    lime_df = pd.DataFrame(lime_rows)
    lime_df.to_json(C.SUMMARY / "lime_runs.json", orient="records", force_ascii=False)
    log("SHAP y permutación")
    bg = Ztr[rng.choice(len(Ztr), min(C.SHAP_BACKGROUND, len(Ztr)), replace=False)]
    sv = X.shap_values(f, bg, Zc)
    perm = X.permutation_importance(f, Zva[: min(20_000, len(Zva))], yva[ix][: min(20_000, len(Zva))], space.cols)
    perm.to_csv(C.SUMMARY / "permutation_importance.csv", index=False)
    cases_out = cases.assign(**{f"x_{c}": space.decode(Zc)[c].to_list() for c in space.cols})
    cases_out.to_csv(C.SUMMARY / "lime_casos.csv", index=False)
    # Todo lo necesario para repetir o ampliar la interpretabilidad sin reentrenar.
    from .persist import save_pickle_safe
    save_pickle_safe({"encoding": enc, "seed": med_seed, "threshold": thr, "params": params, "model": model,
                      "prep": prep, "space": space, "Ztr_muestra": Ztr, "Zva_muestra": Zva, "Zcasos": Zc,
                      "casos": cases, "shap": sv, "lime": lime_df, "permutacion": perm},
                     C.SUMMARY / "explain_objetos.pkl", "los objetos de interpretabilidad")
    save_json({"encoding": enc, "seed": med_seed, "threshold": thr, "fidelidad_espacio": fid_space,
               "cols": space.cols, "shap": sv, "p_casos": f(Zc)[:, 1]}, out)
    log("Interpretabilidad lista")


# ------------------------------------------------------------------ estado
def status() -> None:
    from .persist import collect
    for env in ("sklearn", "pyspark"):
        df = collect(env)
        if df.empty:
            print(f"{env}: sin corridas")
            continue
        om = df["omitido"].map(lambda v: v is True) if "omitido" in df else False
        print(f"{env}: {len(df)} corridas ({int(np.sum(om))} omitidas por presupuesto)")
        print(df.groupby("exp").size().to_string())


def smoke() -> None:
    """Ejecuta todo con datos sintéticos en una carpeta aparte (data_smoke/, results_smoke/)."""
    env = {**os.environ, "AVAZU_SMOKE": "1"}
    for st in ("prepare", "sklearn", "spark", "bench", "explain"):
        print(f"--- smoke: {st}", flush=True)
        subprocess.run([sys.executable, "-m", "src.runner", st], check=True, env=env, cwd=C.ROOT)


STAGES = {"prepare": stage_prepare, "sklearn": stage_sklearn, "spark": stage_spark, "bench": stage_bench,
          "explain": stage_explain, "escala_sklearn": stage_escala_sklearn, "ablaciones": stage_ablaciones}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all", "status", "smoke"])
    a = ap.parse_args()
    if a.stage == "status":
        return status()
    if a.stage == "smoke":
        return smoke()
    C.ensure_dirs()
    todo = list(STAGES) if a.stage == "all" else [a.stage]
    for st in todo:
        if st == "spark" and not os.environ.get(_CHILD):
            _spark_supervised()
            continue
        t0 = time.perf_counter()
        hijo = bool(os.environ.get(_CHILD))       # proceso relanzado por _spark_supervised: sin encabezados repetidos
        if not hijo:
            log(f"== Etapa {st} ==")
        try:
            STAGES[st]()
        except Exception as e:
            from .persist import SinTiempo, save_json as _sj
            from .spark_models import OOM_EXIT_CODE, RESTART_EXIT_CODE, SparkOOM, SparkRestart
            if isinstance(e, SinTiempo):
                log(f"Etapa {st}: {e}. No se inician corridas nuevas; lo pendiente queda sin ejecutar.")
                reg = C.COMMON / "limite_tiempo.json"
                prev = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {}
                prev[st] = {"limite": f"{C.LIMITE:%Y-%m-%d %H:%M}", "detenida": datetime.now().isoformat(timespec="seconds")}
                _sj(prev, reg)
                sys.exit(0)
            if isinstance(e, SparkRestart):
                sys.exit(RESTART_EXIT_CODE)
            if isinstance(e, SparkOOM):
                log(f"Memoria insuficiente en la JVM ({e}); la corrida queda registrada como omitida "
                    "y la etapa se reinicia en un proceso nuevo")
                sys.exit(OOM_EXIT_CODE)
            log(traceback.format_exc())
            raise
        if not hijo:
            log(f"== Etapa {st} terminada en {(time.perf_counter() - t0) / 60:.1f} min ==")


_CHILD = "AVAZU_STAGE_CHILD"


def _spark_supervised(max_restarts: int = 20) -> None:
    """Ejecuta la etapa de Spark en un proceso hijo y lo relanza si la JVM se quedó sin memoria.

    Tras un OutOfMemoryError la JVM no es confiable; el hijo registra la corrida como omitida y
    termina con OOM_EXIT_CODE, y el nuevo proceso retoma desde las corridas ya guardadas.
    """
    from .spark_models import OOM_EXIT_CODE, RESTART_EXIT_CODE
    log("== Etapa spark ==")
    env = {**os.environ, _CHILD: "1"}
    t0, oom = time.perf_counter(), 0
    while True:
        rc = subprocess.run([sys.executable, "-m", "src.runner", "spark"], env=env, cwd=C.ROOT).returncode
        if rc == 0:
            log(f"== Etapa spark terminada en {(time.perf_counter() - t0) / 60:.1f} min ==")
            return
        if rc == RESTART_EXIT_CODE:
            continue
        if rc != OOM_EXIT_CODE:
            raise SystemExit(rc)
        oom += 1
        if oom > max_restarts:
            raise SystemExit("La etapa spark se reinició demasiadas veces por falta de memoria.")
        log(f"Reinicio {oom} de la etapa spark tras falta de memoria")


if __name__ == "__main__":
    main()
