"""PySpark: sesión local, preprocesamiento equivalente al de scikit-learn, modelos y benchmarking.

El preprocesamiento sigue la misma lógica que `features.py`:
- categorías raras -> `__RARE__` con el umbral contado en entrenamiento (uniones con tablas de frecuencia);
- hashing con `FeatureHasher` (pares "variable=valor", MurmurHash3, valor 1);
- frecuencia: log(1 + conteo) + `StandardScaler` para alta cardinalidad y
  `StringIndexer` + `OneHotEncoder` para baja cardinalidad.

Las predicciones se ordenan por (hour, u), el mismo orden de scikit-learn, para comparar
ambos entornos observación por observación.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from itertools import product

from pathlib import Path

import numpy as np

from . import bench, config as C, data as D, metrics as M
from .persist import is_done, load_json, run_dir, save_array, save_json, save_pickle_safe, comprobar_tiempo

_SPARK = {"session": None, "cores": None}

OOM_EXIT_CODE = 75       # la JVM se quedó sin memoria: el orquestador relanza la etapa
RESTART_EXIT_CODE = 76   # cambio de configuración de datos: el orquestador relanza la etapa con una JVM nueva
_STATE = {"prepared": None}


class SparkOOM(RuntimeError):
    """La JVM se quedó sin memoria. La corrida queda registrada como omitida y la etapa se reinicia
    en un proceso nuevo, porque tras un OutOfMemoryError el estado de la JVM no es confiable."""


class SparkRestart(RuntimeError):
    """Se necesita otra configuración de datos: se continúa en una JVM nueva para que la memoria retenida
    por la anterior (planes, tablas auxiliares, modelos) no se acumule ni distorsione los tiempos."""


def _is_oom(e: BaseException) -> bool:
    """OutOfMemoryError dentro de una tarea, o JVM terminada por -XX:+ExitOnOutOfMemoryError."""
    txt = f"{type(e).__name__} {e}"
    return any(k in txt for k in ("OutOfMemoryError", "Py4JNetworkError", "Answer from Java side is empty",
                                   "ConnectionRefusedError", "Connection refused"))


# ------------------------------------------------------------------ sesión
def get_spark(cores: int = C.SPARK_CORES_MAIN):
    from pyspark.sql import SparkSession
    if _SPARK["session"] is not None and _SPARK["cores"] == cores:
        return _SPARK["session"]
    stop_spark()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.name == "nt":  # Windows: Hadoop necesita winutils.exe y hadoop.dll en %HADOOP_HOME%\\bin
        home = os.environ.get("HADOOP_HOME") or ("C:\\hadoop" if os.path.isdir("C:\\hadoop\\bin") else "")
        if home:
            os.environ["HADOOP_HOME"] = home
            os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + os.environ.get("PATH", "")
    C.SPARK_TMP.mkdir(parents=True, exist_ok=True)
    b = (SparkSession.builder.master(f"local[{cores}]").appName(f"avazu-ctr-local{cores}")
         .config("spark.local.dir", C.SPARK_TMP.as_posix()))
    for k, v in C.SPARK_CONF.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    _SPARK.update(session=spark, cores=cores)
    return spark


def stop_spark() -> None:
    if _SPARK["session"] is not None:
        _CACHE.clear()      # los DataFrames en caché pertenecen a la sesión que se cierra
        _SPARK["session"].stop()
        _SPARK.update(session=None, cores=None)
        time.sleep(2)


def spark_info(spark) -> dict:
    sc = spark.sparkContext
    conf = {k: v for k, v in sc.getConf().getAll() if k.startswith("spark.") and "password" not in k}
    return {"version": spark.version, "master": sc.master, "default_parallelism": sc.defaultParallelism,
            "ui": sc.uiWebUrl, "conf": conf}


# ------------------------------------------------------------------ métricas de la Spark UI (API REST)
def _rest(spark, path: str):
    url = f"{spark.sparkContext.uiWebUrl}/api/v1/applications/{spark.sparkContext.applicationId}/{path}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def last_stage_id(spark) -> int:
    try:
        st = _rest(spark, "stages")
        return max((s["stageId"] for s in st), default=-1)
    except Exception:   # noqa: BLE001
        return -1


def stage_metrics(spark, since: int) -> dict:
    """Shuffle, spill y sesgo entre tareas de las etapas ejecutadas después de `since`."""
    try:
        st = [s for s in _rest(spark, "stages?status=complete") if s["stageId"] > since]
    except Exception:   # noqa: BLE001
        return {}
    if not st:
        return {}
    out = {"n_etapas": len(st), "n_tareas": int(sum(s.get("numCompleteTasks", 0) for s in st)),
           "shuffle_read_mb": sum(s.get("shuffleReadBytes", 0) for s in st) / bench.MB,
           "shuffle_write_mb": sum(s.get("shuffleWriteBytes", 0) for s in st) / bench.MB,
           "spill_memoria_mb": sum(s.get("memoryBytesSpilled", 0) for s in st) / bench.MB,
           "spill_disco_mb": sum(s.get("diskBytesSpilled", 0) for s in st) / bench.MB,
           "executor_run_s": sum(s.get("executorRunTime", 0) for s in st) / 1000}
    big = max(st, key=lambda s: s.get("executorRunTime", 0))
    try:
        q = _rest(spark, f"stages/{big['stageId']}/{big['attemptId']}/taskSummary?quantiles=0.5,0.95,1.0")
        rt = q["executorRunTime"]
        out.update(etapa_mayor=big.get("name", "")[:80], tareas_etapa_mayor=big.get("numTasks"),
                   tarea_mediana_s=rt[0] / 1000, tarea_p95_s=rt[1] / 1000, tarea_max_s=rt[2] / 1000,
                   sesgo_max_mediana=rt[2] / max(rt[0], 1e-9))
    except Exception:   # noqa: BLE001
        pass
    return out


# ------------------------------------------------------------------ datos
def read_window(spark, days: list[int], n: int | None):
    from pyspark.sql import functions as F
    thr = D.sample_threshold(days, n)
    df = spark.read.parquet(C.PARQUET_DIR.as_uri()).filter(F.col("day").isin(days))
    if thr < 1.0:
        df = df.filter(F.col(C.SAMPLE_KEY) <= thr)
    return df.select(*C.CAT_COLS, C.TARGET, "hour", C.SAMPLE_KEY).withColumn("label", F.col(C.TARGET).cast("double"))


class SparkPrep:
    """Preprocesamiento en Spark ajustado con entrenamiento: `encoding` en {hash, freq}."""

    def __init__(self, encoding: str, dim: int = C.SPARK_HASH_DIM):
        self.encoding, self.dim = encoding, dim
        g = C.column_groups()
        self.low = [c for c in C.CAT_COLS if c in g["baja"]]
        self.high = [c for c in C.CAT_COLS if c not in self.low]

    def fit(self, train):
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (FeatureHasher, OneHotEncoder, StandardScaler, StringIndexer,
                                        VectorAssembler)
        from pyspark.sql import functions as F
        # Sin sugerencia de broadcast: Spark decide en tiempo de ejecución (AQE) según el tamaño real de cada
        # tabla; las de device_ip y device_id pueden ser demasiado grandes para copiarlas a cada tarea.
        self.keep_ = {c: train.groupBy(c).count().filter(F.col("count") >= C.RARE_MIN_COUNT)
                      .select(c, F.lit(True).alias(f"_k_{c}")).cache() for c in C.CAT_COLS}
        self.counts_ = {}
        g = self.rare(train)
        if self.encoding == "hash":
            stages = [FeatureHasher(inputCols=C.CAT_COLS, outputCol="features", numFeatures=self.dim)]
        else:
            self.counts_ = {c: g.groupBy(c).count().withColumnRenamed("count", f"_n_{c}").cache()
                            for c in self.high}
            g = self.freq(g)
            num = [f"_lf_{c}" for c in self.high]
            stages = [StringIndexer(inputCols=self.low, outputCols=[f"_i_{c}" for c in self.low], handleInvalid="keep"),
                      OneHotEncoder(inputCols=[f"_i_{c}" for c in self.low], outputCols=[f"_o_{c}" for c in self.low],
                                    dropLast=False, handleInvalid="keep"),
                      VectorAssembler(inputCols=num, outputCol="_num"),
                      StandardScaler(inputCol="_num", outputCol="_numz", withMean=True, withStd=True),
                      VectorAssembler(inputCols=["_numz", *[f"_o_{c}" for c in self.low]], outputCol="features")]
        self.pipe_ = Pipeline(stages=stages).fit(g)
        return self

    def release(self) -> None:
        """Libera de la caché las tablas auxiliares (categorías conservadas y frecuencias)."""
        for t in [*getattr(self, "keep_", {}).values(), *getattr(self, "counts_", {}).values()]:
            t.unpersist()

    def rare(self, df):
        from pyspark.sql import functions as F
        for c in C.CAT_COLS:
            df = (df.join(self.keep_[c], on=c, how="left")
                    .withColumn(c, F.when(F.col(f"_k_{c}"), F.col(c)).otherwise(F.lit(C.RARE_TOKEN)))
                    .drop(f"_k_{c}"))
        return df

    def freq(self, df):
        from pyspark.sql import functions as F
        for c in self.high:
            df = (df.join(self.counts_[c], on=c, how="left")
                    .withColumn(f"_lf_{c}", F.log1p(F.coalesce(F.col(f"_n_{c}"), F.lit(0)).cast("double")))
                    .drop(f"_n_{c}"))
        return df

    def transform(self, df):
        g = self.rare(df)
        if self.encoding == "freq":
            g = self.freq(g)
        return self.pipe_.transform(g).select("features", "label", "hour", C.SAMPLE_KEY)

    def n_features(self, df_t) -> int:
        return int(df_t.first()["features"].size)


def make_model(kind: str, params: dict, seed: int, n_features: int):
    from pyspark.ml.classification import GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier
    p = dict(params)
    if kind in ("lr", "lr_lbfgs"):
        return LogisticRegression(maxIter=p.get("maxIter", C.SPARK_LR["maxIter"]), regParam=p.get("regParam", C.P1_REG),
                                  elasticNetParam=0.0, standardization=False, tol=p.get("tol", 1e-6))
    if kind == "gbt":
        return GBTClassifier(maxIter=p.get("maxIter", C.SPARK_GBT["maxIter"]), maxDepth=p.get("maxDepth", C.SPARK_GBT["maxDepth"]),
                             stepSize=p.get("stepSize", C.SPARK_GBT["stepSize"]), seed=seed)
    if kind in ("mlp", "mlp_lbfgs"):
        hidden = p.get("layers", [C.P1_HIDDEN])
        return MultilayerPerceptronClassifier(layers=[n_features, *hidden, 2], maxIter=p.get("maxIter", C.P1_MAX_ITER),
                                              stepSize=p.get("stepSize", 0.03), solver=p.get("solver", "l-bfgs"),
                                              tol=p.get("tol", 1e-6), seed=seed)
    raise ValueError(kind)


def predict(model, df_t):
    """Probabilidad de clic en el orden (hour, u). Devuelve (y, p, segundos)."""
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    t0 = time.perf_counter()
    pdf = (model.transform(df_t).select("hour", C.SAMPLE_KEY, "label",
                                        vector_to_array("probability")[1].alias("p"))
           .orderBy("hour", C.SAMPLE_KEY).toPandas())
    return pdf["label"].to_numpy().astype(np.int8), pdf["p"].to_numpy().astype(np.float32), time.perf_counter() - t0


# ------------------------------------------------------------------ estado de la sesión (datos preprocesados)
_CACHE: dict = {}


def prepared(encoding: str, size: str, cores: int) -> dict:
    """Lee y preprocesa entrenamiento, validación y prueba; mide lectura y preprocesamiento."""
    key = (encoding, size, cores)
    if key in _CACHE:
        return _CACHE[key]
    if _STATE["prepared"] is not None and _STATE["prepared"] != key:
        raise SparkRestart(f"{_STATE['prepared']} -> {key}")
    _STATE["prepared"] = key
    for v in _CACHE.values():
        for k in ("tr", "va", "te"):
            v[k].unpersist()
        v["prep"].release()
    _CACHE.clear()
    spark = get_spark(cores)
    n = C.SIZES[size] if size in C.SIZES else int(size)     # etiqueta ("1M") o número de filas
    rec: dict = {}
    with bench.stage(rec, "lectura"):
        tr_raw = read_window(spark, C.TRAIN_DAYS, n).cache()
        n_tr = tr_raw.count()
    rec["lectura_rows_s"] = n_tr / max(rec["lectura_s"], 1e-9)
    va_raw = read_window(spark, [C.VAL_DAY], C.VAL_ROWS)
    te_raw = read_window(spark, [C.TEST_DAY], C.TEST_ROWS)
    prep = SparkPrep("hash" if encoding.startswith("hash") else "freq",
                     dim=C.HASH_DIMS.get(encoding, C.SPARK_HASH_DIM))
    with bench.stage(rec, "prep", rows=n_tr):
        prep.fit(tr_raw)
        tr = prep.transform(tr_raw).cache()
        tr.count()
    va = prep.transform(va_raw).cache()
    te = prep.transform(te_raw).cache()
    va.count(), te.count()
    tr_raw.unpersist()
    rec["n_features"] = prep.n_features(tr)
    _CACHE[key] = {"tr": tr, "va": va, "te": te, "n_train": n_tr, "rec": rec, "prep": prep}
    return _CACHE[key]


def run(exp: str, kind: str, encoding: str, size: str, seed: int, params: dict | None = None,
        cores: int = C.SPARK_CORES_MAIN, save_preds: bool = True, tag: str = "", evaluate: bool = True) -> dict:
    key = f"{kind}_{encoding}_{size}_s{seed:02d}_c{cores}{tag}"
    d = run_dir("pyspark", exp, key)
    if is_done(d):
        return load_json(d / "result.json")
    comprobar_tiempo()
    spark = get_spark(cores)
    P = prepared(encoding, size, cores)
    if kind.startswith("mlp"):
        chk = mem_check(P["n_train"], P["rec"]["n_features"])
        if not chk["cabe"]:
            return skipped(exp, kind, encoding, size, seed, cores, tag=tag, motivo=mem_motivo(chk))
    rec = dict(P["rec"])
    model = make_model(kind, params or {}, seed, P["rec"]["n_features"])
    since = last_stage_id(spark)
    try:
        with bench.stage(rec, "fit", rows=P["n_train"]):
            fitted = model.fit(P["tr"])
    except Exception as e:  # noqa: BLE001
        if not _is_oom(e):
            raise
        skipped(exp, kind, encoding, size, seed, cores, tag=tag,
                motivo=f"la JVM se quedó sin memoria (OutOfMemoryError) al ajustar con {P['n_train']:,} filas; "
                       f"memoria máxima de la JVM {jvm_heap_gb():.1f} GB")
        raise SparkOOM(f"{kind} {encoding} {size} s{seed}") from e
    rec["spark_ui_fit"] = stage_metrics(spark, since)
    n_iter = None
    for attr in ("totalIterations", "objectiveHistory"):
        try:
            v = getattr(fitted.summary, attr)
            n_iter = int(v) if attr == "totalIterations" else len(v)
            break
        except Exception:   # noqa: BLE001
            continue
    res = {"env": "PySpark", "exp": exp, "model": kind, "encoding": encoding, "size": size, "n_train": P["n_train"],
           "seed": seed, "cores": cores, "params": params or {}, "n_iter": n_iter, "bench": rec}
    if evaluate:
        yv, pv, _ = predict(fitted, P["va"])
        with bench.stage(rec, "predict"):
            yt, pt, _ = predict(fitted, P["te"])
        rec["predict_rows_s"] = len(yt) / max(rec["predict_s"], 1e-9)
        thr = M.best_threshold(yv, pv)
        from sklearn.metrics import roc_auc_score
        res["val"] = {"roc_auc": float(roc_auc_score(yv, pv))}
        res["test"] = M.binary_metrics(yt, pt, thr)
        if save_preds:
            save_array(pt, d / "p_test.npy")
            save_array(pv, d / "p_val.npy")
    rec["fit_s_per_iter"] = rec["fit_s"] / max(n_iter or 1, 1)
    res["run_dir"] = d.relative_to(C.RUNS).as_posix()
    res["artefactos"] = guardar_modelo_spark(d, fitted, kind)      # antes de result.json
    save_json(res, d / "result.json")
    return {**res, "_nuevo": True}


def guardar_modelo_spark(d, fitted, kind: str) -> dict:
    """Guarda el modelo ajustado: formato nativo de Spark (se recarga con `.load`) y, cuando existen,
    los parámetros como arreglos NumPy en un pickle. Un fallo avisa pero no tumba la corrida."""
    import sys
    arte: dict = {}
    try:
        fitted.write().overwrite().save((Path(d) / "modelo_spark").resolve().as_uri())
        arte["modelo_spark"] = "modelo_spark"
    except Exception as e:  # noqa: BLE001
        print(f"[AVISO] No se pudo guardar el modelo nativo de Spark en {d}: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
    try:
        par: dict = {"kind": kind}
        for a in ("weights", "coefficients"):
            if hasattr(fitted, a):
                par[a] = np.asarray(getattr(fitted, a).toArray())
        if hasattr(fitted, "intercept"):
            par["intercept"] = float(fitted.intercept)
        if hasattr(fitted, "featureImportances"):
            par["feature_importances"] = np.asarray(fitted.featureImportances.toArray())
        if len(par) > 1 and save_pickle_safe(par, Path(d) / "parametros.pkl", f"los parámetros de {d.name}"):
            arte["parametros"] = "parametros.pkl"
    except Exception as e:  # noqa: BLE001
        print(f"[AVISO] No se pudieron extraer los parámetros de {d}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return arte


def driver_gb() -> float:
    v = C.SPARK_CONF["spark.driver.memory"].lower()
    return float(v[:-1]) / (1024 if v.endswith("m") else 1) if v[-1] in "gm" else float(v) / 1024 ** 3


def jvm_heap_gb() -> float:
    """Memoria máxima real de la JVM (puede diferir de spark.driver.memory si la sesión ya existía)."""
    s = _SPARK["session"]
    if s is not None:
        try:
            return s.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory() / 1024 ** 3
        except Exception:  # noqa: BLE001
            pass
    return driver_gb()


def mlp_stack_gb(n: int, d: int = C.SPARK_HASH_DIM) -> float:
    """Memoria de la entrada apilada en bloques densos que el MLP de Spark guarda en caché: n * d * 8 bytes."""
    return n * d * 8 / 1024 ** 3


def heap_free_gb() -> float:
    """Memoria libre de la JVM tras recolectar basura. Sin sesión activa, estimación a partir del driver."""
    s = _SPARK["session"]
    if s is not None:
        try:
            rt = s.sparkContext._jvm.java.lang.Runtime.getRuntime()
            rt.gc()
            return (rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())) / 1024 ** 3
        except Exception:  # noqa: BLE001
            pass
    return 0.85 * driver_gb()


def mem_check(n: int, d: int = C.SPARK_HASH_DIM) -> dict:
    """¿Cabe el MLP de Spark con n filas? Registra la medición en common/spark_mem_checks.json."""
    libre = heap_free_gb()
    r = {"filas": n, "entrada_apilada_gb": mlp_stack_gb(n, d), "heap_max_gb": jvm_heap_gb(), "heap_libre_gb": libre,
         "limite_gb": libre / C.SPARK_MLP_MEM_FACTOR}
    r["cabe"] = r["entrada_apilada_gb"] <= r["limite_gb"]
    out = C.COMMON / "spark_mem_checks.json"
    hist = load_json(out) if out.exists() else []
    save_json([*hist, r], out)
    return r


def mlp_fits_memory(n: int, d: int = C.SPARK_HASH_DIM) -> bool:
    return mlp_stack_gb(n, d) * C.SPARK_MLP_MEM_FACTOR <= heap_free_gb()


def mem_motivo(r: dict) -> str:
    lim = (f"la memoria libre de la JVM ({r['heap_libre_gb']:.2f} GB)" if C.SPARK_MLP_MEM_FACTOR == 1 else
           f"el límite de {r['limite_gb']:.2f} GB (memoria libre de la JVM {r['heap_libre_gb']:.2f} GB / {C.SPARK_MLP_MEM_FACTOR:g})")
    return f"la entrada densa apilada ({r['entrada_apilada_gb']:.2f} GB) no cabe en {lim}"


def skipped(exp: str, kind: str, encoding: str, size: str, seed: int, cores: int, est_s: float | None = None,
            tag: str = "", motivo: str | None = None) -> dict:
    d = run_dir("pyspark", exp, f"{kind}_{encoding}_{size}_s{seed:02d}_c{cores}{tag}")
    res = {"env": "PySpark", "exp": exp, "model": kind, "encoding": encoding, "size": size, "n_train": C.SIZES.get(size),
           "seed": seed, "cores": cores, "omitido": True, "fit_s_estimado": est_s,
           "motivo": motivo or f"tiempo estimado de ajuste {est_s / 60:.0f} min > presupuesto de {C.FIT_BUDGET_MIN} min"}
    save_json(res, d / "result.json")
    return {**res, "_nuevo": True}


# ------------------------------------------------------------------ rejilla del MLP (validación temporal)
def grid_tune(cores: int = C.SPARK_CORES_MAIN) -> dict:
    """ParamGridBuilder recorrido a mano: cada combinación se evalúa en el día 29.

    `CrossValidator` y `TrainValidationSplit` parten los datos al azar, lo que rompería
    la validación temporal; por eso se usa la rejilla con una evaluación explícita.
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder
    out = C.COMMON / "spark_grid_mlp.json"
    if out.exists():
        return load_json(out)
    spark = get_spark(cores)
    P = prepared(C.P1_ENCODING, str(C.SPARK_TUNE_ROWS), cores)
    chk = mem_check(P["n_train"], P["rec"]["n_features"])
    if not chk["cabe"]:
        raise RuntimeError("La muestra de la rejilla del MLP de Spark no cabe en memoria: " + mem_motivo(chk)
                           + ". Reduzca SPARK_TUNE_ROWS en src/config.py o aumente spark.driver.memory.")
    base = make_model("mlp", {}, C.OPTUNA_SEED, P["rec"]["n_features"])
    G = C.SPARK_GRID
    grid = (ParamGridBuilder().addGrid(base.stepSize, G["stepSize"]).addGrid(base.maxIter, G["maxIter"])
            .addGrid(base.solver, G["solver"]).build())
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    parcial = C.COMMON / "spark_grid_mlp_parcial.json"
    previas = load_json(parcial)["parcial"] if parcial.exists() else []      # reanudación tras un corte
    rows, cost_per_unit = [], None     # segundos por (iteración x neurona de la primera capa oculta)

    def _units(cfg):
        h = cfg["layers"]
        return cfg["maxIter"] * (h[0] + (h[1] if len(h) > 1 else 0) * 0.05)

    for hidden, pm in product(G["layers"], grid):
        cfg = {"layers": hidden, "stepSize": pm[base.stepSize], "maxIter": pm[base.maxIter], "solver": pm[base.solver]}
        units = _units(cfg)
        hecha = next((r for r in previas if all(r[k] == v for k, v in cfg.items())), None)
        if hecha is not None:
            rows.append(hecha)
            if hecha.get("fit_s"):
                cost_per_unit = max(cost_per_unit or 0.0, hecha["fit_s"] / units)
            continue
        est = cost_per_unit * units if cost_per_unit else None
        if est is not None and est > C.FIT_BUDGET_MIN * 60:
            rows.append({**cfg, "auc_val": None, "fit_s": None, "fit_s_estimado": est, "omitido": True})
            save_json({"parcial": rows}, parcial)
            continue
        m = make_model("mlp", cfg, C.OPTUNA_SEED, P["rec"]["n_features"])
        t0 = time.perf_counter()
        try:
            fm = m.fit(P["tr"])
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            rows.append({**cfg, "auc_val": None, "fit_s": None, "fit_s_estimado": est, "omitido": True,
                         "motivo": "OutOfMemoryError de la JVM"})
            save_json({"parcial": rows}, parcial)
            raise SparkOOM(f"rejilla {cfg}") from e
        dt = time.perf_counter() - t0
        cost_per_unit = dt / units if cost_per_unit is None else max(cost_per_unit, dt / units)
        auc = ev.evaluate(fm.transform(P["va"]))
        rows.append({**cfg, "auc_val": auc, "fit_s": dt, "fit_s_estimado": est, "omitido": False})
        save_json({"parcial": rows}, parcial)
    done = [r for r in rows if not r["omitido"]]
    best = max(done, key=lambda r: r["auc_val"])
    res = {"n_train": P["n_train"], "d": P["rec"]["n_features"], "cores": cores, "resultados": rows,
           "best_params": {k: best[k] for k in ("layers", "stepSize", "maxIter", "solver")}, "best_auc_val": best["auc_val"]}
    save_json(res, out)
    return res


# ------------------------------------------------------------------ escalamiento
def scaling_run(kind: str, n: int, cores: int, max_iter: int, exp: str) -> dict:
    """Ajuste con trabajo fijo (tol = 0: siempre `max_iter` iteraciones) para medir aceleración."""
    key = f"{kind}_n{n}_c{cores}"
    d = run_dir("pyspark", exp, key)
    if is_done(d):
        return load_json(d / "result.json")
    comprobar_tiempo()
    P = prepared(C.P1_ENCODING, str(n), cores)
    spark = get_spark(cores)
    if kind.startswith("mlp"):
        chk = mem_check(P["n_train"], P["rec"]["n_features"])
        if not chk["cabe"]:
            res = {"env": "PySpark", "exp": exp, "model": kind, "n_train": P["n_train"], "cores": cores,
                   "max_iter": max_iter, "omitido": True, "motivo": mem_motivo(chk)}
            save_json(res, d / "result.json")
            return {**res, "_nuevo": True}
    model = make_model(kind, {"maxIter": max_iter, "tol": 0.0, "layers": [C.P1_HIDDEN]}, 0, P["rec"]["n_features"])
    rec: dict = {}
    since = last_stage_id(spark)
    try:
        with bench.stage(rec, "fit", rows=P["n_train"]):
            model.fit(P["tr"])
    except Exception as e:  # noqa: BLE001
        if not _is_oom(e):
            raise
        save_json({"env": "PySpark", "exp": exp, "model": kind, "n_train": P["n_train"], "cores": cores,
                   "max_iter": max_iter, "omitido": True,
                   "motivo": f"la JVM se quedó sin memoria (OutOfMemoryError); memoria máxima {jvm_heap_gb():.1f} GB"},
                  d / "result.json")
        raise SparkOOM(f"{exp} {kind} n={n} p={cores}") from e
    res = {"env": "PySpark", "exp": exp, "model": kind, "n_train": P["n_train"], "cores": cores, "max_iter": max_iter,
           "bench": {**rec, "spark_ui_fit": stage_metrics(spark, since)}}
    save_json(res, d / "result.json")
    return {**res, "_nuevo": True}


# ------------------------------------------------------------------ lectura
def read_csv_count(path) -> dict:
    spark = get_spark(C.SPARK_CORES_MAIN)
    rec: dict = {}
    with bench.stage(rec, "lectura"):
        df = spark.read.csv(path.resolve().as_uri(), header=True)
        df.write.format("noop").mode("overwrite").save()     # materializa todas las columnas sin escribir
    parts = df.rdd.getNumPartitions()
    return {"particiones": parts, **rec}


def read_parquet_count(days: list[int]) -> dict:
    from pyspark.sql import functions as F
    spark = get_spark(C.SPARK_CORES_MAIN)
    rec: dict = {}
    with bench.stage(rec, "lectura"):
        df = spark.read.parquet(C.PARQUET_DIR.as_uri()).filter(F.col("day").isin(days)).select(*C.CAT_COLS, C.TARGET)
        df.write.format("noop").mode("overwrite").save()
    parts = df.rdd.getNumPartitions()
    return {"particiones": parts, **rec}
