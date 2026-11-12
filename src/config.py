"""Configuración central del Proyecto Integrador (MLP para predicción de clics en Avazu).

Todas las decisiones del experimento viven aquí. Se declararon antes de entrenar
(capítulo 1 del informe) y no se modifican después de ver los resultados de prueba.

Variables de entorno:
    AVAZU_SMOKE=1      activa la prueba de humo (datos sintéticos y tamaños pequeños)
    AVAZU_RESULTS      carpeta de resultados alternativa
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------ modo
SMOKE = os.environ.get("AVAZU_SMOKE", "0") == "1"

# ------------------------------------------------------------------ rutas
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ("data_smoke" if SMOKE else "data")
RAW_DIR = DATA / "raw"                       # train.gz de Kaggle (ignorado por git)
PARQUET_DIR = DATA / "parquet" / "avazu"     # Parquet particionado por día (ignorado por git)
INTERIM = DATA / "interim"                   # cachés (ignorado por git)
SPARK_TMP = DATA / "spark_tmp"               # archivos temporales de Spark (ignorado por git)
SUMS = DATA / "SHA256SUMS.txt"               # huellas de verificación (se versiona)
MANIFEST = DATA / "manifest.json"            # filas por día y huellas del Parquet (se versiona)

RESULTS = Path(os.environ.get("AVAZU_RESULTS", str(ROOT / ("results_smoke" if SMOKE else "results"))))
COMMON = RESULTS / "common"
SUMMARY = RESULTS / "summary"
FIGURES = RESULTS / "figures"
EDA_OUT = RESULTS / "eda"
LOGS = RESULTS / "logs"
RUNS = RESULTS / "runs"

RAW_CANDIDATES = ["train.gz", "train.csv.gz", "train.csv", "train"]

# ------------------------------------------------------------------ datos
TARGET = "click"
ORIGINAL_CAT = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
    "app_category", "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
TIME_COLS = ["hora", "dia_semana", "franja"]           # derivadas de `hour` (YYMMDDHH)
CAT_COLS = ORIGINAL_CAT + TIME_COLS                     # 24 variables categóricas de entrada
CROSSES = [("site_id", "device_type"), ("app_id", "device_type")]   # ablación de cruces

TRAIN_DAYS = list(range(21, 29))   # 21 a 28 de octubre de 2014
VAL_DAY = 29
TEST_DAY = 30
FRANJAS = {"madrugada": range(0, 6), "mañana": range(6, 12), "tarde": range(12, 18), "noche": range(18, 24)}

# Muestras escalonadas, anidadas: la muestra de n filas contiene a todas las menores.
if SMOKE:
    SIZES = {"5k": 5_000, "20k": 20_000, "60k": 60_000}
    MAIN_SIZE = "20k"
    VAL_ROWS = 10_000
    TUNE_ROWS = 10_000
    SPARK_TUNE_ROWS = 3_000
    SPARK_MLP_SIZE = "5k"
    SCALING_ROWS = 20_000
    WEAK_BASE_ROWS = 5_000
else:
    SIZES = {"100k": 100_000, "1M": 1_000_000, "3M": 3_000_000, "10M": 10_000_000}
    MAIN_SIZE = "1M"
    VAL_ROWS = 1_000_000          # muestra del día 29 para ajuste de hiperparámetros y umbral
    TUNE_ROWS = 500_000           # muestra de entrenamiento para Optuna
    SPARK_TUNE_ROWS = 20_000      # muestra para la rejilla del MLP de Spark (entrada densa de 2^12)
    SPARK_MLP_SIZE = "100k"       # tamaño de las 5 semillas del MLP de Spark
    SCALING_ROWS = 1_000_000      # tamaño fijo del escalamiento fuerte
    WEAK_BASE_ROWS = 250_000      # filas por núcleo en el escalamiento débil
TEST_ROWS = None                  # None = todo el día 30

SAMPLE_KEY = "u"                  # clave uniforme [0, 1) por fila, fija en el Parquet

# ------------------------------------------------------------------ preprocesamiento
RARE_MIN_COUNT = 10               # categorías con menos de 10 apariciones en entrenamiento -> __RARE__
RARE_TOKEN = "__RARE__"
LOW_CARD_MAX = 64                 # cardinalidad <= 64 en entrenamiento: one-hot en las estrategias densas
HASH_DIMS = {"hash_12": 2**12, "hash_14": 2**14, "hash_16": 2**16}
ENCODINGS = ["hash_12", "hash_14", "hash_16", "freq", "target"]
ENCODING_LABELS = {"hash_12": "Hashing 2^12", "hash_14": "Hashing 2^14", "hash_16": "Hashing 2^16",
                   "freq": "Frecuencia", "target": "Target encoding OOF"}
TARGET_ENCODER_CV = 5

# ------------------------------------------------------------------ inferencia (declarado a priori)
PRIMARY_METRIC = "roc_auc"
DELTA_AUC = 0.005                 # margen de relevancia práctica en ROC AUC
ALPHA = 0.05
SEEDS = [0, 1, 2, 3, 4]           # mínimo de 5 repeticiones
SEEDS_BY_SIZE = ({"5k": 2, "20k": 2, "60k": 1} if SMOKE else {"100k": 5, "1M": 5, "3M": 3, "10M": 3})
N_BOOT = 200 if SMOKE else 1000
BOOT_MAX_ROWS = 500_000           # el bootstrap usa una submuestra fija del día 30 de este tamaño
ECE_BINS = 15                     # contenedores de igual frecuencia
SAMPLE_SEED = 1234

# ------------------------------------------------------------------ scikit-learn
OPTUNA_TRIALS = 6 if SMOKE else 30
OPTUNA_TRIALS_BASELINES = 4 if SMOKE else 15
OPTUNA_SEED = 100
MLP_SPACE = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (256, 128)],
    "alpha": (1e-5, 1e-1),              # escala logarítmica
    "learning_rate_init": (1e-4, 1e-2),  # escala logarítmica
    "batch_size": [256, 512, 1024, 2048],
}
MLP_FIXED = {"activation": "relu", "solver": "adam", "early_stopping": True, "validation_fraction": 0.1,
             "n_iter_no_change": 5, "tol": 1e-4, "max_iter": 30 if SMOKE else 100}
MLP_TUNE_ENCODING = "hash_14"
DEPTH_ABLATION = [(100,), (100, 50), (256, 128), (256, 128, 64)]

# P1 (punto de cruce): el mismo modelo en ambos entornos, con la misma entrada (hashing 2^12),
# la misma arquitectura, la misma activación sigmoide y el mismo optimizador (L-BFGS).
P1_ENCODING = "hash_12"
P1_HIDDEN = 10 if SMOKE else 100
P1_MAX_ITER = 10 if SMOKE else 100
P1_REG = 1e-5                     # regularización L2 de la regresión logística
P1_MLP_ALPHA = 0.0                # el MLP de Spark no tiene regularización

# ------------------------------------------------------------------ PySpark
SPARK_HASH_DIM = 2**12            # el MLP de Spark trabaja con matrices densas: d = 2^12
SPARK_CONF = {
    # Memoria de la JVM de Spark; se puede cambiar sin editar el código con la variable AVAZU_SPARK_MEMORY
    # (por ejemplo, $env:AVAZU_SPARK_MEMORY = "12g" en PowerShell si el equipo tiene menos RAM libre).
    "spark.driver.memory": os.environ.get("AVAZU_SPARK_MEMORY", "6g" if SMOKE else "16g"),
    "spark.driver.maxResultSize": "4g",
    "spark.sql.shuffle.partitions": "8" if SMOKE else "32",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.ui.showConsoleProgress": "false",
    "spark.sql.adaptive.enabled": "true",
    # Uniones con tablas auxiliares: el plan estático nunca copia tablas a cada tarea (broadcast), porque las
    # de device_ip y device_id pueden ser grandes; AQE sí lo hace en ejecución cuando el tamaño real es pequeño.
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.sql.adaptive.autoBroadcastJoinThreshold": "32MB",
    # La etapa de Spark encadena miles de etapas; la Spark UI conserva solo las recientes para no
    # acumular memoria en el driver (las métricas se leen justo después de cada ajuste).
    "spark.ui.retainedTasks": "20000",
    # Si la JVM se queda sin memoria, termina de inmediato en lugar de quedar colgada; el orquestador
    # registra la corrida como omitida y continúa en un proceso nuevo.
    "spark.driver.extraJavaOptions": "-XX:+ExitOnOutOfMemoryError",
    "spark.sql.ui.retainedExecutions": "200",
}
SPARK_CORES_MAIN = min(8, os.cpu_count() or 1)
SPARK_CORES_SCALING = [p for p in (1, 2, 4, 8) if p <= (os.cpu_count() or 1)]
SPARK_GRID = ({"layers": [[10], [20]], "stepSize": [0.1], "maxIter": [5, 10], "solver": ["l-bfgs", "gd"]}
              if SMOKE else
              {"layers": [[10], [50], [100], [128, 64]],    # capas ocultas; la entrada es d y la salida 2
               "stepSize": [0.1, 0.01],
               "maxIter": [100, 200],
               "solver": ["l-bfgs", "gd"]})
SPARK_GBT = {"maxIter": 10 if SMOKE else 50, "maxDepth": 5, "stepSize": 0.1}
SPARK_LR = {"maxIter": 20 if SMOKE else 100, "regParam": 1e-5, "elasticNetParam": 0.0}
# Antes de cada ajuste del MLP de Spark se mide la memoria libre de la JVM (tras recolectar basura) y se exige
# que quepa SPARK_MLP_MEM_FACTOR veces la entrada densa apilada (n x d x 8 bytes). Si no cabe, Spark no puede
# guardarla en caché y la reconstruye en cada iteración de L-BFGS, lo que vuelve el ajuste impracticable.
SPARK_MLP_MEM_FACTOR = 1.0
# Límite de tiempo opcional (lo fija `src.todo --plan`): al alcanzarlo, la etapa deja de iniciar corridas nuevas.
_lim = os.environ.get("AVAZU_LIMITE")
LIMITE = datetime.fromisoformat(_lim) if _lim else None
FIT_BUDGET_MIN = 5 if SMOKE else 90   # se omite un ajuste cuya duración estimada supere este presupuesto

# ------------------------------------------------------------------ interpretabilidad
LIME_TOP_K_CATEGORIES = 30        # categorías por variable en el espacio interpretable (resto -> __RARE__)
LIME_N_CASES = 10                 # falsos positivos y falsos negativos con mayor confianza errónea (cada uno)
LIME_SEEDS = list(range(10))
LIME_NUM_SAMPLES = [1000, 5000]
LIME_KERNEL_FACTORS = [0.5, 0.75, 1.5]   # ancho = factor * sqrt(número de variables)
LIME_TOP_FEATURES = 5
SHAP_BACKGROUND = 100

# ------------------------------------------------------------------ costo en la nube (referencia)
CLOUD_PRICE_USD_H = {"AWS m7i.2xlarge (8 vCPU, 32 GiB)": 0.4032}

# ------------------------------------------------------------------ figuras
FIG_DPI = 200
ENV_COLORS = {"scikit-learn": "#3A506B", "PySpark": "#5BC0BE"}
MODEL_COLORS = {"Regresión logística": "#5BC0BE", "Gradient boosting": "#3A506B", "MLP": "#1C2541"}
ENCODING_COLORS = {"hash_12": "#9DB4C0", "hash_14": "#5C7A8C", "hash_16": "#253237",
                   "freq": "#5BC0BE", "target": "#3A506B"}
ACCENT = "#C8553D"
BAR = "#4C78A8"

MODEL_NAMES = {"lr": "Regresión logística", "hgb": "Gradient boosting", "gbt": "Gradient boosting",
               "mlp": "MLP"}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PARQUET_DIR.parent, INTERIM, SPARK_TMP, COMMON, SUMMARY, FIGURES, EDA_OUT, LOGS, RUNS):
        d.mkdir(parents=True, exist_ok=True)


def column_groups() -> dict:
    """Grupos de baja y alta cardinalidad fijados en el EDA con la ventana de entrenamiento completa."""
    p = COMMON / "column_groups.json"
    if not p.exists():
        raise FileNotFoundError(
            "Falta results/common/column_groups.json. Ejecute primero `python -m src.runner prepare` "
            "o el notebook docs/02_eda.ipynb.")
    return json.loads(p.read_text(encoding="utf-8"))
