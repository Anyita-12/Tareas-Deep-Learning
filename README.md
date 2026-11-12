# Predicción de clics en anuncios móviles con MLP: scikit-learn frente a PySpark

Proyecto Integrador de Deep Learning (sección 2.12), Universidad del Norte. Profesor: Dr. Lihki Rubio.

**Autores:** Anyely Sanjuan Benavides (Msc. Analítica de Datos) y Juan David Díaz Fernández (Msc. Ingeniería Mecánica), Universidad del Norte.

**Informe (Jupyter Book):** https://anyita-12.github.io/Tareas-Deep-Learning/

## Descripción

Construimos y comparamos clasificadores para predecir si un usuario hará clic en un anuncio móvil (`click = 1`) con el conjunto de datos *Avazu Click-Through Rate Prediction*. El modelo central es un perceptrón multicapa (MLP), entrenado en scikit-learn y en PySpark con la misma lógica de preprocesamiento, y comparado con la regresión logística y los árboles potenciados. El protocolo usa una partición temporal (entrenamiento días 21 a 28, validación día 29, prueba día 30), cinco semillas por configuración y pruebas estadísticas (bootstrap, DeLong, McNemar y $t$ pareada con Holm). La interpretabilidad local se analiza con LIME y se contrasta con SHAP.

Preguntas de investigación:

- **P1:** ¿a partir de qué volumen PySpark entrena más rápido que scikit-learn sin degradar la calidad?
- **P2:** ¿cómo afecta la codificación de alta cardinalidad (hashing, frecuencia, target encoding)?
- **P3:** ¿mejora el MLP de forma significativa a las líneas base?
- **P4:** ¿son estables y fieles las explicaciones de LIME?

## Estructura del repositorio

```
data/                 datos (no se versionan) + SHA256SUMS.txt y manifest.json (sí se versionan)
docs/                 informe en Jupyter Book: capítulos 01 a 11 (.ipynb) y referencias
src/                  código del experimento
  config.py           todas las decisiones del experimento
  data.py             conversión a Parquet, manifiesto, huellas y muestreo
  features.py         preprocesamiento de scikit-learn (raras, hashing, frecuencia, target encoding)
  spark_models.py     sesión de Spark, preprocesamiento equivalente, modelos y métricas de la Spark UI
  sk_models.py        modelos de scikit-learn y búsqueda con Optuna
  metrics.py          métricas, bootstrap, DeLong y McNemar
  bench.py            tiempo y memoria pico por etapa
  explain.py          LIME, SHAP e importancia por permutación
  eda.py              consultas del EDA con DuckDB
  runner.py           orquestador reanudable de todas las etapas
results/              resultados por corrida (JSON), tablas y figuras
requirements.txt      dependencias con versiones fijadas
requirements-docs.txt dependencias para compilar el libro
environment.yml       alternativa con conda
```

## Instalación (Windows, PowerShell)

Requisitos: Python 3.12, Java 17, los binarios de Hadoop para Windows (ver abajo) y Git (o GitHub Desktop).

```powershell
winget install EclipseAdoptium.Temurin.17.JDK      # Java para PySpark (cerrar y abrir la terminal después)
java -version

cd ruta\a\Tareas-Deep-Learning
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-docs.txt
```

Si PowerShell bloquea la activación del entorno: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Hadoop para Windows (necesario para PySpark).** Spark lee los archivos a través de Hadoop, que en Windows requiere `winutils.exe` y `hadoop.dll`. Se descargan de la carpeta `hadoop-3.3.6/bin` del repositorio [cdarlint/winutils](https://github.com/cdarlint/winutils) y se copian en `C:\hadoop\bin`:

```powershell
mkdir C:\hadoop\bin
dir C:\hadoop\bin        # deben aparecer winutils.exe y hadoop.dll
```

El código detecta `C:\hadoop` automáticamente; no hace falta configurar `HADOOP_HOME`.

## Datos

1. Iniciar sesión en Kaggle, abrir https://www.kaggle.com/c/avazu-ctr-prediction/data y aceptar las reglas de la competencia.
2. Descargar `train.gz` (unos 1.1 GB) y colocarlo, sin descomprimir, en `data/raw/train.gz`.

Los datos no se redistribuyen en este repositorio. `data/SHA256SUMS.txt` y `data/manifest.json` registran sus huellas, las filas por día y la huella del conjunto de prueba.

## Ejecución

Con el entorno activado, desde la raíz del repositorio:

```powershell
python -m src.runner smoke      # prueba de humo con datos sintéticos (unos minutos); verifica la instalación
python -m src.runner prepare    # CSV -> Parquet por día, manifiesto, huellas y grupos de variables
python -m src.runner sklearn    # Optuna, codificaciones (P2), líneas base (P3), ablaciones y escalado (P1)
python -m src.runner spark      # rejilla del MLP, modelos, escalado (P1) y escalamiento fuerte y débil
python -m src.runner bench      # lectura del CSV y del Parquet con pandas, DuckDB, Polars y Spark
python -m src.runner explain    # LIME, SHAP e importancia por permutación (P4)
python -m src.runner status     # avance, desde otra terminal
```

### Ejecución automática de principio a fin

Para dejar todo corriendo sin supervisión (etapas, cuadernos 01 a 11 e informe):

```powershell
python -m src.todo
```

Verifica Java, datos, Hadoop y disco; guarda las versiones de las librerías; ejecuta `runner all` con hasta 5 reintentos; ejecuta los cuadernos en orden; compila el informe y escribe `results/logs/TERMINADO.txt` o `results/logs/FALLO.txt`. El registro queda en `results/logs/todo.log`. Si se interrumpe, se vuelve a ejecutar el mismo comando. No publica en GitHub.

Con una fecha de entrega fija, `--plan` asigna una hora límite a cada etapa. Al alcanzarla, la etapa deja de iniciar corridas nuevas (las terminadas se conservan) y pasa a la siguiente. El orden es prepare, sklearn, spark, bench, explain y ablaciones. Lo que queda sin ejecutar se registra en `results/common/limite_tiempo.json` y en la sección 10.9 del informe:

```powershell
python -m src.todo --plan "sklearn=2026-11-11 01:00;spark=2026-11-11 06:30;ablaciones=2026-11-11 08:30"
```

**Dos equipos.** `--etapas` elige qué etapas corre cada equipo (el orden es fijo: prepare, sklearn, explain, spark, escala_sklearn, bench, ablaciones) y `--sin-informe` omite los cuadernos. Lo que depende del hardware (spark, escala_sklearn, bench) se corre como un bloque en un mismo equipo. Después, en el equipo que compila el libro:

```powershell
python -m src.fusionar "ruta\a\results_del_otro_equipo"   # toma el bloque del equipo con más corridas de PySpark
python -m src.todo --etapas ninguna                          # solo cuadernos e informe
```

Todas las etapas son reanudables: si se interrumpen (corte de luz, reinicio), al volver a ejecutarlas omiten las corridas terminadas. Los ajustes cuya duración estimada supera 90 minutos, o cuya entrada densa del MLP de Spark no cabe en la memoria libre de la JVM, se omiten y quedan registrados con su motivo.

**Memoria de Spark.** La JVM de Spark usa 16 GB. Durante la etapa `spark` conviene cerrar el navegador y otros programas pesados. Si el equipo tiene menos memoria libre, se puede reducir sin editar el código:

```powershell
$env:AVAZU_SPARK_MEMORY = "12g"
python -m src.runner spark
```

Cada combinación de codificación, tamaño y número de núcleos se entrena en una JVM nueva, para que la memoria retenida por los ajustes anteriores no se acumule. Si aun así la JVM se queda sin memoria, termina de inmediato, la corrida queda registrada como omitida y la etapa continúa sola. Para reintentar una corrida omitida, se borra su carpeta en `results/runs/pyspark/` y se vuelve a ejecutar la etapa.

Después se ejecutan en orden los notebooks de `docs/` (01 a 11), que leen los resultados de `results/`.

## Qué se guarda de cada entrenamiento

Cada corrida escribe sus archivos de forma atómica (temporal, volcado a disco y renombrado), de modo que un corte de luz deja el archivo anterior intacto o el nuevo completo, nunca uno a medias. Una corrida con `result.json` ya tiene guardado todo lo demás.

| Archivo | Contenido |
|---|---|
| `result.json` | Métricas, curvas de pérdida, tiempos, parámetros y la lista de artefactos |
| `p_val.npy`, `p_test.npy` | Probabilidades en validación y prueba |
| `model.pkl` (scikit-learn) | Modelo ajustado |
| `results/models/sklearn_prep/*.pkl` | Preprocesador ajustado, uno por codificación y tamaño |
| `modelo_spark/` y `parametros.pkl` (PySpark) | Modelo nativo de Spark y sus pesos o coeficientes en NumPy |
| `common/optuna_*_study.pkl` | Estudio de Optuna completo |
| `summary/explain_objetos.pkl` | Modelo, preprocesador, espacio interpretable, muestras, casos, LIME, SHAP y permutación |

Los `.pkl` dependen de las versiones de las librerías. Para poder abrirlos más adelante, guarda las versiones exactas: `pip freeze > results\requirements_freeze.txt`. Los `.pkl`, `.npy` y `modelo_spark/` no se suben a GitHub (ver `.gitignore`); consérvalos en un respaldo aparte. Para recargar un modelo de scikit-learn: `from src.persist import load_pickle; m = load_pickle("results/runs/.../model.pkl")`.

## Compilar y publicar el informe

```powershell
jupyter-book build docs/ --all
ghp-import -n -p -f docs/_build/html
```

GitHub Pages se activa una vez en *Settings → Pages*, con la rama `gh-pages` como fuente.

## Reproducibilidad

- Semillas de entrenamiento 0 a 4; Optuna con semilla 100; bootstrap con semilla 1234.
- Muestras anidadas definidas por una clave `u = hash(id)` fija en el Parquet, idénticas en scikit-learn y PySpark.
- Todas las transformaciones se ajustan solo con entrenamiento; el conjunto de prueba se verifica con su huella antes de cada evaluación.
- El anexo A del informe registra versiones, hardware, configuración de Spark y duración de las etapas.

## Flujo de trabajo del equipo

El proyecto es en pareja. Anyely Sanjuan Benavides administra el repositorio, ejecuta el experimento y publica el informe; ambos autores revisan el código, los resultados y la redacción.

## Uso de inteligencia artificial

Usamos Claude (Anthropic) como apoyo en la programación, el análisis de datos y la redacción. Los autores revisamos, verificamos y aprobamos todo el contenido, y asumimos la responsabilidad por él.

## Referencia del conjunto de datos

Avazu. (2014). *Click-Through Rate Prediction* [Competencia y conjunto de datos]. Kaggle. https://www.kaggle.com/c/avazu-ctr-prediction
