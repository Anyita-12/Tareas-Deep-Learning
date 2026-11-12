"""Modelos de scikit-learn: líneas base, MLP, búsqueda bayesiana con Optuna y ejecución de corridas.

Cada corrida ajusta el preprocesamiento con la muestra de entrenamiento (días 21 a 28),
elige el umbral de decisión en validación (día 29) y evalúa una sola vez en prueba (día 30).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import bench, config as C, data as D, features as F, metrics as M
from .persist import is_done, load_json, run_dir, save_array, save_json, save_pickle_safe, comprobar_tiempo

warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DATA: dict = {}
_FEAT: dict = {}


# ------------------------------------------------------------------ datos en memoria (caché del proceso)
def get_data(which: str, n: int | None = None):
    key = (which, n)
    if key not in _DATA:
        rec: dict = {}
        with bench.stage(rec, "lectura"):
            df = {"train": D.load_train, "val": D.load_val, "test": D.load_test}[which](n)
        if which == "train":                       # se conserva una sola muestra de entrenamiento
            for k in [k for k in _DATA if k[0] == "train"]:
                del _DATA[k]
        _DATA[key] = (df, rec)
    return _DATA[key]


def get_features(encoding: str, n_train: int, crosses: bool = False) -> dict:
    key = (encoding, n_train, crosses)
    if key not in _FEAT:
        _FEAT.clear()
        tr, rec_read = get_data("train", n_train)
        va, _ = get_data("val", C.VAL_ROWS)
        te, _ = get_data("test", C.TEST_ROWS)
        ytr = D.y_of(tr)
        prep = F.CTRPreprocessor(encoding, crosses=crosses)
        rec = {"lectura_s": rec_read["lectura_s"], "lectura_rows_s": tr.height / max(rec_read["lectura_s"], 1e-9)}
        with bench.stage(rec, "prep", rows=tr.height):
            Xtr = prep.fit_transform(tr, ytr)
        Xva, Xte = prep.transform(va), prep.transform(te)
        rec["n_features"] = prep.n_features()
        rec["matrix_mb"] = F.matrix_bytes(Xtr) / bench.MB
        _FEAT[key] = {"prep": prep, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": D.y_of(va), "Xte": Xte,
                      "yte": D.y_of(te), "rec": rec}
    return _FEAT[key]


# ------------------------------------------------------------------ modelos
def hidden_from_label(lbl: str) -> tuple:
    return tuple(int(x) for x in lbl.split("-"))


def make_model(kind: str, params: dict, seed: int):
    p = dict(params)
    if kind == "lr":
        return SGDClassifier(loss="log_loss", penalty="l2", alpha=p.get("alpha", 1e-5), max_iter=p.get("max_iter", 20),
                             tol=1e-4, random_state=seed)
    if kind == "hgb":
        return HistGradientBoostingClassifier(learning_rate=p.get("learning_rate", 0.1),
                                              max_leaf_nodes=p.get("max_leaf_nodes", 31),
                                              min_samples_leaf=p.get("min_samples_leaf", 100),
                                              l2_regularization=p.get("l2_regularization", 0.0),
                                              max_iter=p.get("max_iter", 300), early_stopping=True,
                                              random_state=seed)
    if kind == "mlp":
        hl = p.get("hidden_layer_sizes", (100,))
        hl = hidden_from_label(hl) if isinstance(hl, str) else tuple(hl)
        return MLPClassifier(hidden_layer_sizes=hl, alpha=p.get("alpha", 1e-4),
                             learning_rate_init=p.get("learning_rate_init", 1e-3),
                             batch_size=p.get("batch_size", 512), random_state=seed, **C.MLP_FIXED)
    if kind == "lr_lbfgs":     # equivalente a LogisticRegression de Spark (P1)
        n = p["n_train"]
        return LogisticRegression(C=1.0 / (C.P1_REG * n), solver="lbfgs", max_iter=C.P1_MAX_ITER, tol=1e-6)
    if kind == "mlp_lbfgs":    # equivalente a MultilayerPerceptronClassifier de Spark (P1)
        return MLPClassifier(hidden_layer_sizes=(C.P1_HIDDEN,), activation="logistic", solver="lbfgs",
                             alpha=C.P1_MLP_ALPHA, max_iter=C.P1_MAX_ITER, tol=1e-6, random_state=seed)
    raise ValueError(kind)


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)


def run(exp: str, kind: str, encoding: str, size: str, seed: int, params: dict | None = None,
        weighted: bool = False, crosses: bool = False, save_preds: bool = True, tag: str = "") -> dict:
    n_train = C.SIZES[size]
    key = f"{kind}_{encoding}_{size}_s{seed:02d}" + ("_w" if weighted else "") + ("_x" if crosses else "") + tag
    d = run_dir("sklearn", exp, key)
    if is_done(d):
        return load_json(d / "result.json")
    comprobar_tiempo()
    feats = get_features(encoding, n_train, crosses)
    params = dict(params or {})
    if kind == "lr_lbfgs":
        params["n_train"] = feats["Xtr"].shape[0]
    model = make_model(kind, params, seed)
    rec = dict(feats["rec"])
    sw = compute_sample_weight("balanced", feats["ytr"]) if weighted else None
    with bench.stage(rec, "fit", rows=feats["Xtr"].shape[0]):
        if sw is not None:
            model.fit(feats["Xtr"], feats["ytr"], sample_weight=sw)
        else:
            model.fit(feats["Xtr"], feats["ytr"])
    p_val = _proba(model, feats["Xva"])
    with bench.stage(rec, "predict", rows=feats["Xte"].shape[0]):
        p_test = _proba(model, feats["Xte"])
    thr = M.best_threshold(feats["yva"], p_val)
    res = {
        "env": "scikit-learn", "exp": exp, "model": kind, "encoding": encoding, "size": size,
        "n_train": int(feats["Xtr"].shape[0]), "seed": seed, "weighted": weighted, "crosses": crosses,
        "params": {k: v for k, v in params.items() if k != "n_train"},
        "n_iter": int(np.max(np.atleast_1d(getattr(model, "n_iter_", 0)))),
        "loss_curve": list(map(float, getattr(model, "loss_curve_", []) or [])),
        "val_curve": list(map(float, getattr(model, "validation_scores_", []) or [])),
        "val": {"roc_auc": float(roc_auc_score(feats["yva"], p_val))},
        "test": M.binary_metrics(feats["yte"], p_test, thr),
        "bench": rec,
    }
    res["bench"]["fit_s_per_iter"] = rec["fit_s"] / max(res["n_iter"], 1)
    if save_preds:
        save_array(p_test, d / "p_test.npy")
        save_array(p_val, d / "p_val.npy")
    res["run_dir"] = d.relative_to(C.RUNS).as_posix()
    # Modelo ajustado (pickle) y preprocesador (uno por codificación y tamaño, compartido entre semillas).
    # Se guardan ANTES de result.json: una corrida con result.json siempre tiene su modelo en disco.
    arte = {}
    if save_pickle_safe(model, d / "model.pkl", f"el modelo {key}"):
        arte["modelo"] = "model.pkl"
    pk = C.RESULTS / "models" / "sklearn_prep" / f"{encoding}_{size}{'_x' if crosses else ''}.pkl"
    if pk.exists() or save_pickle_safe(feats["prep"], pk, f"el preprocesador {encoding} {size}"):
        arte["preprocesador"] = pk.relative_to(C.RESULTS).as_posix()
    res["artefactos"] = arte
    save_json(res, d / "result.json")
    return res


# ------------------------------------------------------------------ Optuna
def tune(kind: str, encoding: str, n_trials: int, n_train: int = C.TUNE_ROWS) -> dict:
    """Búsqueda bayesiana (TPE) maximizando ROC AUC en validación temporal (día 29)."""
    out = C.COMMON / f"optuna_{kind}_{encoding}.json"
    if out.exists():
        return load_json(out)
    tr = D.load_train(n_train)
    va = D.load_val()
    ytr, yva = D.y_of(tr), D.y_of(va)
    prep = F.CTRPreprocessor(encoding)
    Xtr = prep.fit_transform(tr, ytr)
    Xva = prep.transform(va)

    def objective(trial: optuna.Trial) -> float:
        if kind == "mlp":
            lo, hi = C.MLP_SPACE["alpha"]
            params = {"hidden_layer_sizes": trial.suggest_categorical(
                          "hidden_layer_sizes", ["-".join(map(str, h)) for h in C.MLP_SPACE["hidden_layer_sizes"]]),
                      "alpha": trial.suggest_float("alpha", lo, hi, log=True),
                      "learning_rate_init": trial.suggest_float("learning_rate_init", *C.MLP_SPACE["learning_rate_init"], log=True),
                      "batch_size": trial.suggest_categorical("batch_size", C.MLP_SPACE["batch_size"])}
        elif kind == "lr":
            params = {"alpha": trial.suggest_float("alpha", 1e-7, 1e-3, log=True)}
        elif kind == "hgb":
            params = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                      "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 1000, log=True),
                      "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True)}
        else:
            raise ValueError(kind)
        m = make_model(kind, params, C.OPTUNA_SEED)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        trial.set_user_attr("fit_s", time.perf_counter() - t0)
        trial.set_user_attr("n_iter", int(np.max(getattr(m, "n_iter_", 0))))
        return float(roc_auc_score(yva, _proba(m, Xva)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials)
    df = study.trials_dataframe()
    df.to_csv(C.COMMON / f"optuna_{kind}_{encoding}_trials.csv", index=False)
    best = {"kind": kind, "encoding": encoding, "n_train": int(tr.height), "n_val": int(va.height),
            "n_trials": n_trials, "best_value": study.best_value, "best_params": study.best_params}
    save_pickle_safe(study, C.COMMON / f"optuna_{kind}_{encoding}_study.pkl", "el estudio de Optuna")
    save_json(best, out)
    return best


def best_params(kind: str, encoding: str) -> dict:
    p = C.COMMON / f"optuna_{kind}_{encoding}.json"
    return load_json(p)["best_params"] if p.exists() else {}
