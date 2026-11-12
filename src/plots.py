"""Estilo de figuras (el mismo del Entregable 1) y guardado en PNG a 200 dpi."""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import config as C


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#3A506B", "axes.labelcolor": "#0B132B", "text.color": "#0B132B",
        "xtick.color": "#1C2541", "ytick.color": "#1C2541", "axes.grid": True, "grid.color": "#E3E8EE",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 100,
    })


def save(fig, name: str) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", dpi=C.FIG_DPI, bbox_inches="tight")


def env_style(env: str) -> dict:
    return {"color": C.ENV_COLORS[env], "ls": "-" if env == "scikit-learn" else "--"}
