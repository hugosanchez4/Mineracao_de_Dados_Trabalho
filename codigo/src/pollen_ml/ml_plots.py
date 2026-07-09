"""Gráficos de comparação entre modelos de classificação."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_comparacao_metricas(compare: pd.DataFrame, fig_dir: Path) -> Path:
    """Barras agrupadas: F1-macro e acurácia por modelo (média CV ± desvio)."""
    models = compare["modelo"].tolist()
    x = np.arange(len(models))
    width = 0.36

    out = fig_dir / "ml_comparacao_metricas.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - width / 2,
        compare["f1_macro_mean"],
        width,
        yerr=compare["f1_macro_std"],
        label="F1-macro",
        color="steelblue",
        edgecolor="0.2",
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "0.15"},
    )
    ax.bar(
        x + width / 2,
        compare["accuracy_mean"],
        width,
        yerr=compare["accuracy_std"],
        label="Acurácia",
        color="coral",
        edgecolor="0.2",
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "0.15"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (validação cruzada)")
    ax.set_xlabel("")
    n_classes = compare.attrs.get("n_classes", 4)
    ax.axhline(1 / n_classes, color="0.5", ls="--", lw=1, alpha=0.7)
    ax.text(
        0.02,
        0.98,
        "Linha tracejada: acaso (1/n classes)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="0.45",
    )
    ax.set_title("Comparação de modelos — F1-macro e acurácia (K-Fold)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_comparacao_acuracia(compare: pd.DataFrame, fig_dir: Path) -> Path:
    """Barras horizontais só com acurácia (média CV ± desvio)."""
    out = fig_dir / "ml_comparacao_acuracia.png"
    df = compare[["modelo", "accuracy_mean", "accuracy_std"]].copy()
    df = df.sort_values("accuracy_mean", ascending=True)

    plt.figure(figsize=(8, 4.5))
    colors = ["#9aa0a6" if "Dummy" in m else "#4285f4" for m in df["modelo"]]
    bars = plt.barh(df["modelo"], df["accuracy_mean"], color=colors, edgecolor="0.2")
    for bar, (_, row) in zip(bars, df.iterrows()):
        if row["accuracy_std"] > 0:
            plt.errorbar(
                row["accuracy_mean"],
                bar.get_y() + bar.get_height() / 2,
                xerr=row["accuracy_std"],
                fmt="none",
                color="0.15",
                capsize=4,
            )
        plt.text(
            min(row["accuracy_mean"] + row["accuracy_std"] + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{row['accuracy_mean']:.3f}",
            va="center",
            fontsize=9,
        )
    plt.xlim(0, 1.05)
    plt.xlabel("Acurácia (validação cruzada)")
    plt.title("Acurácia por modelo")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_metricas_por_fold(fold_df: pd.DataFrame, fig_dir: Path) -> Path:
    """Linhas por fold: F1-macro e acurácia para cada modelo."""
    out = fig_dir / "ml_metricas_por_fold.png"
    long_df = fold_df.melt(
        id_vars=["fold", "modelo"],
        value_vars=["f1_macro", "accuracy"],
        var_name="métrica",
        value_name="score",
    )
    long_df["métrica"] = long_df["métrica"].map(
        {"f1_macro": "F1-macro", "accuracy": "Acurácia"}
    )

    g = sns.relplot(
        data=long_df,
        x="fold",
        y="score",
        hue="modelo",
        style="métrica",
        kind="line",
        markers=True,
        dashes=False,
        height=5,
        aspect=1.5,
        palette="tab10",
    )
    g.set(ylim=(0, 1.05), xticks=sorted(fold_df["fold"].unique()))
    g.set_axis_labels("Fold (K-Fold)", "Score")
    g.fig.suptitle("Desempenho por fold — comparação entre modelos", y=1.02)
    g.savefig(out, bbox_inches="tight")
    plt.close("all")
    return out


def generate_ml_plots(compare: pd.DataFrame, fold_df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Gera todos os gráficos de comparação ML."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_comparacao_metricas(compare, fig_dir),
        plot_comparacao_acuracia(compare, fig_dir),
        plot_metricas_por_fold(fold_df, fig_dir),
    ]
    return paths
