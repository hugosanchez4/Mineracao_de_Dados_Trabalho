"""Funções de visualização para EDA morfométrica de pólen."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

KEY_FEATS_DEFAULT = [
    "eixo_polar_um",
    "eixo_equatorial_um",
    "circularidade",
    "espessura_exina_um",
    "comprimento_ectoabertura_um",
]


def _grid_shape(n: int) -> tuple[int, int]:
    cols = 4
    rows = math.ceil(n / cols)
    return rows, cols


def plot_missing_por_feature(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    missing_pct = (df[features].isna().mean() * 100).sort_values(ascending=True)
    out = fig_dir / "missing_por_feature.png"
    plt.figure(figsize=(10, max(5, len(features) * 0.35)))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, orient="h", color="steelblue")
    plt.xlabel("% ausente")
    plt.ylabel("Feature")
    plt.title("Valores ausentes por feature (dados brutos)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_missing_por_especie(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    miss = df.groupby("species")[features].apply(lambda g: g.isna().mean() * 100)
    out = fig_dir / "missing_por_especie.png"
    plt.figure(figsize=(14, max(4, len(miss) * 0.8)))
    sns.heatmap(miss, annot=True, fmt=".0f", cmap="YlOrRd", vmin=0, vmax=100)
    plt.title("% ausente por espécie e feature")
    plt.ylabel("Espécie")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_correlacao_features(corr: pd.DataFrame, fig_dir: Path) -> Path:
    out = fig_dir / "correlacao_features.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Matriz de correlação — features morfométricas")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_correlacao_clustermap(corr: pd.DataFrame, fig_dir: Path) -> Path:
    out = fig_dir / "correlacao_clustermap.png"
    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(12, 10),
        dendrogram_ratio=0.12,
    )
    g.fig.suptitle("Clustermap de correlação entre features", y=1.02)
    g.savefig(out, bbox_inches="tight")
    plt.close("all")
    return out


def _feature_grid_boxplots(df: pd.DataFrame, features: list[str], fig_dir: Path, filename: str, title: str) -> Path:
    rows, cols = _grid_shape(len(features))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat = np.array(axes).flatten()
    palette = sns.color_palette("husl", n_colors=df["species"].nunique())
    for i, col in enumerate(features):
        ax = axes_flat[i]
        sns.boxplot(data=df, x="species", y=col, hue="species", ax=ax, palette=palette, legend=False)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    out = fig_dir / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_boxplots_todas_features(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    return _feature_grid_boxplots(
        df, features, fig_dir, "boxplots_todas_features.png", "Box plots — todas as features por espécie"
    )


def plot_histogramas_todas_features(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    rows, cols = _grid_shape(len(features))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten()
    for i, col in enumerate(features):
        ax = axes_flat[i]
        for sp, sub in df.groupby("species"):
            ax.hist(sub[col], bins=10, alpha=0.5, label=sp, density=True)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Histogramas — todas as features por espécie", fontsize=13, y=1.01)
    plt.tight_layout()
    out = fig_dir / "histogramas_todas_features.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_violinos_todas_features(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    rows, cols = _grid_shape(len(features))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat = np.array(axes).flatten()
    palette = sns.color_palette("husl", n_colors=df["species"].nunique())
    for i, col in enumerate(features):
        ax = axes_flat[i]
        sns.violinplot(data=df, x="species", y=col, hue="species", ax=ax, palette=palette, cut=0, inner="quartile", legend=False)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Violin plots — todas as features por espécie", fontsize=13, y=1.01)
    plt.tight_layout()
    out = fig_dir / "violinos_todas_features.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_ridgelines_features(df: pd.DataFrame, kw: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    top = [f for f in kw.sort_values("p_value")["feature"].head(6) if f in features]
    if not top:
        top = features[:6]
    out = fig_dir / "ridgelines_features.png"
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 2.2 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]
    species_list = sorted(df["species"].unique())
    palette = dict(zip(species_list, sns.color_palette("husl", len(species_list))))
    for ax, col in zip(axes, top):
        for sp in species_list:
            sub = df.loc[df["species"] == sp, col]
            if len(sub) < 2:
                continue
            sns.kdeplot(sub, ax=ax, fill=True, alpha=0.4, color=palette[sp], label=sp, linewidth=1)
        ax.set_ylabel(col, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Valor")
    fig.suptitle("Densidade (KDE) — features mais discriminantes (Kruskal-Wallis)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_histogramas_especies(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    key_feats = [f for f in KEY_FEATS_DEFAULT if f in features]
    rows, cols = _grid_shape(len(key_feats))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten()
    for i, col in enumerate(key_feats):
        ax = axes_flat[i]
        for sp, sub in df.groupby("species"):
            ax.hist(sub[col], bins=12, alpha=0.5, label=sp, density=True)
        ax.set_title(f"Histograma — {col}")
        ax.set_xlabel(col)
        ax.legend(fontsize=7, ncol=2)
    for j in range(len(key_feats), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    out = fig_dir / "histogramas_especies.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_boxplots_especies(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    key_feats = [f for f in KEY_FEATS_DEFAULT if f in features]
    plt.figure(figsize=(14, 6))
    plot_df = df.melt(id_vars="species", value_vars=key_feats, var_name="feature", value_name="value")
    sns.boxplot(data=plot_df, x="feature", y="value", hue="species")
    plt.xticks(rotation=30, ha="right")
    plt.title("Box plots por espécie (features principais)")
    plt.tight_layout()
    out = fig_dir / "boxplots_especies.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_kruskal_wallis_significancia(kw: pd.DataFrame, fig_dir: Path) -> Path:
    kw_plot = kw.copy()
    kw_plot["neg_log10_p"] = -np.log10(kw_plot["p_value"].clip(lower=1e-300))
    kw_plot = kw_plot.sort_values("neg_log10_p", ascending=True)
    alpha_line = -np.log10(0.05)
    out = fig_dir / "kruskal_wallis_significancia.png"
    plt.figure(figsize=(10, max(5, len(kw_plot) * 0.35)))
    colors = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in kw_plot["p_value"]]
    plt.barh(kw_plot["feature"], kw_plot["neg_log10_p"], color=colors)
    plt.axvline(alpha_line, color="red", linestyle="--", linewidth=1, label="α = 0,05")
    plt.xlabel("-log₁₀(p-value)")
    plt.ylabel("Feature")
    plt.title("Kruskal-Wallis — significância entre espécies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_scatter_eixos(df: pd.DataFrame, fig_dir: Path) -> Path:
    out = fig_dir / "scatter_eixos.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="eixo_polar_um",
        y="eixo_equatorial_um",
        hue="species",
        style="view" if "view" in df.columns else None,
        alpha=0.8,
    )
    plt.title("Eixo polar vs equatorial")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_scatter_morfologia(df: pd.DataFrame, fig_dir: Path) -> Path:
    pairs = [
        ("area_um2", "perimeter_um", "Área vs perímetro"),
        ("area_um2", "feret_um", "Área vs Feret"),
        ("circularidade", "solidity", "Circularidade vs solidez"),
        ("comprimento_espinho_um", "largura_espinho_um", "Espinho: comprimento vs largura"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (x, y, title) in zip(axes.flatten(), pairs):
        if x not in df.columns or y not in df.columns:
            ax.set_visible(False)
            continue
        sns.scatterplot(data=df, x=x, y=y, hue="species", ax=ax, alpha=0.75, s=50)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Relações morfológicas bivariadas", fontsize=13)
    plt.tight_layout()
    out = fig_dir / "scatter_morfologia.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_scatter_aberturas(df: pd.DataFrame, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        ("comprimento_ectoabertura_um", "largura_ectoabertura_um", "Ectoabertura"),
        ("comprimento_endoabertura_um", "largura_endoabertura_um", "Endoabertura"),
    ]
    for ax, (x, y, title) in zip(axes, panels):
        if x not in df.columns or y not in df.columns:
            ax.set_visible(False)
            continue
        sub = df.dropna(subset=[x, y])
        sns.scatterplot(data=sub, x=x, y=y, hue="species", ax=ax, alpha=0.8, s=55)
        ax.set_title(f"{title} (n={len(sub)})")
        ax.legend(fontsize=7)
    fig.suptitle("Comprimento vs largura das aberturas", fontsize=13)
    plt.tight_layout()
    out = fig_dir / "scatter_aberturas.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_pairplot_top_features(df: pd.DataFrame, kw: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    top = [f for f in kw.sort_values("p_value")["feature"].head(5) if f in features]
    if len(top) < 2:
        top = features[: min(5, len(features))]
    out = fig_dir / "pairplot_top_features.png"
    g = sns.pairplot(df, vars=top, hue="species", corner=True, plot_kws={"alpha": 0.7, "s": 35}, diag_kind="kde")
    g.fig.suptitle("Pairplot — 5 features mais discriminantes", y=1.02)
    g.savefig(out, bbox_inches="tight")
    plt.close("all")
    return out


def _fit_pca_full(df: pd.DataFrame, features: list[str]) -> tuple[PCA, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    n_comp = min(len(features), len(df))
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    return pca, coords, scaler


def compute_shapiro_wilk(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shapiro–Wilk por feature (global) e por espécie × feature."""
    global_rows: list[dict] = []
    by_sp_rows: list[dict] = []

    for col in features:
        x = df[col].dropna().astype(float)
        if len(x) >= 3:
            stat, p = stats.shapiro(x)
            global_rows.append(
                {
                    "feature": col,
                    "n": int(len(x)),
                    "W": round(float(stat), 4),
                    "p_value": float(p),
                    "normal_005": "sim" if p >= 0.05 else "não",
                }
            )

    for species, grp in df.groupby("species"):
        for col in features:
            x = grp[col].dropna().astype(float)
            if len(x) >= 3:
                stat, p = stats.shapiro(x)
                by_sp_rows.append(
                    {
                        "species": species,
                        "feature": col,
                        "n": int(len(x)),
                        "W": round(float(stat), 4),
                        "p_value": float(p),
                        "normal_005": "sim" if p >= 0.05 else "não",
                    }
                )

    global_df = pd.DataFrame(global_rows).sort_values("p_value")
    by_sp_df = pd.DataFrame(by_sp_rows).sort_values(["species", "p_value"])
    return global_df, by_sp_df


def plot_shapiro_wilk_significancia(shapiro_df: pd.DataFrame, fig_dir: Path) -> Path:
    """Barras −log10(p): verde se p≥0,05 (não rejeita normalidade), cinza se p<0,05."""
    plot_df = shapiro_df.copy()
    plot_df["neg_log_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-300))
    plot_df = plot_df.sort_values("neg_log_p", ascending=True)
    plot_df["cor"] = np.where(plot_df["p_value"] >= 0.05, "seagreen", "0.65")

    out = fig_dir / "shapiro_wilk_significancia.png"
    plt.figure(figsize=(10, max(5, len(plot_df) * 0.38)))
    plt.barh(plot_df["feature"], plot_df["neg_log_p"], color=plot_df["cor"], edgecolor="0.2")
    plt.axvline(-np.log10(0.05), color="crimson", ls="--", lw=1.2, label=r"$\alpha = 0{,}05$")
    plt.xlabel(r"$-\log_{10}$(p-value)")
    plt.ylabel("Feature")
    plt.title("Shapiro–Wilk — normalidade por feature (amostra global)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    *,
    n_std: float = 2.447,
    edgecolor: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "-",
) -> None:
    """Elipse de confiança ~95% (2 df, χ²) no plano PC1–PC2."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0:
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    angle = np.degrees(np.arctan2(cov[1, 1], cov[0, 1]))
    mean_x, mean_y = np.mean(x), np.mean(y)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    ell = Ellipse(
        (mean_x, mean_y),
        width=ell_radius_x * 2 * scale_x,
        height=ell_radius_y * 2 * scale_y,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(ell)


def plot_pca_especies_ellipse(
    df: pd.DataFrame, features: list[str], fig_dir: Path
) -> tuple[Path, np.ndarray]:
    """PCA 2D com elipses de confiança 95% por espécie (dados padronizados)."""
    pca, coords, _ = _fit_pca_full(df, features)
    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    ev = pca.explained_variance_ratio_

    out = fig_dir / "pca_especies_elipse.png"
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", n_colors=pca_df["species"].nunique())
    species_order = sorted(pca_df["species"].unique())

    for color, species in zip(palette, species_order):
        sub = pca_df[pca_df["species"] == species]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=55,
            alpha=0.8,
            color=color,
            label=species.replace("Cololobus ", "C. "),
            edgecolors="white",
            linewidths=0.4,
        )
        _confidence_ellipse(
            sub["PC1"].to_numpy(),
            sub["PC2"].to_numpy(),
            ax,
            edgecolor=color,
            linewidth=2.0,
        )

    all_x = pca_df["PC1"].to_numpy()
    all_y = pca_df["PC2"].to_numpy()
    _confidence_ellipse(all_x, all_y, ax, edgecolor="0.15", linewidth=1.5, linestyle="--")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"PCA 2D com elipses 95% — dados padronizados "
        f"(variância: {ev[0]:.1%}, {ev[1]:.1%})"
    )
    ax.legend(title="Espécie", loc="best", fontsize=8)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out, ev[:2]


def plot_pca_especies_orientador(
    df: pd.DataFrame, features: list[str], fig_dir: Path
) -> tuple[Path, np.ndarray]:
    """
    PCA estilo relatório palinológico: pontos por grão, barras de EP (SEM da espécie
    em PC1/PC2), elipses 95% e centróides destacados.
    """
    pca, coords, _ = _fit_pca_full(df, features)
    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    ev = pca.explained_variance_ratio_

    sem = pca_df.groupby("species")[["PC1", "PC2"]].sem()
    pca_df = pca_df.join(sem, on="species", rsuffix="_sem")

    out = fig_dir / "pca_especies_orientador.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    species_order = sorted(pca_df["species"].unique())
    palette = sns.color_palette("tab10", n_colors=len(species_order))

    for color, species in zip(palette, species_order):
        sub = pca_df[pca_df["species"] == species]
        label = species.replace("Cololobus ", "C. ")
        ax.errorbar(
            sub["PC1"],
            sub["PC2"],
            xerr=sub["PC1_sem"],
            yerr=sub["PC2_sem"],
            fmt="none",
            ecolor="0.25",
            elinewidth=0.9,
            capsize=2,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=48,
            alpha=0.85,
            color=color,
            label=label,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        _confidence_ellipse(
            sub["PC1"].to_numpy(),
            sub["PC2"].to_numpy(),
            ax,
            edgecolor=color,
            linewidth=2.2,
        )
        ax.scatter(
            sub["PC1"].mean(),
            sub["PC2"].mean(),
            s=140,
            marker="D",
            color=color,
            edgecolors="0.15",
            linewidths=0.8,
            zorder=4,
        )

    _confidence_ellipse(
        pca_df["PC1"].to_numpy(),
        pca_df["PC2"].to_numpy(),
        ax,
        edgecolor="0.1",
        linewidth=2.0,
        linestyle="--",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"PCA 2D — estilo centróide ± EP (SEM) por espécie "
        f"(variância: {ev[0]:.1%}, {ev[1]:.1%})"
    )
    ax.legend(title="Espécie", loc="best", fontsize=8)
    ax.axhline(0, color="0.88", lw=0.7)
    ax.axvline(0, color="0.88", lw=0.7)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out, ev[:2]


def plot_pca_especies(df: pd.DataFrame, features: list[str], fig_dir: Path) -> tuple[Path, np.ndarray]:
    pca, coords, _ = _fit_pca_full(df, features)
    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    ev = pca.explained_variance_ratio_
    out = fig_dir / "pca_especies.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="species", s=60, alpha=0.85)
    plt.title(f"PCA 2D (variância explicada: {ev[0]:.1%}, {ev[1]:.1%})")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out, ev[:2]


def plot_pca_scree(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    pca, _, _ = _fit_pca_full(df, features)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    n_show = np.searchsorted(cum, 0.80) + 1
    n_show = min(max(n_show, 2), len(ev))
    out = fig_dir / "pca_scree.png"
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(1, n_show + 1)
    ax1.bar(x, ev[:n_show] * 100, color="steelblue", alpha=0.8, label="Variância individual")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("% variância explicada")
    ax2 = ax1.twinx()
    ax2.plot(x, cum[:n_show] * 100, color="coral", marker="o", label="Variância acumulada")
    ax2.set_ylabel("% acumulada")
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle("Scree plot — PCA")
    fig.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_pca_loadings(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    pca, _, _ = _fit_pca_full(df, features)
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=["PC1", "PC2"],
        index=features,
    )
    out = fig_dir / "pca_loadings.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(features) * 0.3)))
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        sorted_ld = loadings[pc].sort_values()
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in sorted_ld.values]
        ax.barh(sorted_ld.index, sorted_ld.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Loadings — {pc}")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Contribuição das features aos componentes principais", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_heatmap_perfil_especies(df: pd.DataFrame, features: list[str], fig_dir: Path) -> Path:
    means = df.groupby("species")[features].mean()
    z = (means - means.mean()) / means.std()
    out = fig_dir / "heatmap_perfil_especies.png"
    plt.figure(figsize=(14, max(4, len(means) * 0.9)))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title("Perfil morfométrico por espécie (médias padronizadas)")
    plt.ylabel("Espécie")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_outliers_por_feature(outlier_counts: dict[str, int], fig_dir: Path) -> Path:
    tbl = pd.Series(outlier_counts).sort_values(ascending=True)
    out = fig_dir / "outliers_por_feature.png"
    plt.figure(figsize=(10, max(5, len(tbl) * 0.35)))
    sns.barplot(x=tbl.values, y=tbl.index, orient="h", color="indianred")
    plt.xlabel("Nº de outliers (IQR 1,5×)")
    plt.ylabel("Feature")
    plt.title("Outliers por feature")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def generate_all_eda_plots(
    df_raw: pd.DataFrame,
    df_imp: pd.DataFrame,
    features: list[str],
    kw: pd.DataFrame,
    corr: pd.DataFrame,
    outlier_counts: dict[str, int],
    fig_dir: Path,
    *,
    shapiro_global: pd.DataFrame | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Gera todas as figuras EDA e retorna paths + variância PCA (2 comp.)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plotters = [
        lambda: plot_missing_por_feature(df_raw, features, fig_dir),
        lambda: plot_missing_por_especie(df_raw, features, fig_dir),
        lambda: plot_correlacao_features(corr, fig_dir),
        lambda: plot_correlacao_clustermap(corr, fig_dir),
        lambda: plot_boxplots_todas_features(df_imp, features, fig_dir),
        lambda: plot_histogramas_todas_features(df_imp, features, fig_dir),
        lambda: plot_violinos_todas_features(df_imp, features, fig_dir),
        lambda: plot_ridgelines_features(df_imp, kw, features, fig_dir),
        lambda: plot_histogramas_especies(df_imp, features, fig_dir),
        lambda: plot_boxplots_especies(df_imp, features, fig_dir),
        lambda: plot_kruskal_wallis_significancia(kw, fig_dir),
        lambda: plot_scatter_eixos(df_imp, fig_dir),
        lambda: plot_scatter_morfologia(df_imp, fig_dir),
        lambda: plot_scatter_aberturas(df_raw, fig_dir),
        lambda: plot_pairplot_top_features(df_imp, kw, features, fig_dir),
        lambda: plot_outliers_por_feature(outlier_counts, fig_dir),
        lambda: plot_heatmap_perfil_especies(df_imp, features, fig_dir),
        lambda: plot_pca_scree(df_imp, features, fig_dir),
        lambda: plot_pca_loadings(df_imp, features, fig_dir),
    ]

    for fn in plotters:
        paths.append(fn())

    pca_path, ev = plot_pca_especies(df_imp, features, fig_dir)
    paths.append(pca_path)
    paths.append(plot_pca_especies_ellipse(df_imp, features, fig_dir)[0])
    paths.append(plot_pca_especies_orientador(df_imp, features, fig_dir)[0])

    if shapiro_global is not None and len(shapiro_global) > 0:
        paths.append(plot_shapiro_wilk_significancia(shapiro_global, fig_dir))

    return paths, ev
