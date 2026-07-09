"""Três níveis de pré-processamento para comparação de PCA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pollen_ml.config import FEATURE_COLUMNS
from pollen_ml.eda_plots import RANDOM_STATE, _confidence_ellipse, _fit_pca_full
from pollen_ml.preprocess import clean_dataframe, impute_by_view, mark_not_applicable

LOW_NA_THRESHOLD = 0.10
IQR_K = 1.5


@dataclass
class PipelineLevel:
    key: str
    label: str
    description: str
    df: pd.DataFrame
    features: list[str]
    n_samples: int


def _features_low_na(df: pd.DataFrame, threshold: float = LOW_NA_THRESHOLD) -> list[str]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    rates = df[cols].isna().mean()
    return [c for c in cols if rates[c] < threshold]


def _winsorize_iqr(df: pd.DataFrame, features: list[str], k: float = IQR_K) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        s = out[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out[col] = s.clip(q1 - k * iqr, q3 + k * iqr)
    return out


def prepare_level_minimo(df_raw: pd.DataFrame) -> PipelineLevel:
    """valid=1 + imputação (vista-aware + mediana) — pipeline atual."""
    work = clean_dataframe(df_raw)
    work = mark_not_applicable(work)
    features = [c for c in FEATURE_COLUMNS if c in work.columns]
    work = impute_by_view(work, features)
    imputer = SimpleImputer(strategy="median")
    work[features] = imputer.fit_transform(work[features])
    return PipelineLevel(
        key="minimo",
        label="Nível 1 — Mínimo (atual)",
        description="valid=1; imputação mediana (view-aware); z-score no PCA",
        df=work,
        features=features,
        n_samples=len(work),
    )


def prepare_level_melhorado(df_raw: pd.DataFrame) -> PipelineLevel:
    """valid=1; só features com <10% NA; winsorização IQR; sem imputar aberturas."""
    work = clean_dataframe(df_raw)
    features = _features_low_na(work)
    sub = work.dropna(subset=features).copy()
    sub = _winsorize_iqr(sub, features)
    return PipelineLevel(
        key="melhorado",
        label="Nível 2 — Melhorado",
        description=(
            f"valid=1; {len(features)} features com <{LOW_NA_THRESHOLD:.0%} NA; "
            "complete-case (sem imputar aberturas); winsorização IQR 1,5×"
        ),
        df=sub,
        features=features,
        n_samples=len(sub),
    )


def prepare_level_exploratorio(df_raw: pd.DataFrame, kw: pd.DataFrame) -> PipelineLevel:
    """Nível 2 + apenas features significativas no Kruskal–Wallis (p < 0,05)."""
    base = prepare_level_melhorado(df_raw)
    sig = set(kw.loc[kw["p_value"] < 0.05, "feature"])
    features = [f for f in base.features if f in sig]
    sub = base.df.dropna(subset=features).copy()
    sub = _winsorize_iqr(sub, features)
    return PipelineLevel(
        key="exploratorio",
        label="Nível 3 — Exploratório",
        description=(
            f"nível 2 + {len(features)} features com Kruskal–Wallis p<0,05 "
            f"(sem largura_espinho e aberturas com muito NA)"
        ),
        df=sub,
        features=features,
        n_samples=len(sub),
    )


def draw_pca_ellipses_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    features: list[str],
    *,
    title: str,
    show_legend: bool = False,
) -> np.ndarray:
    """Desenha PCA 2D com elipses 95% no eixo fornecido."""
    pca, coords, _ = _fit_pca_full(df, features)
    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    ev = pca.explained_variance_ratio_

    species_order = sorted(pca_df["species"].unique())
    palette = sns.color_palette("tab10", n_colors=len(species_order))

    for color, species in zip(palette, species_order):
        sub = pca_df[pca_df["species"] == species]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=40,
            alpha=0.75,
            color=color,
            label=species.replace("Cololobus ", "C. "),
            edgecolors="white",
            linewidths=0.3,
        )
        _confidence_ellipse(
            sub["PC1"].to_numpy(),
            sub["PC2"].to_numpy(),
            ax,
            edgecolor=color,
            linewidth=1.8,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\n(PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}; n={len(df)})", fontsize=10)
    ax.axhline(0, color="0.9", lw=0.6)
    ax.axvline(0, color="0.9", lw=0.6)
    if show_legend:
        ax.legend(title="Espécie", fontsize=7, loc="best")
    return ev[:2]


def generate_pca_pipeline_comparison(
    df_raw: pd.DataFrame,
    kw: pd.DataFrame,
    fig_dir: Path,
) -> dict:
    """Gera figuras individuais e painel 1×3 dos três níveis de pré-processamento."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    levels = [
        prepare_level_minimo(df_raw),
        prepare_level_melhorado(df_raw),
        prepare_level_exploratorio(df_raw, kw),
    ]

    paths: list[Path] = []
    ev_rows: list[dict] = []

    for lvl in levels:
        out = fig_dir / f"pca_pipeline_{lvl.key}.png"
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ev = draw_pca_ellipses_on_ax(
            ax,
            lvl.df,
            lvl.features,
            title=lvl.label,
            show_legend=True,
        )
        ax.text(
            0.02,
            0.02,
            f"{len(lvl.features)} features",
            transform=ax.transAxes,
            fontsize=8,
            color="0.4",
            va="bottom",
        )
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        paths.append(out)
        ev_rows.append(
            {
                "nivel": lvl.label,
                "n": lvl.n_samples,
                "n_features": len(lvl.features),
                "pc1": ev[0],
                "pc2": ev[1],
                "features": ", ".join(lvl.features),
            }
        )

    panel_out = fig_dir / "pca_pipeline_comparacao.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, lvl in zip(axes, levels):
        draw_pca_ellipses_on_ax(
            ax,
            lvl.df,
            lvl.features,
            title=lvl.key.replace("_", " ").title(),
            show_legend=(lvl.key == "exploratorio"),
        )
    fig.suptitle(
        "Comparação de pré-processamento — PCA 2D com elipses 95% (z-score)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(panel_out, bbox_inches="tight")
    plt.close()
    paths.append(panel_out)

    summary_df = pd.DataFrame(ev_rows)
    return {
        "levels": levels,
        "paths": paths,
        "summary": summary_df,
        "panel_path": panel_out,
    }


def write_pca_pipeline_report(result: dict, out: Path) -> Path:
    """Gera resultados_pca_pipelines.md."""
    summary = result["summary"].copy()
    summary_show = summary.copy()
    summary_show["pc1"] = summary_show["pc1"].map(lambda x: f"{x:.1%}")
    summary_show["pc2"] = summary_show["pc2"].map(lambda x: f"{x:.1%}")

    def _md_table(df: pd.DataFrame) -> str:
        headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
        return "\n".join([headers, sep, *rows])

    lines = [
        "# Comparação de pré-processamento — PCA (3 níveis)",
        "",
        "## Níveis",
        "",
    ]
    for lvl in result["levels"]:
        lines += [
            f"### {lvl.label}",
            "",
            f"{lvl.description} — **n = {lvl.n_samples}** grãos.",
            "",
            f"Features (`{len(lvl.features)}`): `{', '.join(lvl.features)}`",
            "",
            f"![{lvl.key}](../outputs/figures/pca_pipeline_{lvl.key}.png)",
            "",
        ]

    lines += [
        "## Variância explicada",
        "",
        _md_table(summary_show[["nivel", "n", "n_features", "pc1", "pc2"]]),
        "",
        "## Painel comparativo",
        "",
        "![Comparação](../outputs/figures/pca_pipeline_comparacao.png)",
        "",
        "## Interpretação",
        "",
        "- **Nível 1:** imputação em traits com muito NA pode artificialmente aproximar espécies.",
        "- **Nível 2:** complete-case + winsorização reduz ruído de outliers e aberturas imputadas.",
        "- **Nível 3:** menos features, porém mais discriminantes (Kruskal–Wallis); variância em 2D pode subir ou descer.",
        "- Overlap parcial entre *longiangustatus*, *rupestris* e *ruschianur* tende a persistir (congêneres).",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
