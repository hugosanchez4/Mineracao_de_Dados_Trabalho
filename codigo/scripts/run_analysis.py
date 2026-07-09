#!/usr/bin/env python3
"""Executa EDA e classificação ML sobre polen_features_analyze.csv e gera relatórios .md."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pollen_ml.config import FEATURE_COLUMNS, FEATURES_CSV, load_features
from pollen_ml.eda_plots import compute_shapiro_wilk, generate_all_eda_plots
from pollen_ml.pca_pipelines import generate_pca_pipeline_comparison, write_pca_pipeline_report
from pollen_ml.ml_plots import generate_ml_plots
from pollen_ml.ml_interpretation import generate_interpretation_plots
from pollen_ml.preprocess import build_features, clean_dataframe, make_preprocessor

FIG_DIR = ROOT / "outputs" / "figures"
DOCS_DIR = ROOT / "docs"
RANDOM_STATE = 42
CV_FOLDS = 5


def _md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(format(val, float_fmt))
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([headers, sep, *rows])


def run_eda(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    df = clean_dataframe(df_raw)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]

    missing_pct = (df[features].isna().mean() * 100).sort_values(ascending=False)
    missing_by_species = (
        df.groupby("species")[features]
        .apply(lambda g: g.isna().mean().mean() * 100)
        .sort_values(ascending=False)
    )

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df[features])
    df_imp = df.copy()
    df_imp[features] = X_imp

    outlier_counts: dict[str, int] = {}
    outlier_flags = pd.DataFrame(index=df_imp.index)
    for col in features:
        q1, q3 = df_imp[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_flags[col] = (df_imp[col] < low) | (df_imp[col] > high)
        outlier_counts[col] = int(outlier_flags[col].sum())
    df_imp["is_outlier_any"] = outlier_flags.any(axis=1)

    desc_mean = df_imp.groupby("species")[features].mean().round(3)
    desc_std = df_imp.groupby("species")[features].std().round(3)
    desc_median = df_imp.groupby("species")[features].median().round(3)

    kw_rows = []
    for col in features:
        groups = [g[col].dropna().values for _, g in df_imp.groupby("species") if len(g) > 0]
        if len(groups) < 2:
            continue
        stat, p = stats.kruskal(*groups)
        kw_rows.append(
            {
                "feature": col,
                "H": round(stat, 3),
                "p_value": p,
                "significativo_005": "sim" if p < 0.05 else "não",
            }
        )
    kw = pd.DataFrame(kw_rows).sort_values("p_value")

    shapiro_global, shapiro_by_species = compute_shapiro_wilk(df_imp, features)

    corr = df_imp[features].corr().round(3)

    figures_generated, ev = generate_all_eda_plots(
        df_raw=df,
        df_imp=df_imp,
        features=features,
        kw=kw,
        corr=corr,
        outlier_counts=outlier_counts,
        fig_dir=FIG_DIR,
        shapiro_global=shapiro_global,
    )

    pca_pipeline = generate_pca_pipeline_comparison(df_raw, kw, FIG_DIR)
    write_pca_pipeline_report(pca_pipeline, DOCS_DIR / "resultados_pca_pipelines.md")

    clean_path = ROOT / "data" / "tables" / "pollen_features_clean.csv"
    df_imp.to_csv(clean_path, index=False)

    species_counts = df["species"].value_counts().sort_index()
    view_counts = df["view"].value_counts() if "view" in df.columns else pd.Series(dtype=int)

    artifacts = {
        "n_raw": len(df_raw),
        "n_clean": len(df),
        "n_removed_valid0": len(df_raw) - len(df),
        "species_counts": species_counts,
        "view_counts": view_counts,
        "missing_pct": missing_pct,
        "missing_by_species": missing_by_species,
        "outlier_counts": outlier_counts,
        "n_outlier_any": int(df_imp["is_outlier_any"].sum()),
        "desc_mean": desc_mean,
        "desc_std": desc_std,
        "desc_median": desc_median,
        "kw": kw,
        "shapiro_global": shapiro_global,
        "shapiro_by_species": shapiro_by_species,
        "corr": corr,
        "pca_variance": ev,
        "figures_generated": figures_generated,
        "pca_pipeline": pca_pipeline,
        "clean_path": clean_path,
        "df_imp": df_imp,
        "features": features,
    }
    return df_imp, artifacts


def write_eda_report(artifacts: dict, source: Path) -> Path:
    out = DOCS_DIR / "resultados_eda.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    species_tbl = pd.DataFrame(
        {"espécie": artifacts["species_counts"].index, "n": artifacts["species_counts"].values}
    )

    missing_tbl = artifacts["missing_pct"].reset_index()
    missing_tbl.columns = ["feature", "%_ausente"]
    missing_tbl["%_ausente"] = missing_tbl["%_ausente"].round(1)

    outlier_tbl = pd.DataFrame(
        [{"feature": k, "n_outliers_iqr": v} for k, v in artifacts["outlier_counts"].items()]
    ).sort_values("n_outliers_iqr", ascending=False)

    kw_show = artifacts["kw"].copy()
    kw_show["p_value"] = kw_show["p_value"].map(lambda x: f"{x:.2e}")

    mean_show = artifacts["desc_mean"].reset_index()

    high_corr = []
    corr = artifacts["corr"]
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            v = corr.loc[c1, c2]
            if abs(v) >= 0.7:
                high_corr.append({"par": f"{c1} × {c2}", "r": round(v, 3)})
    high_corr_df = pd.DataFrame(high_corr).sort_values("r", key=abs, ascending=False) if high_corr else pd.DataFrame()

    lines = [
        "# Resultados da análise exploratória (EDA)",
        "",
        f"**Gerado em:** {now}  ",
        f"**Fonte:** `{source.relative_to(ROOT)}`",
        "",
        "## 1. Visão geral do dataset",
        "",
        f"- Registros no arquivo: **{artifacts['n_raw']}**",
        f"- Após excluir `valid=0`: **{artifacts['n_clean']}** ({artifacts['n_removed_valid0']} removidos)",
        f"- Features morfométricas: **{len(artifacts['features'])}**",
        "",
        "### Amostras por espécie",
        "",
        _md_table(species_tbl),
        "",
    ]

    if len(artifacts["view_counts"]) > 0:
        view_tbl = pd.DataFrame(
            {"vista": artifacts["view_counts"].index, "n": artifacts["view_counts"].values}
        )
        lines += ["### Amostras por vista", "", _md_table(view_tbl), ""]

    lines += [
        "## 2. Valores ausentes",
        "",
        "Células vazias ou `NI` no CSV foram tratadas como NA antes da imputação (mediana global na EDA).",
        "",
        _md_table(missing_tbl),
        "",
        "## 3. Outliers (regra IQR 1,5×)",
        "",
        f"Registros com pelo menos um outlier em alguma feature: **{artifacts['n_outlier_any']}**.",
        "",
        _md_table(outlier_tbl),
        "",
        "## 4. Estatística descritiva (média por espécie, µm ou adimensional)",
        "",
        _md_table(mean_show, float_fmt=".2f"),
        "",
        "## 5. Kruskal-Wallis (diferença entre espécies)",
        "",
        "H₀: distribuições iguais entre espécies. α = 0,05.",
        "",
        _md_table(kw_show),
        "",
        f"Features com p < 0,05: **{(artifacts['kw']['p_value'] < 0.05).sum()}** de {len(artifacts['kw'])}.",
        "",
    ]

    ev = artifacts["pca_variance"]
    fig_paths = artifacts.get("figures_generated", [])
    rel_figs = [p.relative_to(ROOT) for p in fig_paths]

    def _fig_line(name: str, rel: str) -> str:
        return f"| {name} | `{rel}` |"

    fig_table_rows = [
        _fig_line("Ausentes por feature", "outputs/figures/missing_por_feature.png"),
        _fig_line("Ausentes por espécie", "outputs/figures/missing_por_especie.png"),
        _fig_line("Correlação", "outputs/figures/correlacao_features.png"),
        _fig_line("Clustermap correlação", "outputs/figures/correlacao_clustermap.png"),
        _fig_line("Box plots (14 features)", "outputs/figures/boxplots_todas_features.png"),
        _fig_line("Histogramas (14 features)", "outputs/figures/histogramas_todas_features.png"),
        _fig_line("Violin plots (14 features)", "outputs/figures/violinos_todas_features.png"),
        _fig_line("Ridgelines KDE", "outputs/figures/ridgelines_features.png"),
        _fig_line("Histogramas principais", "outputs/figures/histogramas_especies.png"),
        _fig_line("Box plots principais", "outputs/figures/boxplots_especies.png"),
        _fig_line("Kruskal-Wallis", "outputs/figures/kruskal_wallis_significancia.png"),
        _fig_line("Shapiro–Wilk", "outputs/figures/shapiro_wilk_significancia.png"),
        _fig_line("Scatter eixos", "outputs/figures/scatter_eixos.png"),
        _fig_line("Scatter morfologia", "outputs/figures/scatter_morfologia.png"),
        _fig_line("Scatter aberturas", "outputs/figures/scatter_aberturas.png"),
        _fig_line("Pairplot top features", "outputs/figures/pairplot_top_features.png"),
        _fig_line("Outliers IQR", "outputs/figures/outliers_por_feature.png"),
        _fig_line("Perfil por espécie", "outputs/figures/heatmap_perfil_especies.png"),
        _fig_line("PCA scree", "outputs/figures/pca_scree.png"),
        _fig_line("PCA loadings", "outputs/figures/pca_loadings.png"),
        _fig_line("PCA 2D", "outputs/figures/pca_especies.png"),
        _fig_line("PCA 2D elipses", "outputs/figures/pca_especies_elipse.png"),
        _fig_line("PCA estilo orientador", "outputs/figures/pca_especies_orientador.png"),
    ]

    lines += [
        "## 5.1 Normalidade (Shapiro–Wilk)",
        "",
        "H₀: os dados seguem distribuição normal. α = 0,05 — se p ≥ 0,05, **não** rejeita-se H₀.",
        "",
        "Teste aplicado por feature na amostra global (dados imputados, *n* = 89). Com *n* pequeno e outliers, o teste tem baixo poder; use junto com os violin plots.",
        "",
    ]

    shapiro = artifacts.get("shapiro_global")
    if shapiro is not None and len(shapiro) > 0:
        sh_show = shapiro.copy()
        sh_show["p_value"] = sh_show["p_value"].map(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
        n_normal = int((artifacts["shapiro_global"]["p_value"] >= 0.05).sum())
        lines += [
            f"Features com p ≥ 0,05 (compatíveis com normalidade): **{n_normal}** de {len(shapiro)}.",
            "",
            "![Shapiro–Wilk](../outputs/figures/shapiro_wilk_significancia.png)",
            "",
            _md_table(sh_show[["feature", "n", "W", "p_value", "normal_005"]]),
            "",
        ]

    lines += [
        "## 6. Correlação entre features",
        "",
        "![Matriz de correlação](../outputs/figures/correlacao_features.png)",
        "",
        "![Clustermap](../outputs/figures/correlacao_clustermap.png)",
        "",
    ]

    if len(high_corr_df) > 0:
        lines += ["### Pares com |r| ≥ 0,70", "", _md_table(high_corr_df), ""]

    lines += [
        "## 7. PCA",
        "",
        f"- PC1: **{ev[0]:.1%}** da variância",
        f"- PC2: **{ev[1]:.1%}** da variância",
        f"- Total (2 comp.): **{ev.sum():.1%}**",
        "",
        "![PCA por espécie](../outputs/figures/pca_especies.png)",
        "",
        "![PCA com elipses 95%](../outputs/figures/pca_especies_elipse.png)",
        "",
        "Elipses coloridas: intervalo de confiança ~95% por espécie no plano PC1–PC2 (dados padronizados com z-score). Elipse tracejada: conjunto total.",
        "",
        "![PCA estilo orientador (EP + elipses)](../outputs/figures/pca_especies_orientador.png)",
        "",
        "Pontos = grãos; barras de erro = EP (SEM) da espécie em PC1 e PC2; losangos = centróide; elipses = 95% por táxon.",
        "",
        "![Scree plot](../outputs/figures/pca_scree.png)",
        "",
        "### Comparação de pré-processamento (3 níveis)",
        "",
        "![PCA pipelines](../outputs/figures/pca_pipeline_comparacao.png)",
        "",
        "| Nível | n | features | Descrição |",
        "| --- | --- | --- | --- |",
    ]

    pca_pipe = artifacts.get("pca_pipeline")
    if pca_pipe:
        for lvl in pca_pipe["levels"]:
            lines.append(
                f"| {lvl.label} | {lvl.n_samples} | {len(lvl.features)} | {lvl.description} |"
            )
        lines += [
            "",
            "Detalhes em [`resultados_pca_pipelines.md`](resultados_pca_pipelines.md).",
            "",
        ]

    lines += [
        "![Loadings](../outputs/figures/pca_loadings.png)",
        "",
        "## 8. Visualizações expandidas",
        "",
        "### Qualidade dos dados",
        "",
        "Medidas de abertura concentram os maiores % de ausentes; o heatmap por espécie mostra se o padrão é uniforme ou específico de cada táxon.",
        "",
        "![Ausentes por feature](../outputs/figures/missing_por_feature.png)",
        "",
        "![Ausentes por espécie](../outputs/figures/missing_por_especie.png)",
        "",
        "![Outliers](../outputs/figures/outliers_por_feature.png)",
        "",
        "### Distribuições por espécie (14 features)",
        "",
        "![Box plots todas](../outputs/figures/boxplots_todas_features.png)",
        "",
        "![Histogramas todas](../outputs/figures/histogramas_todas_features.png)",
        "",
        "![Violinos](../outputs/figures/violinos_todas_features.png)",
        "",
        "![Ridgelines](../outputs/figures/ridgelines_features.png)",
        "",
        "### Significância estatística",
        "",
        f"Features com p < 0,05 no Kruskal-Wallis: **{(artifacts['kw']['p_value'] < 0.05).sum()}** de {len(artifacts['kw'])}.",
        "",
        "![Kruskal-Wallis](../outputs/figures/kruskal_wallis_significancia.png)",
        "",
        "### Relações bivariadas",
        "",
        "![Scatter morfologia](../outputs/figures/scatter_morfologia.png)",
        "",
        "![Scatter aberturas](../outputs/figures/scatter_aberturas.png)",
        "",
        "![Scatter eixos](../outputs/figures/scatter_eixos.png)",
        "",
        "![Pairplot](../outputs/figures/pairplot_top_features.png)",
        "",
        "### Perfil morfométrico",
        "",
        "Heatmap com médias padronizadas (z-score) por espécie — compara traits relativos entre táxons.",
        "",
        "![Perfil espécies](../outputs/figures/heatmap_perfil_especies.png)",
        "",
        "## 9. Índice de figuras",
        "",
        "| Figura | Arquivo |",
        "| --- | --- |",
        *fig_table_rows,
        "",
        f"Total de figuras geradas: **{len(rel_figs)}**",
        "",
        f"Dataset imputado exportado em: `{artifacts['clean_path'].relative_to(ROOT)}`",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ml(df_raw: pd.DataFrame) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    include_view = "view" in df_raw.columns and df_raw["view"].nunique() > 1
    # Dados brutos (sem imputar/escalar): a padronização é feita DENTRO da CV,
    # via Pipeline, para não vazar informação do fold de teste.
    X_in, y, numeric_cols, feature_names, include_view = build_features(
        df_raw, include_view=include_view, view_aware=True
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def make_pipe(estimator) -> Pipeline:
        """Encapsula imputação + StandardScaler + estimador num só Pipeline."""
        return Pipeline(
            [
                ("pre", make_preprocessor(numeric_cols, include_view=include_view)),
                ("clf", estimator),
            ]
        )

    def _strip(params: dict) -> dict:
        return {k.replace("clf__", ""): v for k, v in params.items()}

    dummy = make_pipe(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))
    dummy_scores = cross_val_score(dummy, X_in, y_enc, cv=cv, scoring="f1_macro")
    dummy_acc_scores = cross_val_score(dummy, X_in, y_enc, cv=cv, scoring="accuracy")

    rf_pipe = make_pipe(RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
    rf_param = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10, 15],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }
    rf_grid = GridSearchCV(
        rf_pipe, rf_param, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True
    )
    rf_grid.fit(X_in, y_enc)

    svm_pipe = make_pipe(SVC(kernel="rbf", random_state=RANDOM_STATE, class_weight="balanced"))
    svm_param = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", "auto", 0.01, 0.1]}
    svm_grid = GridSearchCV(
        svm_pipe, svm_param, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True
    )
    svm_grid.fit(X_in, y_enc)

    rf_best_params = _strip(rf_grid.best_params_)
    svm_best_params = _strip(svm_grid.best_params_)

    rf_f1_scores = cross_val_score(rf_grid.best_estimator_, X_in, y_enc, cv=cv, scoring="f1_macro")
    rf_acc_scores = cross_val_score(rf_grid.best_estimator_, X_in, y_enc, cv=cv, scoring="accuracy")
    svm_f1_scores = cross_val_score(svm_grid.best_estimator_, X_in, y_enc, cv=cv, scoring="f1_macro")
    svm_acc_scores = cross_val_score(svm_grid.best_estimator_, X_in, y_enc, cv=cv, scoring="accuracy")

    compare_models = pd.DataFrame(
        {
            "modelo": ["Dummy (baseline)", "Random Forest", "SVM RBF"],
            "f1_macro_mean": [
                dummy_scores.mean(),
                rf_f1_scores.mean(),
                svm_f1_scores.mean(),
            ],
            "f1_macro_std": [
                dummy_scores.std(),
                rf_f1_scores.std(),
                svm_f1_scores.std(),
            ],
            "accuracy_mean": [
                dummy_acc_scores.mean(),
                rf_acc_scores.mean(),
                svm_acc_scores.mean(),
            ],
            "accuracy_std": [
                dummy_acc_scores.std(),
                rf_acc_scores.std(),
                svm_acc_scores.std(),
            ],
        }
    )
    compare_models.attrs["n_classes"] = n_classes

    model_specs = [
        ("Dummy (baseline)", dummy),
        ("Random Forest", rf_grid.best_estimator_),
        ("SVM RBF", svm_grid.best_estimator_),
    ]
    fold_all_rows: list[dict] = []
    X_arr = X_in.to_numpy() if hasattr(X_in, "to_numpy") else X_in
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_enc), start=1):
        for model_name, estimator in model_specs:
            estimator.fit(X_in.iloc[train_idx], y_enc[train_idx])
            pred = estimator.predict(X_in.iloc[test_idx])
            fold_all_rows.append(
                {
                    "fold": fold_idx,
                    "modelo": model_name,
                    "f1_macro": f1_score(y_enc[test_idx], pred, average="macro"),
                    "accuracy": float((pred == y_enc[test_idx]).mean()),
                }
            )
    fold_all_df = pd.DataFrame(fold_all_rows)

    ml_plot_paths = generate_ml_plots(compare_models, fold_all_df, FIG_DIR)

    if rf_grid.best_score_ >= svm_grid.best_score_:
        final_pipe = rf_grid.best_estimator_
        model_name = "Random Forest"
    else:
        final_pipe = svm_grid.best_estimator_
        model_name = "SVM RBF"

    y_pred = cross_val_predict(final_pipe, X_in, y_enc, cv=cv)
    report = classification_report(y_enc, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_enc, y_pred)

    fold_f1 = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_enc), start=1):
        final_pipe.fit(X_in.iloc[train_idx], y_enc[train_idx])
        pred = final_pipe.predict(X_in.iloc[test_idx])
        fold_f1.append(
            {
                "fold": fold_idx,
                "f1_macro": f1_score(y_enc[test_idx], pred, average="macro"),
                "n_teste": len(test_idx),
            }
        )
    fold_df = pd.DataFrame(fold_f1)

    if model_name == "Random Forest":
        importances = final_pipe.named_steps["clf"].feature_importances_
        imp_df = (
            pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=imp_df, y="feature", x="importance", orient="h")
        plt.title("Importância de variáveis — Random Forest")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "rf_feature_importance.png", bbox_inches="tight")
        plt.close()
    else:
        imp_df = pd.DataFrame()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusão (CV {CV_FOLDS}-Fold) — {model_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Para interpretação (permutation importance, SHAP, PCA) e clustering usa-se
    # a matriz transformada; aqui o scaler pode ser ajustado no conjunto completo,
    # pois descreve o modelo final treinado com todos os dados (não é métrica de CV).
    preprocessor_full = make_preprocessor(numeric_cols, include_view=include_view)
    X = preprocessor_full.fit_transform(X_in)
    final_estimator = final_pipe.named_steps["clf"]

    interpret = generate_interpretation_plots(
        df_raw,
        X,
        y_enc,
        y_pred,
        final_estimator,
        feature_names,
        class_names,
        cm,
        FIG_DIR,
        model_name=model_name,
    )

    kmeans = KMeans(n_clusters=n_classes, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X)
    ari = adjusted_rand_score(y_enc, clusters)

    summary_path = ROOT / "outputs" / "ml_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_amostras": len(y),
        "n_classes": n_classes,
        "cv_folds": CV_FOLDS,
        "modelo_final": model_name,
        "include_view": include_view,
        "rf_best_params": rf_best_params,
        "svm_best_params": svm_best_params,
        "f1_macro_dummy": float(dummy_scores.mean()),
        "f1_macro_rf": float(rf_f1_scores.mean()),
        "f1_macro_svm": float(svm_f1_scores.mean()),
        "accuracy_dummy": float(dummy_acc_scores.mean()),
        "accuracy_rf": float(rf_acc_scores.mean()),
        "accuracy_svm": float(svm_acc_scores.mean()),
        "ari_kmeans": float(ari),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return {
        "n_samples": len(y),
        "n_classes": n_classes,
        "class_names": class_names,
        "feature_names": feature_names,
        "include_view": include_view,
        "cv_folds": CV_FOLDS,
        "dummy_scores": dummy_scores,
        "dummy_acc_scores": dummy_acc_scores,
        "rf_f1_scores": rf_f1_scores,
        "rf_acc_scores": rf_acc_scores,
        "svm_f1_scores": svm_f1_scores,
        "svm_acc_scores": svm_acc_scores,
        "compare_models": compare_models,
        "fold_all_df": fold_all_df,
        "ml_plot_paths": ml_plot_paths,
        "rf_grid": rf_grid,
        "svm_grid": svm_grid,
        "rf_best_params": rf_best_params,
        "svm_best_params": svm_best_params,
        "model_name": model_name,
        "final_model": final_estimator,
        "y_enc": y_enc,
        "y_pred": y_pred,
        "report": report,
        "cm": cm,
        "fold_df": fold_df,
        "imp_df": imp_df,
        "interpret": interpret,
        "ari": ari,
        "summary": summary,
        "summary_path": summary_path,
    }


def write_ml_report(artifacts: dict, source: Path) -> Path:
    out = DOCS_DIR / "resultados_ml.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    compare = artifacts["compare_models"].copy()
    compare_show = compare.rename(
        columns={
            "f1_macro_mean": "f1_macro",
            "f1_macro_std": "f1_std",
            "accuracy_mean": "acuracia",
            "accuracy_std": "acur_std",
        }
    )
    for col in ["f1_macro", "f1_std", "acuracia", "acur_std"]:
        compare_show[col] = compare_show[col].round(3)

    fold_show = artifacts["fold_df"].copy()
    fold_show["fold"] = fold_show["fold"].astype(int)
    fold_show["f1_macro"] = fold_show["f1_macro"].round(3)
    fold_show["n_teste"] = fold_show["n_teste"].astype(int)

    report = artifacts["report"]
    per_class_rows = []
    for name in artifacts["class_names"]:
        if name in report:
            per_class_rows.append(
                {
                    "espécie": name,
                    "precision": round(report[name]["precision"], 3),
                    "recall": round(report[name]["recall"], 3),
                    "f1": round(report[name]["f1-score"], 3),
                    "suporte": int(report[name]["support"]),
                }
            )
    per_class_df = pd.DataFrame(per_class_rows)

    rf_params = ", ".join(f"`{k}={v}`" for k, v in artifacts["rf_best_params"].items())
    svm_params = ", ".join(f"`{k}={v}`" for k, v in artifacts["svm_best_params"].items())

    lines = [
        "# Resultados do modelo de classificação (ML)",
        "",
        f"**Gerado em:** {now}  ",
        f"**Fonte:** `{source.relative_to(ROOT)}`",
        "",
        "## 1. Problema e configuração",
        "",
        "- **Tarefa:** classificação supervisionada multiclasse (`species`)",
        f"- **Amostras (valid=1):** {artifacts['n_samples']}",
        f"- **Classes:** {artifacts['n_classes']} — {', '.join(artifacts['class_names'])}",
        f"- **Features numéricas:** {len([f for f in artifacts['feature_names'] if not f.startswith('view_')])}",
        f"- **Vista como feature:** {'sim' if artifacts['include_view'] else 'não (apenas uma vista no dataset)'}",
        "",
        "## 2. Pré-processamento",
        "",
        "1. Exclusão de registros com `valid=0`.",
        "2. Células vazias / `NI` tratadas como NA.",
        "3. Imputação da **mediana** e **StandardScaler** encapsulados num "
        "`Pipeline` do scikit-learn, **ajustado dentro de cada fold** da validação "
        "cruzada (evita vazamento de dados: o scaler/imputer não usa o fold de teste).",
        "",
        f"Features usadas: `{', '.join(artifacts['feature_names'])}`",
        "",
        "## 3. Validação cruzada (Stratified K-Fold)",
        "",
        f"- **K = {artifacts['cv_folds']}** (estratificado por classe)",
        "- **Métrica principal:** F1-macro",
        "- Matriz de confusão e relatório por classe: predições **out-of-fold** (`cross_val_predict`)",
        "",
        "### F1-macro por fold (modelo final)",
        "",
        _md_table(fold_show),
        "",
        "## 4. Modelos avaliados",
        "",
        "### 4.1 Baseline — DummyClassifier",
        "",
        "Estratégia `stratified`: predição aleatória respeitando proporção das classes.",
        "",
        f"- F1-macro: **{artifacts['dummy_scores'].mean():.3f}** ± {artifacts['dummy_scores'].std():.3f}",
        f"- Acurácia: **{artifacts['dummy_acc_scores'].mean():.3f}** ± {artifacts['dummy_acc_scores'].std():.3f}",
        "",
        "### 4.2 Random Forest — GridSearchCV",
        "",
        "Grade testada:",
        "- `n_estimators`: 100, 200, 300",
        "- `max_depth`: None, 5, 10, 15",
        "- `min_samples_leaf`: 1, 2, 4",
        "- `max_features`: sqrt, log2",
        "",
        f"**Melhores parâmetros:** {rf_params}",
        f"**F1-macro (CV):** {artifacts['rf_f1_scores'].mean():.3f} ± {artifacts['rf_f1_scores'].std():.3f}",
        f"**Acurácia (CV):** {artifacts['rf_acc_scores'].mean():.3f} ± {artifacts['rf_acc_scores'].std():.3f}",
        "",
        "### 4.3 SVM (kernel RBF) — GridSearchCV",
        "",
        "Grade testada:",
        "- `C`: 0.1, 1, 10, 100",
        "- `gamma`: scale, auto, 0.01, 0.1",
        "",
        f"**Melhores parâmetros:** {svm_params}",
        f"**F1-macro (CV):** {artifacts['svm_f1_scores'].mean():.3f} ± {artifacts['svm_f1_scores'].std():.3f}",
        f"**Acurácia (CV):** {artifacts['svm_acc_scores'].mean():.3f} ± {artifacts['svm_acc_scores'].std():.3f}",
        "",
        "## 5. Comparação entre modelos",
        "",
        _md_table(compare_show),
        "",
        "### Gráficos de comparação",
        "",
        "![Comparação F1 e acurácia](../outputs/figures/ml_comparacao_metricas.png)",
        "",
        "![Comparação acurácia](../outputs/figures/ml_comparacao_acuracia.png)",
        "",
        "![Métricas por fold](../outputs/figures/ml_metricas_por_fold.png)",
        "",
        f"**Modelo final escolhido:** {artifacts['model_name']} (maior F1-macro na validação cruzada).",
        "",
        "## 6. Desempenho do modelo final",
        "",
        f"- **F1-macro (macro avg):** {report['macro avg']['f1-score']:.3f}",
        f"- **Acurácia:** {report['accuracy']:.3f}",
        "",
        "### Métricas por espécie",
        "",
        _md_table(per_class_df),
        "",
        "### Matriz de confusão",
        "",
        "![Matriz de confusão](../outputs/figures/confusion_matrix.png)",
        "",
        "![Matriz normalizada](../outputs/figures/ml_confusion_matrix_norm.png)",
        "",
        "![Pares confundidos](../outputs/figures/ml_confusion_pares.png)",
        "",
        "![PCA acertos vs erros](../outputs/figures/ml_pca_acertos_erros.png)",
        "",
    ]

    interpret = artifacts.get("interpret", {})
    if interpret:
        pairs_show = interpret.get("pairs_df")
        if pairs_show is not None and len(pairs_show) > 0:
            lines += [
                "### Pares mais confundidos",
                "",
                _md_table(pairs_show[["par", "n"]]),
                "",
            ]
        perm_df = interpret.get("perm_df")
        if perm_df is not None and len(perm_df) > 0:
            perm_show = perm_df.head(10).copy()
            perm_show["importance_mean"] = perm_show["importance_mean"].round(4)
            perm_show["importance_std"] = perm_show["importance_std"].round(4)
            lines += [
                "### Importância por permutação (top 10)",
                "",
                "![Permutation importance](../outputs/figures/ml_permutation_importance.png)",
                "",
                _md_table(perm_show),
                "",
            ]
        shap_df = interpret.get("shap_df")
        if shap_df is not None and len(shap_df) > 0:
            shap_show = shap_df.head(10).copy()
            shap_show["mean_abs_shap"] = shap_show["mean_abs_shap"].round(4)
            lines += [
                "### SHAP (contribuição média absoluta)",
                "",
                "![SHAP importance](../outputs/figures/ml_shap_importance.png)",
                "",
                "![SHAP summary](../outputs/figures/ml_shap_summary.png)",
                "",
                _md_table(shap_show),
                "",
            ]
        lines += [
            "![Erros vs acertos por par](../outputs/figures/ml_erros_features_par.png)",
            "",
            f"- Grãos corretos (OOF): **{interpret.get('n_correct', '—')}**",
            f"- Grãos errados (OOF): **{interpret.get('n_errors', '—')}**",
            "",
        ]

    if len(artifacts["imp_df"]) > 0:
        lines += [
            "### Importância de variáveis (Random Forest)",
            "",
            "![Importância RF](../outputs/figures/rf_feature_importance.png)",
            "",
            _md_table(artifacts["imp_df"].head(10).round(4)),
            "",
        ]

    lines += [
        "## 7. Clustering exploratório (K-Means)",
        "",
        f"- **k** = {artifacts['n_classes']} (número de espécies)",
        f"- **Adjusted Rand Index (ARI):** {artifacts['ari']:.3f}",
        "- ARI ≈ 1: agrupamento concordante com os rótulos; ≈ 0: aleatório.",
        "",
        "## 8. Artefatos gerados",
        "",
        "| Arquivo | Conteúdo |",
        "| --- | --- |",
        f"| `{artifacts['summary_path'].relative_to(ROOT)}` | Resumo numérico |",
        "| `outputs/figures/confusion_matrix.png` | Matriz de confusão |",
        "| `outputs/figures/ml_confusion_matrix_norm.png` | Matriz normalizada (recall) |",
        "| `outputs/figures/ml_confusion_pares.png` | Pares mais confundidos |",
        "| `outputs/figures/ml_pca_acertos_erros.png` | PCA acertos vs erros |",
        "| `outputs/figures/ml_permutation_importance.png` | Importância por permutação |",
        "| `outputs/figures/ml_shap_importance.png` | SHAP (média |valor|) |",
        "| `outputs/figures/ml_shap_summary.png` | SHAP summary plot |",
        "| `outputs/figures/ml_erros_features_par.png` | Features nos erros vs acertos |",
        "| `outputs/figures/ml_comparacao_metricas.png` | F1-macro e acurácia por modelo |",
        "| `outputs/figures/ml_comparacao_acuracia.png` | Acurácia por modelo |",
        "| `outputs/figures/ml_metricas_por_fold.png` | Métricas por fold (todos os modelos) |",
        "| `outputs/figures/rf_feature_importance.png` | Importância RF (se RF venceu) |",
        "| `docs/resultados_eda.md` | Estatística exploratória |",
        "",
        "## 9. Limitações desta execução",
        "",
        "- Dataset parcial (medições em andamento); resultados podem mudar ao incluir vistas polares e mais espécies.",
        "- Com poucas amostras por classe, o K-Fold pode ter folds desbalanceados.",
        "- Imputação de medianas pode mascarar ausências estruturais; interpretar features de abertura com cautela.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA + ML sobre polen_features_analyze.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=FEATURES_CSV,
        help="CSV de features (default: polen_features_analyze.csv)",
    )
    args = parser.parse_args()

    print(f"Carregando {args.input}...")
    df_raw = load_features(args.input)
    print(f"  {len(df_raw)} linhas, {df_raw['species'].nunique()} espécies")

    print("Executando EDA...")
    _, eda_artifacts = run_eda(df_raw)
    eda_path = write_eda_report(eda_artifacts, args.input)
    print(f"  Relatório EDA: {eda_path}")

    print("Executando ML (K-Fold)...")
    ml_artifacts = run_ml(df_raw)
    ml_path = write_ml_report(ml_artifacts, args.input)
    print(f"  Relatório ML: {ml_path}")
    print(f"  Modelo final: {ml_artifacts['model_name']}")
    best_f1 = max(ml_artifacts["summary"]["f1_macro_rf"], ml_artifacts["summary"]["f1_macro_svm"])
    print(f"  F1-macro ({ml_artifacts['model_name']}): {best_f1:.3f}")


if __name__ == "__main__":
    main()
