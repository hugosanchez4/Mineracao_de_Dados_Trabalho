"""Interpretabilidade do classificador: SHAP, permutação e análise de erros."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pollen_ml.config import FEATURE_COLUMNS
from pollen_ml.preprocess import clean_dataframe, impute_by_view, mark_not_applicable

RANDOM_STATE = 42


def _species_short(name: str) -> str:
    parts = name.replace("Cololobus ", "C. ").split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name


def get_imputed_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame com features imputadas (µm), sem scaling — para gráficos interpretáveis."""
    work = clean_dataframe(df)
    if "view" in work.columns:
        work = mark_not_applicable(work)
    numeric_cols = [c for c in FEATURE_COLUMNS if c in work.columns]
    work = impute_by_view(work, numeric_cols)
    imputer = SimpleImputer(strategy="median")
    work[numeric_cols] = imputer.fit_transform(work[numeric_cols])
    return work


def confusion_pairs_df(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "real": true_name,
                        "predito": pred_name,
                        "n": int(cm[i, j]),
                        "par": f"{_species_short(true_name)} → {_species_short(pred_name)}",
                    }
                )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def plot_confusion_matrix_normalized(
    cm: np.ndarray, class_names: list[str], fig_dir: Path, *, model_name: str
) -> Path:
    """Matriz normalizada por linha (% do real predito em cada coluna)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0)
    short = [_species_short(c) for c in class_names]
    out = fig_dir / "ml_confusion_matrix_norm.png"
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",
        xticklabels=short,
        yticklabels=short,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusão normalizada (por linha) — {model_name}")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_confusion_pairs(pairs_df: pd.DataFrame, fig_dir: Path) -> Path:
    """Barras dos pares mais confundidos (fora da diagonal)."""
    out = fig_dir / "ml_confusion_pares.png"
    plot_df = pairs_df.head(10).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.45 * len(plot_df))))
    colors = sns.color_palette("Reds_r", n_colors=len(plot_df))
    plt.barh(plot_df["par"], plot_df["n"], color=colors, edgecolor="0.2")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(row["n"] + 0.1, i, str(row["n"]), va="center", fontsize=9)
    plt.xlabel("Número de grãos")
    plt.title("Pares de espécies mais confundidos pelo modelo")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_permutation_importance(
    model,
    X: np.ndarray,
    y_enc: np.ndarray,
    feature_names: list[str],
    fig_dir: Path,
    *,
    model_name: str,
) -> tuple[Path, pd.DataFrame]:
    fitted = clone(model)
    fitted.fit(X, y_enc)
    pi = permutation_importance(
        fitted,
        X,
        y_enc,
        scoring="f1_macro",
        n_repeats=30,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    imp_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    out = fig_dir / "ml_permutation_importance.png"
    top = imp_df.head(14).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="teal", capsize=3)
    plt.xlabel("Queda no F1-macro ao permutar feature (média CV)")
    plt.title(f"Importância por permutação — {model_name}")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out, imp_df


def _fit_explainer_model(model, X: np.ndarray, y_enc: np.ndarray):
    """Clona e ajusta modelo para SHAP (SVM com probability=True se necessário)."""
    explainer_model = clone(model)
    if isinstance(explainer_model, SVC) and not getattr(explainer_model, "probability", False):
        explainer_model.set_params(probability=True)
    explainer_model.fit(X, y_enc)
    return explainer_model


def compute_and_plot_shap(
    model,
    X: np.ndarray,
    y_enc: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    fig_dir: Path,
    *,
    model_name: str,
) -> tuple[list[Path], pd.DataFrame | None]:
    try:
        import shap
    except ImportError:
        return [], None

    explainer_model = _fit_explainer_model(model, X, y_enc)
    paths: list[Path] = []

    if isinstance(explainer_model, RandomForestClassifier):
        explainer = shap.TreeExplainer(explainer_model)
        shap_values = explainer.shap_values(X)
    else:
        n_bg = min(25, len(X))
        background = shap.kmeans(X, n_bg)
        explainer = shap.KernelExplainer(explainer_model.predict_proba, background)
        nsamples = min(200, 2 * len(X) + 1)
        shap_values = explainer.shap_values(X, nsamples=nsamples)

    if isinstance(shap_values, list):
        abs_stack = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = abs_stack.mean(axis=(0, 1))
    else:
        sv = np.abs(np.asarray(shap_values))
        if sv.ndim == 3:
            if sv.shape[-1] == len(class_names):
                mean_abs = sv.mean(axis=(0, 2))
            elif sv.shape[0] == len(class_names):
                mean_abs = sv.mean(axis=(0, 1))
            else:
                mean_abs = sv.reshape(-1, sv.shape[-1]).mean(axis=0)
        else:
            mean_abs = sv.mean(axis=0)
    mean_abs = np.asarray(mean_abs).ravel()[: len(feature_names)]

    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_bar = fig_dir / "ml_shap_importance.png"
    top = shap_df.head(14).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="mediumpurple", edgecolor="0.2")
    plt.xlabel("Média |SHAP| (todas as classes)")
    plt.title(f"Importância SHAP — {model_name}")
    plt.tight_layout()
    plt.savefig(out_bar, bbox_inches="tight")
    plt.close()
    paths.append(out_bar)

    out_summary = fig_dir / "ml_shap_summary.png"
    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            class_names=[_species_short(c) for c in class_names],
            show=False,
            max_display=14,
        )
    else:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=14)
    plt.title(f"SHAP summary — {model_name}")
    plt.tight_layout()
    plt.savefig(out_summary, bbox_inches="tight")
    plt.close()
    paths.append(out_summary)

    if isinstance(shap_values, list):
        mis_mask = explainer_model.predict(X) != y_enc
        if mis_mask.sum() >= 3:
            class_idx = int(np.bincount(y_enc[mis_mask]).argmax())
            out_err = fig_dir / "ml_shap_erros.png"
            plt.figure()
            shap.summary_plot(
                shap_values[class_idx][mis_mask],
                X[mis_mask],
                feature_names=feature_names,
                show=False,
                max_display=14,
            )
            plt.title(
                f"SHAP — grãos classificados errado (classe real: {_species_short(class_names[class_idx])})"
            )
            plt.tight_layout()
            plt.savefig(out_err, bbox_inches="tight")
            plt.close()
            paths.append(out_err)

    return paths, shap_df


def plot_pca_acertos_erros(
    X: np.ndarray,
    y_enc: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    fig_dir: Path,
) -> Path:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    plot_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "species": [class_names[i] for i in y_enc],
            "status": np.where(y_enc == y_pred, "Correto", "Erro"),
        }
    )
    out = fig_dir / "ml_pca_acertos_erros.png"
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="species",
        style="status",
        markers={"Correto": "o", "Erro": "X"},
        s=70,
        alpha=0.85,
    )
    plt.title("PCA — acertos vs erros do modelo (predição OOF)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_erros_features_par(
    feat_df: pd.DataFrame,
    y_enc: np.ndarray,
    y_pred: np.ndarray,
    pairs_df: pd.DataFrame,
    top_features: list[str],
    class_names: list[str],
    fig_dir: Path,
    *,
    n_pairs: int = 3,
) -> Path:
    """Boxplots: grãos do par confuso vs acertos da espécie real."""
    available = [f for f in top_features if f in feat_df.columns]
    if not available or pairs_df.empty:
        return fig_dir / "ml_erros_features_par.png"

    rows = []
    for _, par in pairs_df.head(n_pairs).iterrows():
        real_idx = class_names.index(par["real"])
        pred_idx = class_names.index(par["predito"])
        mask_real = y_enc == real_idx
        mask_conf = mask_real & (y_pred == pred_idx)
        mask_ok = mask_real & (y_pred == real_idx)
        for feat in available[:6]:
            for grupo, mask in [
                (f"Erro: {par['par']}", mask_conf),
                (f"Acerto: {_species_short(par['real'])}", mask_ok),
            ]:
                if mask.sum() == 0:
                    continue
                for val in feat_df.loc[mask, feat]:
                    rows.append({"par": par["par"], "grupo": grupo, "feature": feat, "valor": val})

    if not rows:
        return fig_dir / "ml_erros_features_par.png"

    long_df = pd.DataFrame(rows)
    out = fig_dir / "ml_erros_features_par.png"
    g = sns.catplot(
        data=long_df,
        x="feature",
        y="valor",
        hue="grupo",
        col="par",
        kind="box",
        sharey=False,
        height=4,
        aspect=1.1,
        col_wrap=min(n_pairs, 3),
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.fig.suptitle("Features nos erros vs acertos (principais pares confusos)", y=1.03)
    g.savefig(out, bbox_inches="tight")
    plt.close("all")
    return out


def generate_interpretation_plots(
    df_raw: pd.DataFrame,
    X: np.ndarray,
    y_enc: np.ndarray,
    y_pred: np.ndarray,
    final_model,
    feature_names: list[str],
    class_names: list[str],
    cm: np.ndarray,
    fig_dir: Path,
    *,
    model_name: str,
) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    pairs_df = confusion_pairs_df(cm, class_names)
    paths.append(plot_confusion_matrix_normalized(cm, class_names, fig_dir, model_name=model_name))
    if not pairs_df.empty:
        paths.append(plot_confusion_pairs(pairs_df, fig_dir))

    paths.append(plot_pca_acertos_erros(X, y_enc, y_pred, class_names, fig_dir))

    perm_path, perm_df = plot_permutation_importance(
        final_model, X, y_enc, feature_names, fig_dir, model_name=model_name
    )
    paths.append(perm_path)

    shap_paths, shap_df = compute_and_plot_shap(
        final_model,
        X,
        y_enc,
        feature_names,
        class_names,
        fig_dir,
        model_name=model_name,
    )
    paths.extend(shap_paths)

    feat_df = get_imputed_feature_frame(df_raw)
    top_feats = perm_df["feature"].tolist()
    numeric_feats = [f for f in top_feats if f in feat_df.columns]
    err_path = plot_erros_features_par(
        feat_df,
        y_enc,
        y_pred,
        pairs_df,
        numeric_feats,
        class_names,
        fig_dir,
    )
    if err_path.exists():
        paths.append(err_path)

    n_errors = int((y_enc != y_pred).sum())
    n_correct = int((y_enc == y_pred).sum())

    return {
        "paths": paths,
        "pairs_df": pairs_df,
        "perm_df": perm_df,
        "shap_df": shap_df,
        "n_errors": n_errors,
        "n_correct": n_correct,
    }
