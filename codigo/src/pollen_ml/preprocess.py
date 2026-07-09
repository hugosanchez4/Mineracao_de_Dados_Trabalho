from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pollen_ml.config import (
    FEATURE_COLUMNS,
    FEATURES_BOTH_VIEWS,
    FEATURES_MAINLY_EQUATORIAL,
    FEATURES_MAINLY_POLAR,
)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "valid" in out.columns:
        out = out[out["valid"] == 1]
    return out.reset_index(drop=True)


def mark_not_applicable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante NA onde a medida não existe naquela vista (não é erro de coleta).
    Preencha só o que mediu no Fiji; o resto pode ficar vazio.
    """
    out = df.copy()
    if "view" not in out.columns:
        return out

    polar = out["view"].str.lower() == "polar"
    equatorial = out["view"].str.lower() == "equatorial"

    for col in FEATURES_MAINLY_EQUATORIAL:
        if col in out.columns:
            out.loc[polar, col] = np.nan

    for col in FEATURES_MAINLY_POLAR:
        if col in out.columns:
            out.loc[equatorial, col] = np.nan

    return out


def impute_by_view(
    df: pd.DataFrame,
    columns: list[str],
    *,
    view_col: str = "view",
) -> pd.DataFrame:
    """
    Imputa mediana apenas dentro da mesma vista.
    Evita preencher linha polar com mediana de aberturas da equatorial.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for view in out[view_col].dropna().unique():
            mask = out[view_col].str.lower() == str(view).lower()
            subset = out.loc[mask, col]
            if subset.notna().sum() == 0:
                continue
            med = subset.median()
            out.loc[mask & out[col].isna(), col] = med
    return out


def prepare_xy(
    df: pd.DataFrame,
    *,
    include_view: bool = True,
    view_aware: bool = True,
    impute_strategy: str = "median",
    drop_high_missing: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, list[str], Pipeline]:
    """
    Retorna X, y (species), feature_names após imputação e scaling.

    view_aware=True: NA estrutural por vista + imputação por grupo de vista.
    """
    work = clean_dataframe(df)
    if view_aware:
        work = mark_not_applicable(work)
        work = impute_by_view(work, [c for c in FEATURE_COLUMNS if c in work.columns])

    y = work["species"].astype(str).values

    numeric_cols = [c for c in FEATURE_COLUMNS if c in work.columns]
    # Remove colunas quase sempre vazias (opcional)
    if drop_high_missing < 1.0:
        rates = work[numeric_cols].isna().mean()
        numeric_cols = [c for c in numeric_cols if rates[c] < drop_high_missing]

    if view_aware:
        work = impute_by_view(work, numeric_cols)
    imputer = SimpleImputer(strategy=impute_strategy)
    work[numeric_cols] = imputer.fit_transform(work[numeric_cols])

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    if include_view and "view" in work.columns:
        preprocess = ColumnTransformer(
            [
                ("num", numeric_pipe, numeric_cols),
                ("view", OneHotEncoder(drop=None, handle_unknown="ignore"), ["view"]),
            ]
        )
        feature_names = numeric_cols + ["view_polar", "view_equatorial"]
    else:
        preprocess = ColumnTransformer([("num", numeric_pipe, numeric_cols)])
        feature_names = numeric_cols

    X = preprocess.fit_transform(work)
    return X, y, feature_names, preprocess


def build_features(
    df: pd.DataFrame,
    *,
    include_view: bool = True,
    view_aware: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], bool]:
    """
    Prepara os dados para validação cruzada SEM imputar nem escalar.

    A imputação (mediana) e a padronização (StandardScaler) ficam a cargo do
    transformador de `make_preprocessor`, que deve ser encaixado num Pipeline e
    ajustado DENTRO de cada fold da validação cruzada — evitando vazamento de
    dados (o scaler/imputer não "enxerga" o fold de teste).

    Retorna: (X_in, y, numeric_cols, feature_names, include_view_efetivo).
    """
    work = clean_dataframe(df)
    if view_aware:
        work = mark_not_applicable(work)

    y = work["species"].astype(str).values
    numeric_cols = [c for c in FEATURE_COLUMNS if c in work.columns]

    use_view = bool(include_view and "view" in work.columns and work["view"].nunique() > 1)
    cols = numeric_cols + (["view"] if use_view else [])
    X_in = work[cols].copy()

    feature_names = list(numeric_cols)
    if use_view:
        cats = sorted(work["view"].dropna().unique())
        feature_names += [f"view_{c}" for c in cats]

    return X_in, y, numeric_cols, feature_names, use_view


def make_preprocessor(
    numeric_cols: list[str],
    *,
    include_view: bool = False,
    impute_strategy: str = "median",
) -> ColumnTransformer:
    """Transformador (não ajustado) com imputação de mediana + StandardScaler.

    Deve ser usado dentro de um Pipeline junto do estimador, para ser reajustado
    a cada fold da validação cruzada.
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [("num", numeric_pipe, numeric_cols)]
    if include_view:
        transformers.append(
            ("view", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["view"])
        )
    return ColumnTransformer(transformers)


def filter_by_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """Subset só polar ou só equatorial — útil para modelos separados."""
    return df[df["view"].str.lower() == view.lower()].copy()
