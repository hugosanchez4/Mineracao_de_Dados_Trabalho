from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "data" / "tables"
FEATURES_CSV = TABLES_DIR / "polen_features_analyze.csv"
LEGACY_FEATURES_CSV = TABLES_DIR / "pollen_features.csv"
SPECIES_YAML = PROJECT_ROOT / "config" / "species.yaml"

FEATURE_COLUMNS = [
    "eixo_polar_um",
    "eixo_equatorial_um",
    "comprimento_ectoabertura_um",
    "largura_ectoabertura_um",
    "comprimento_endoabertura_um",
    "largura_endoabertura_um",
    "espessura_exina_um",
    "comprimento_espinho_um",
    "largura_espinho_um",
    "circularidade",
    "area_um2",
    "perimeter_um",
    "feret_um",
    "solidity",
]

# Medidas que só fazem sentido em uma vista (deixe vazio no CSV / NA na outra)
FEATURES_MAINLY_POLAR: list[str] = []

FEATURES_MAINLY_EQUATORIAL = [
    "comprimento_ectoabertura_um",
    "largura_ectoabertura_um",
    "comprimento_endoabertura_um",
    "largura_endoabertura_um",
    "espessura_exina_um",
    "comprimento_espinho_um",
    "largura_espinho_um",
]

FEATURES_BOTH_VIEWS = [
    "eixo_polar_um",
    "eixo_equatorial_um",
    "circularidade",
    "area_um2",
    "perimeter_um",
    "feret_um",
    "solidity",
]

NA_VALUES = ["", "NI", "Ni", "ni", "NA", "N/A", "nan"]


def load_species_config() -> dict:
    with open(SPECIES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or FEATURES_CSV
    if not csv_path.exists():
        if LEGACY_FEATURES_CSV.exists():
            csv_path = LEGACY_FEATURES_CSV
        else:
            raise FileNotFoundError(
                f"Dataset não encontrado: {csv_path}\n"
                "Coloque as medições em data/tables/polen_features_analyze.csv"
            )

    df = pd.read_csv(csv_path, decimal=",", na_values=NA_VALUES, keep_default_na=True)
    if "species" in df.columns:
        df["species"] = df["species"].astype(str).str.strip()

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
