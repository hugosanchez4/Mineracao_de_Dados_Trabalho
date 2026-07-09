"""Parser de nomes de arquivo .CZI do projeto."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Ex.: Cololobus hatschbachii_Renon 978_equatorial_02.czi
CZI_PATTERN = re.compile(
    r"^(?P<species>.+)_(?P<specimen>.+)_(?P<view>polar|equatorial)_(?P<grain_id>\d+)\.czi$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CziFilename:
    species: str
    specimen: str
    view: str
    grain_id: str
    filename: str

    @property
    def species_slug(self) -> str:
        """Slug para pastas/CSV: cololobus_hatschbachii"""
        slug = self.species.strip().lower()
        slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
        slug = re.sub(r"[\s-]+", "_", slug)
        return slug

    @property
    def grain_id_padded(self) -> str:
        return self.grain_id.zfill(3)


def parse_czi_filename(name: str | Path) -> CziFilename:
    """Interpreta o padrão: {espécie}_{código amostra}_{polar|equatorial}_{número}.czi"""
    path = Path(name)
    fname = path.name
    m = CZI_PATTERN.match(fname)
    if not m:
        raise ValueError(
            f"Nome não reconhecido: {fname}\n"
            "Esperado: Nome da espécie_Código equatorial_02.czi\n"
            "Ex.: Cololobus hatschbachii_Renon 978_equatorial_02.czi"
        )
    return CziFilename(
        species=m.group("species").strip(),
        specimen=m.group("specimen").strip(),
        view=m.group("view").lower(),
        grain_id=m.group("grain_id"),
        filename=fname,
    )


def parse_czi_filename_from_parts(parts: list[str]) -> CziFilename | None:
    """Fallback: últimos segmentos = grain_id, view, specimen; resto = espécie."""
    if len(parts) < 4:
        return None
    grain_raw = parts[-1].replace(".czi", "").replace(".CZI", "")
    view = parts[-2].lower()
    if view not in ("polar", "equatorial"):
        return None
    specimen = parts[-3]
    species = "_".join(parts[:-3])
    fname = "_".join(parts)
    if not fname.lower().endswith(".czi"):
        fname = fname + ".czi"
    return CziFilename(
        species=species.replace("_", " ") if " " not in species else species,
        specimen=specimen,
        view=view,
        grain_id=grain_raw,
        filename=fname if fname.endswith(".czi") else fname + ".czi",
    )
