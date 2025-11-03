"""Utilities for restructuring the mine laboratory CSV exports into a tidy tabular dataset.

The original CSV files contain pairs of points laid out side-by-side exactly as they
were exported from Excel. Each point lists the stratigraphy (Material, Cohesion,
Friction Angle, Unit Weight) for a set of layers, and only the first material row
stores the Factor of Safety (FoS). The helpers in this module read those sheets and
emit one record per survey point that the machine learning pipeline can consume.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# Pattern used to isolate the numeric component from labels such as "Point 3"
_POINT_PATTERN = re.compile(r"Point\s*(\d+)", re.IGNORECASE)


@dataclass
class PointMetadata:
    """Metadata tracked for each point during parsing."""

    point_label: str
    mine_label: str
    season: str
    source_file: Path

    def as_dict(self) -> Dict[str, object]:
        base: Dict[str, object] = {
            "point_label": self.point_label,
            "mine_label": self.mine_label,
            "season": self.season,
            "source_file": self.source_file.name,
        }
        match = _POINT_PATTERN.search(self.point_label)
        if match:
            base["point_index"] = int(match.group(1))
        return base


def _clean_feature_name(material: str, suffix: str) -> str:
    """Return a snake_case feature name for a material/suffix combination."""

    token = material.strip().lower().replace("/", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return f"{token}_{suffix}" if token else suffix


def _to_float(value: Optional[str]) -> Optional[float]:
    """Convert a string value to float when possible."""

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _ensure_point(buffer: Dict[str, Dict[str, object]], metadata: PointMetadata) -> Dict[str, object]:
    """Retrieve or initialise the accumulator dictionary for a point."""

    if metadata.point_label not in buffer:
        buffer[metadata.point_label] = metadata.as_dict()
    return buffer[metadata.point_label]


def _ingest_row(
    row: List[str],
    left_meta: Optional[PointMetadata],
    right_meta: Optional[PointMetadata],
    point_store: Dict[str, Dict[str, object]],
) -> None:
    """Consume a single data row describing up to two points."""

    if left_meta is None and right_meta is None:
        return

    # The export always keeps left-hand values in columns 1-5 and right-hand values in 7-11.
    if left_meta is not None and len(row) >= 6:
        material = row[1].strip()
        if material and material.lower() != "material":
            point_dict = _ensure_point(point_store, left_meta)
            cohesion = _to_float(row[2] if len(row) > 2 else None)
            friction = _to_float(row[3] if len(row) > 3 else None)
            unit_weight = _to_float(row[4] if len(row) > 4 else None)
            fos = _to_float(row[5] if len(row) > 5 else None)

            if cohesion is not None:
                point_dict[_clean_feature_name(material, "cohesion_kpa")] = cohesion
            if friction is not None:
                point_dict[_clean_feature_name(material, "friction_angle_deg")] = friction
            if unit_weight is not None:
                point_dict[_clean_feature_name(material, "unit_weight_kn_per_m3")] = unit_weight
            if fos is not None:
                point_dict["fos"] = fos

    if right_meta is not None and len(row) >= 12:
        material = row[7].strip() if len(row) > 7 and row[7] else ""
        if material and material.lower() != "material":
            point_dict = _ensure_point(point_store, right_meta)
            cohesion = _to_float(row[8] if len(row) > 8 else None)
            friction = _to_float(row[9] if len(row) > 9 else None)
            unit_weight = _to_float(row[10] if len(row) > 10 else None)
            fos = _to_float(row[11] if len(row) > 11 else None)

            if cohesion is not None:
                point_dict[_clean_feature_name(material, "cohesion_kpa")] = cohesion
            if friction is not None:
                point_dict[_clean_feature_name(material, "friction_angle_deg")] = friction
            if unit_weight is not None:
                point_dict[_clean_feature_name(material, "unit_weight_kn_per_m3")] = unit_weight
            if fos is not None:
                point_dict["fos"] = fos


def _iter_records_from_file(file_path: Path, season: str) -> Iterable[Dict[str, object]]:
    """Yield one dictionary per point from a single CSV export."""

    mine_label = file_path.stem
    point_store: Dict[str, Dict[str, object]] = {}
    left_meta: Optional[PointMetadata] = None
    right_meta: Optional[PointMetadata] = None

    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            # Normalise row length so we can rely on positional indexing later.
            row = [cell.strip() for cell in raw_row]
            if not any(row):
                continue

            left_label = row[1] if len(row) > 1 else ""
            right_label = row[7] if len(row) > 7 else ""
            left_is_point = bool(left_label) and left_label.lower().startswith("point")
            right_is_point = bool(right_label) and right_label.lower().startswith("point")

            if left_is_point or right_is_point:
                left_meta = PointMetadata(left_label, mine_label, season, file_path) if left_is_point else None
                right_meta = PointMetadata(right_label, mine_label, season, file_path) if right_is_point else None
                # Ensure metadata entries are created so downstream rows can rely on them.
                if left_meta is not None:
                    _ensure_point(point_store, left_meta)
                if right_meta is not None:
                    _ensure_point(point_store, right_meta)
                continue

            if left_label.lower() == "material":
                # Header row within a block, nothing to ingest yet.
                continue

            _ingest_row(row, left_meta, right_meta, point_store)

    return point_store.values()


def load_dataset(base_dir: Path) -> pd.DataFrame:
    """Aggregate all CSV exports into a single tidy DataFrame.

    Parameters
    ----------
    base_dir:
        Root directory containing the seasonal sub-folders (e.g. ``postmonsoonwithoutru``).

    Returns
    -------
    pandas.DataFrame
        A dataframe containing one row per survey point with engineered features and
        the target FoS column.
    """

    records: List[Dict[str, object]] = []

    for season_dir in ("postmonsoonwithoutru", "premonsoonwithoutru"):
        folder = base_dir / season_dir
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            records.extend(_iter_records_from_file(csv_path, season_dir))

    if not records:
        raise FileNotFoundError(
            "No records were ingested. Please check that the CSV exports are present in the expected directories."
        )

    frame = pd.DataFrame(records)
    frame = frame.dropna(subset=["fos"])  # ensure the target is defined
    frame["fos"] = frame["fos"].astype(float)

    cohesion_cols = [col for col in frame.columns if col.endswith("cohesion_kpa")]
    friction_cols = [col for col in frame.columns if col.endswith("friction_angle_deg")]
    unit_weight_cols = [col for col in frame.columns if col.endswith("unit_weight_kn_per_m3")]

    # Compute aggregated helper features that summarise the stratigraphy.
    if cohesion_cols:
        frame["mean_cohesion_kpa"] = frame[cohesion_cols].mean(axis=1)
    if friction_cols:
        frame["mean_friction_angle_deg"] = frame[friction_cols].mean(axis=1)
    if unit_weight_cols:
        frame["mean_unit_weight_kn_per_m3"] = frame[unit_weight_cols].mean(axis=1)

    # Season is categorical; normalise spelling to make downstream encoding predictable.
    frame["season"] = (
        frame["season"].str.replace("withoutru", "", regex=False).str.replace("-", "_", regex=False).str.rstrip("_")
    )

    return frame.reset_index(drop=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset = load_dataset(base_dir)
    output_path = base_dir / "calculations" / "prepared_dataset.parquet"
    dataset.to_parquet(output_path, index=False)
    print(f"Prepared dataset written to {output_path.relative_to(base_dir)} with {len(dataset)} points.")


if __name__ == "__main__":
    main()
