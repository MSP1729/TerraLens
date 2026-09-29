"""Prepare a small local EuroSAT RGB demo collection for SatQuery.

Run after installing datasets: python prepare_archive.py
The 64x64 RGB JPEGs have labels but no usable acquisition dates or map coordinates.
"""
import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
CLASSES = {1: "Forest", 4: "Industrial", 7: "Residential", 8: "River", 9: "SeaLake"}
PER_CLASS = 20


def main():
    ARCHIVE.mkdir(exist_ok=True)
    counts = {label: 0 for label in CLASSES}
    records = []
    print("Streaming the EuroSAT RGB training split. Internet is needed this first time.", flush=True)
    dataset = load_dataset("giswqs/EuroSAT_RGB", split="train", streaming=True)
    for row in dataset:
        label = int(row["label"])
        if label not in CLASSES or counts[label] >= PER_CLASS:
            continue
        name = CLASSES[label]
        folder = ARCHIVE / name
        folder.mkdir(exist_ok=True)
        path = folder / f"{name}_{counts[label]:03}.jpg"
        row["image"].convert("RGB").save(path, format="JPEG")
        records.append({"path": path.relative_to(ROOT).as_posix(),
                        "label": name, "dataset": "giswqs/EuroSAT_RGB",
                        "original_filename": str(row.get("filename") or ""),
                        "date": None, "geometry": None})
        counts[label] += 1
        if all(count == PER_CLASS for count in counts.values()):
            break
    (ARCHIVE / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Saved", len(records), "images:", {CLASSES[k]: v for k, v in counts.items()})
    if len(records) != PER_CLASS * len(CLASSES):
        raise RuntimeError("Only saved some classes. Review the dataset rows; do not build an incomplete demo without checking.")


if __name__ == "__main__":
    main()
