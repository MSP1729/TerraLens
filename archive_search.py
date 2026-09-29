"""SatQuery local RGB tile retrieval demo. No geospatial/time metadata inference."""
import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
OUT = ROOT / "archive_index"
MODEL_ID = "chendelong/RemoteCLIP"
WEIGHT_NAME = "RemoteCLIP-ViT-B-32.pt"
_MODEL = None


def model_bundle():
    global _MODEL
    if _MODEL is None:
        try:
            path = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHT_NAME,
                                   local_dir=str(ROOT / "checkpoints"))
        except Exception as exc:
            raise RuntimeError("RemoteCLIP checkpoint is missing or could not download. Connect to the internet once, check free disk space (~605 MB), then retry. For the offline demo, keep the checkpoints directory and do not erase the Hugging Face cache.") from exc
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(checkpoint, strict=True)
        print("Loaded RemoteCLIP:", result)
        model.eval().cpu()
        _MODEL = model, preprocess, open_clip.get_tokenizer("ViT-B-32")
    return _MODEL


def normalize(x):
    x = x.detach().cpu().float().numpy().astype("float32")
    return np.ascontiguousarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))


def embed_image(path):
    model, preprocess, _ = model_bundle()
    with Image.open(path) as img:
        tensor = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        return normalize(model.encode_image(tensor))


def embed_text(query):
    model, _, tokenizer = model_bundle()
    with torch.inference_mode():
        return normalize(model.encode_text(tokenizer([query])))


def _record_for(path, lookup):
    key = path.relative_to(ROOT).as_posix()
    record = dict(lookup.get(key, {"path": key, "label": path.parent.name,
                                   "dataset": "local", "date": None, "geometry": None}))
    record["path"] = key
    return record


def _manifest_lookup():
    path = ARCHIVE / "manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return {item["path"]: item for item in rows}


def _save(index, records):
    if index.ntotal != len(records):
        raise RuntimeError("Index/metadata count mismatch; nothing saved")
    OUT.mkdir(exist_ok=True)
    # Write both temporary files before replacing either. If interrupted between
    # replacements, read-time consistency check will catch the partial update.
    temp_idx, temp_json = OUT / "tiles.faiss.tmp", OUT / "items.json.tmp"
    faiss.write_index(index, str(temp_idx))
    temp_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(temp_idx, OUT / "tiles.faiss")
    os.replace(temp_json, OUT / "items.json")


def _load():
    index = faiss.read_index(str(OUT / "tiles.faiss"))
    records = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    if index.ntotal != len(records):
        raise RuntimeError("Index/items mismatch; rebuild with --build")
    if len({record["path"] for record in records}) != len(records):
        raise RuntimeError("Duplicate indexed paths; rebuild with --build")
    return index, records


def build():
    files = sorted(p for p in ARCHIVE.rglob("*.jpg") if p.is_file())
    if not files:
        raise RuntimeError("No local images: run --prepare first")
    lookup = _manifest_lookup()
    vectors, records = [], []
    for number, path in enumerate(files, 1):
        vectors.append(embed_image(path))
        records.append(_record_for(path, lookup))
        if number % 20 == 0:
            print("Embedded", number, "/", len(files), flush=True)
    vectors = np.concatenate(vectors, axis=0)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    _save(index, records)
    print("Built", index.ntotal, "vectors")


def add():
    """Add only image paths not already in the index; changed images need --build."""
    index, records = _load()
    known = {r["path"] for r in records}
    paths = [p for p in sorted(ARCHIVE.rglob("*.jpg"))
             if p.is_file() and p.relative_to(ROOT).as_posix() not in known]
    if not paths:
        print("No new JPG files. Existing images are not re-embedded; use --build if they changed.")
        return
    lookup = _manifest_lookup()
    for number, path in enumerate(paths, 1):
        vector = embed_image(path)
        if vector.shape[1] != index.d:
            raise RuntimeError("Embedding dimensions changed; use --build")
        index.add(vector)
        records.append(_record_for(path, lookup))
        print("Added", number, "/", len(paths), path.relative_to(ROOT), flush=True)
    _save(index, records)
    print("Index now contains", index.ntotal, "images. Existing image edits require --build.")


def search(query=None, image=None, k=5):
    if (query is None) == (image is None):
        raise ValueError("Provide exactly one of query or image")
    index, records = _load()
    vector = embed_text(query) if query is not None else embed_image(image)
    if vector.shape[1] != index.d:
        raise RuntimeError("Model/index dimensions differ: rebuild")
    scores, ids = index.search(vector, min(k, index.ntotal))
    return [{**records[int(i)], "score": round(float(s), 4), "rank": rank}
            for rank, (i, s) in enumerate(zip(ids[0], scores[0]), 1) if i >= 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--build", action="store_true")
    command.add_argument("--add", action="store_true")
    command.add_argument("--text")
    command.add_argument("--image")
    args = parser.parse_args()
    if args.build:
        build()
    elif args.add:
        add()
    else:
        results = search(query=args.text, image=args.image)
        print(json.dumps(results, indent=2))
        if args.image:
            query_path = Path(args.image).resolve()
            indexed_query = query_path.is_relative_to(ARCHIVE.resolve()) and query_path.suffix.lower() == ".jpg"
            if indexed_query:
                expected = query_path.relative_to(ROOT).as_posix()
                if not results or results[0]["path"] != expected:
                    raise AssertionError(f"Self-match failed: expected {expected} as rank 1. Rebuild index and check model preprocessing.")
                print("Self-match check passed: indexed image ranks first.")