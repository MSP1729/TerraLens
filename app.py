"""SatQuery AI: laptop UI with optional remote Colab vision-language inference.
Run: python app.py   (Windows venv: python app.py)
Set COLAB_URL to the live https://...gradio.live URL before running.
"""
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file

try:
    import rasterio
except ImportError:
    rasterio = None

COLAB_URL = "https://31777f5c4d0fb6e6bb.gradio.live"  # update if Colab restarts
MAX_MB = 150
MAX_PIXELS = 36_000_000
OUT = Path(tempfile.gettempdir()) / "satquery_outputs"
OUT.mkdir(exist_ok=True)
EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def scale8(x):
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    return (np.clip((np.nan_to_num(x, nan=float(lo)) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load(path, modality):
    if not path:
        raise ValueError("Upload the required image(s).")
    p = Path(path)
    if p.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Unsupported format: {p.suffix}. Use GeoTIFF/TIFF/PNG/JPEG.")
    if p.stat().st_size > MAX_MB * 1024 * 1024:
        raise ValueError(f"{p.name} exceeds the {MAX_MB} MB limit.")
    meta = {"name": p.name, "format": p.suffix.lower(), "bytes": p.stat().st_size,
            "modality": modality, "crs": None, "transform": None, "georeferenced": False}
    if p.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ValueError("GeoTIFF needs rasterio. Install it with pip install rasterio.")
        with rasterio.open(p) as src:
            w, h, n = src.width, src.height, src.count
            if w * h > MAX_PIXELS:
                raise ValueError(f"{p.name} has too many pixels ({w} x {h}); crop it first.")
            if n < 1:
                raise ValueError(f"{p.name} contains no raster bands.")
            meta.update(width=w, height=h, bands=n, crs=str(src.crs) if src.crs else None,
                        transform=tuple(src.transform)[:6] if src.crs else None,
                        georeferenced=bool(src.crs))
            # Read all bands.
            a = src.read(indexes=list(range(1,n+1)), masked=True)
            data = np.asarray(a.astype(np.float32).filled(np.nan), dtype=np.float32)
            valid = np.all(np.isfinite(data), axis=0)
    else:
        with Image.open(p) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ValueError(f"{p.name} has too many pixels ({w} x {h}); resize it first.")
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        data = np.moveaxis(a.astype(np.float32), -1, 0)
        valid = np.ones((h, w), dtype=bool)
        meta.update(width=w, height=h, bands=3)
    if not valid.any():
        raise ValueError(f"{p.name} has no finite pixels.")
    if data.shape[0] < 3 or modality == "SAR":
        gray = scale8(data[0]); rgb = np.stack([gray] * 3, axis=-1)
    else:
        rgb = np.stack([scale8(data[k]) for k in range(3)], axis=-1)
    return {"data": data, "valid": valid, "rgb": rgb, "meta": meta}


def same_grid(a, b):
    x, y = a["meta"], b["meta"]
    if (x["width"], x["height"]) != (y["width"], y["height"]):
        raise ValueError("Image sizes differ. Reproject/resample to the same grid first.")
    if bool(x["crs"]) != bool(y["crs"]):
        raise ValueError("One image is georeferenced and the other is not. Align them first.")
    if x["crs"] and (x["crs"] != y["crs"] or not np.allclose(x["transform"], y["transform"], atol=1e-8, rtol=0)):
        raise ValueError("CRS or pixel transforms differ. Reproject/resample to an identical grid first.")
    # Identical dimensions alone do not establish true co-registration for PNG/JPEG.
    return "Georeferenced grids match." if x["crs"] else "Same pixel dimensions; PNG/JPEG alignment cannot be verified."


def small_image(rgb, limit=1200):
    im = Image.fromarray(rgb)
    im.thumbnail((limit, limit))
    return im


def remote_ask(item, question, trace):
    if not COLAB_URL.strip():
        trace.append({"tool": "remote_vqa", "status": "skipped", "reason": "COLAB_URL is blank"})
        return "Remote VQA unavailable: add the live Colab URL at the top of app.py."
    url = COLAB_URL.strip()
    if not re.fullmatch(r"https://[\w-]+\.gradio\.live/?", url):
        trace.append({"tool": "remote_vqa", "status": "skipped", "reason": "invalid URL"})
        return "Remote VQA unavailable: COLAB_URL must be a live https://...gradio.live URL."
    image_path = OUT / (next(tempfile._get_candidate_names()) + ".png")
    small_image(item["rgb"]).save(image_path)
    try:
        client = Client(url)
        # The user's two-input Gradio Interface normally exposes /predict.
        ans = client.predict(handle_file(str(image_path)), question, api_name="/ask")
        trace.append({"tool": "remote_vqa", "params": {"endpoint": url, "question": question, "max_image_side": 1200}, "status": "ok"})
        return str(ans)
    except Exception as exc:
        trace.append({"tool": "remote_vqa", "params": {"endpoint": url, "question": question}, "status": "error", "error": str(exc)[:300]})
        return "Remote VQA failed. Check that Colab is running and its live URL has not changed. Detail: " + str(exc)[:250]
    finally:
        image_path.unlink(missing_ok=True)


def render_mask(item, mask, color=(255, 65, 40)):
    rgb = item["rgb"].copy()
    rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return small_image(rgb)


def save_report(summary, trace, metrics, warnings, gallery):
    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = OUT / key
    root.mkdir()
    images = []
    for i, (im, caption) in enumerate(gallery, 1):
        name = f"evidence_{i}.png"
        im.save(root / name)
        images.append({"file": name, "caption": caption})
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "summary": summary,
              "metrics": metrics, "warnings": warnings, "execution_trace": trace, "evidence": images,
              "note": "Exploratory output, not calibrated classification or field-verified ground truth."}
    with open(root / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    import zipfile
    destination = OUT / (key + ".zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in root.iterdir():
            z.write(file, file.name)
    return str(destination)


def run(task, modality1, modality2, file1, file2, question, green_band, nir_band):
    """Registry-driven controller. Band numbers are 1-based for optical TIFFs."""
    trace, gallery, warnings, metrics = [], [], [], {}
    registry = {"inspect": "validate files/bands/grid", "remote_vqa": "Colab satellite VLM",
                "change": "pixelwise RGB difference", "fusion": "optical + SAR proxy masks", "evidence": "overlay + report"}
    try:
        question = (question or "").strip()
        intent = task
        if task == "Auto":
            q = question.lower()
            if file2 and ("change" in q or "before" in q or "after" in q or "temporal" in q):
                intent = "Change detection"
            elif file2 and {modality1, modality2} == {"Optical", "SAR"}:
                intent = "Optical + SAR fusion"
            elif "caption" in q or "describe" in q or "scene" in q:
                intent = "Scene caption"
            else:
                intent = "VQA"
        tools = ["inspect", "evidence"]
        if intent in ("VQA", "Scene caption"):
            tools.insert(1, "remote_vqa")
        elif intent == "Change detection":
            tools[1:1] = ["change", "remote_vqa"]
        else:
            tools[1:1] = ["fusion", "remote_vqa"]
        trace.append({"controller": "rule-based", "task": intent, "input_config": [modality1, modality2 if file2 else None],
                      "tools_selected": tools, "registry": registry, "question": question})
        one = load(file1, modality1)
        two = load(file2, modality2) if file2 else None
        trace.append({"tool": "inspect", "status": "ok", "inputs": [one["meta"]] + ([two["meta"]] if two else [])})
        gallery.append((small_image(one["rgb"]), "Image 1 preview"))
        if two:
            gallery.append((small_image(two["rgb"]), "Image 2 preview"))
        if intent in ("VQA", "Scene caption"):
            prompt = question or ("Describe the land cover and major objects visible in this image." if intent == "Scene caption" else "What are the main objects and land cover in this image?")
            ans = remote_ask(one, prompt, trace)
            summary = ans
            confidence = "Uncalibrated: remote model does not supply a validated probability."
        elif intent == "Change detection":
            if two is None:
                raise ValueError("Change detection needs two aligned images.")
            if modality1 != modality2:
                raise ValueError("Change detection needs two images of the same modality.")
            alignment = same_grid(one, two)
            warnings.append(alignment)
            both = one["valid"] & two["valid"]
            diff = np.abs(one["rgb"].astype(np.float32) - two["rgb"].astype(np.float32)).mean(axis=2)
            threshold = max(25.0, float(np.percentile(diff[both], 85)))
            mask = (diff >= threshold) & both
            pct = 100 * mask.sum() / both.sum()
            metrics.update(change_threshold_8bit=round(threshold, 2), changed_valid_pixels_percent=round(float(pct), 2), valid_pixels=int(both.sum()))
            gallery.append((render_mask(two, mask), "Red: pixels with strong display-scale difference (not ground-truth change)"))
            trace.append({"tool": "change", "status": "ok", "params": {"method": "mean absolute display-RGB difference", "threshold_8bit": threshold, "same_grid": alignment}})
            # Shared threshold across both dates makes the brightness proxy comparable.
            before_brightness = one["rgb"].mean(axis=2)
            after_brightness = two["rgb"].mean(axis=2)
            bright_threshold = float(np.percentile(np.concatenate((before_brightness[both], after_brightness[both])), 75))
            built_before_pct = 100 * np.count_nonzero((before_brightness >= bright_threshold) & both) / both.sum()
            built_after_pct = 100 * np.count_nonzero((after_brightness >= bright_threshold) & both) / both.sum()
            built_delta_pp = built_after_pct - built_before_pct
            metrics.update(bright_built_proxy_before_percent=round(built_before_pct, 2),
                           bright_built_proxy_after_percent=round(built_after_pct, 2),
                           bright_built_proxy_change_percentage_points=round(built_delta_pp, 2))
            trace.append({"tool": "change", "status": "ok", "params": {"brightness_proxy": "RGB mean >= pooled 75th percentile",
                          "threshold_8bit": round(bright_threshold, 2)}, "note": "Exploratory bright-surface proxy; not confirmed buildings."})
            before = remote_ask(one, "Describe the land cover and major objects in this image briefly.", trace)
            after = remote_ask(two, "Describe the land cover and major objects in this image briefly.", trace)
            summary = (f"Strong display-scale pixel difference: {pct:.1f}% of valid area.\n"
                       f"Bright built-up proxy: {built_before_pct:.1f}% before -> {built_after_pct:.1f}% after "
                       f"({built_delta_pp:+.1f} percentage points).\n\n"
                       f"Before (VLM): {before}\n\nAfter (VLM): {after}\n\n"
                       "Compare these descriptions cautiously; this is not a verified land-cover transition.")
            warnings.append("Bright built-up proxy is only a brightness estimate, not building detection; lighting, season, clouds, speckle, or registration may cause false positives. Independently verify geolocation and acquisition dates.")
            confidence = "Heuristic only; no calibrated confidence."
        elif intent == "Optical + SAR fusion":
            if two is None or {modality1, modality2} != {"Optical", "SAR"}:
                raise ValueError("Fusion needs one Optical and one SAR image.")
            optical, sar = (one, two) if modality1 == "Optical" else (two, one)
            alignment = same_grid(optical, sar)
            warnings.append(alignment)
            valid = optical["valid"] & sar["valid"]
            if not valid.any():
                raise ValueError("No common valid pixels.")
            s = sar["data"][0].astype(np.float32)
            # A low pixel value is a water proxy only for linear intensity or dB backscatter,
            # not for arbitrary styled SAR thumbnails or unknown band conventions.
            s_threshold = float(np.percentile(s[valid], 25))
            low_sar = (s <= s_threshold) & valid
            o = optical["data"]
            gb, nb = int(green_band), int(nir_band)
            if optical["meta"]["format"] in {".png", ".jpg", ".jpeg"}:
                ndwi = None
                warnings.append("Optical PNG/JPEG is RGB; NDWI is unavailable without a known NIR band.")
            elif gb < 1 or nb < 1 or gb == nb or max(gb, nb) > o.shape[0]:
                ndwi = None
                warnings.append("Green/NIR band indices missing or outside the first four TIFF bands; NDWI unavailable.")
            else:
                green, nir = o[gb - 1], o[nb - 1]
                ndwi = (green - nir) / (green + nir + 1e-6)
            if ndwi is not None:
                water = low_sar & (ndwi > 0) & valid
                trace.append({"tool": "fusion", "status": "ok", "params": {"green_band_1based": gb, "nir_band_1based": nb, "ndwi_threshold": 0, "sar_low_quantile": 25}})
            else:
                water = low_sar
                trace.append({"tool": "fusion", "status": "ok", "params": {"sar_low_quantile": 25, "optical_index": "unavailable"}})
            brightness = optical["rgb"].mean(axis=2)
            bright_cutoff = float(np.percentile(brightness[valid], 75))
            built_proxy = (brightness >= bright_cutoff) & (~water) & valid
            metrics.update(water_proxy_percent=round(float(100 * water.sum() / valid.sum()), 2),
                           bright_built_proxy_percent=round(float(100 * built_proxy.sum() / valid.sum()), 2),
                           sar_low_threshold=round(s_threshold, 5), valid_pixels=int(valid.sum()))
            fused = optical["rgb"].copy()
            fused[water] = (0.4 * fused[water] + 0.6 * np.array([20, 115, 255])).astype(np.uint8)
            fused[built_proxy] = (0.4 * fused[built_proxy] + 0.6 * np.array([255, 45, 45])).astype(np.uint8)
            gallery.append((small_image(fused), "Blue: fused water proxy; red: bright built-up proxy (60% color overlay)"))
            desc = remote_ask(optical, question or "Describe visible water and built-up areas in this satellite image.", trace)
            summary = (f"Fused water proxy: {metrics['water_proxy_percent']}% of valid pixels.\n"
                       f"Bright built-up proxy: {metrics['bright_built_proxy_percent']}%.\n\n"
                       f"Optical VLM description (may be wrong): {desc}\n\n"
                       "These are exploratory masks, not validated classes.")
            warnings.append("SAR water proxy assumes low backscatter means water; confirm band, scale, polarization and preprocessing. Bright surfaces are not necessarily built-up.")
            confidence = "Heuristic only; no calibrated confidence."
        else:
            raise ValueError("Unknown task selection.")
        trace.append({"tool": "evidence", "status": "ok", "items": len(gallery)})
        report_path = save_report(summary, trace, metrics, warnings, gallery)
        details = "\n".join("- " + w for w in warnings) or "- None"
        text = f"{summary}\n\nConfidence: {confidence}\n\nWarnings:\n{details}\n\nMetrics: {json.dumps(metrics)}"
        return text, gallery, json.dumps(trace, indent=2, default=str), report_path
    except Exception as exc:
        trace.append({"status": "blocked", "error": str(exc)})
        return "Cannot run: " + str(exc), gallery, json.dumps(trace, indent=2, default=str), None


with gr.Blocks(title="SatQuery AI") as demo:
    gr.Markdown("# SatQuery AI\nSatellite image questions, change maps, and optical-SAR exploratory fusion. Model inference runs on the configured Colab link; local calculations run on your laptop. **Do not upload private imagery to a public Colab share link.**")
    with gr.Row():
        task = gr.Dropdown(["Auto", "VQA", "Scene caption", "Change detection", "Optical + SAR fusion"], value="Auto", label="Task")
        question = gr.Textbox(label="Question", placeholder="Describe the land cover / What changed?", lines=2)
    with gr.Row():
        file1 = gr.File(label="Image 1 (required)", type="filepath", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        modality1 = gr.Dropdown(["Optical", "SAR"], value="Optical", label="Image 1 modality")
    with gr.Row():
        file2 = gr.File(label="Image 2 (change/fusion only)", type="filepath", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        modality2 = gr.Dropdown(["Optical", "SAR"], value="Optical", label="Image 2 modality")
    with gr.Row():
        green = gr.Number(value=2, precision=0, label="Green band (1-based TIFF band; fusion)")
        nir = gr.Number(value=4, precision=0, label="NIR band (1-based TIFF band; fusion)")
    button = gr.Button("Analyze", variant="primary")
    answer = gr.Textbox(label="Result / caveats", lines=12)
    gallery = gr.Gallery(label="Visual evidence", columns=2, height=400)
    trace = gr.Code(label="Execution trace", language="json")
    report = gr.File(label="Download report and images (ZIP)")
    button.click(run, inputs=[task, modality1, modality2, file1, file2, question, green, nir], outputs=[answer, gallery, trace, report])

if __name__ == "__main__":
    demo.launch(inbrowser=True)