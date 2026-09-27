# SatQuery AI

Satellite-image question answering and exploratory multi-image analysis for the Smart India Hackathon ISRO/SAC problem. SatQuery pairs a remote-sensing vision-language model with simple, auditable image-analysis tools. The app can call an online GPU endpoint, or run with a local model server when internet access is unavailable.

> Prototype, not a validated geospatial measurement system. Check every result against the imagery, band order, acquisition conditions and ground truth before using it for decisions.

## What it does

- **Ask or describe:** upload an image and ask a question, or request a scene caption from [AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct](https://huggingface.co/AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct).
- **Before/after change:** compare two same-grid images, show a pixel-difference overlay and the percentage of pixels flagged. A bright-surface proxy reports before/after percentages and the change in percentage points; it does **not** measure confirmed urban growth.
- **Optical + SAR fusion:** combine aligned optical and SAR images, show a blue water proxy and red bright-surface/possible-built-up proxy. For a TIFF with suitable bands, NDWI helps the water proxy and NDVI filters likely vegetation out of the red mask. Bare soil and other bright surfaces can still appear red.
- **Visible reasoning and exports:** a rule-based controller chooses checks, VLM calls and image tools based on the task and uploaded modalities. The UI shows an execution trace. Download a ZIP containing a readable, self-contained `report.html` (with images, metrics and steps), `report.json`, and evidence PNGs.

Supported inputs: GeoTIFF/TIFF, PNG and JPEG. Paired images must have the same pixel dimensions; georeferenced images must also match CRS and pixel transform. Same-size PNG/JPEG files cannot prove geographic alignment. TIFF band numbers are 1-based; **the default Green=3, Red=4, NIR=8 is for the example Sentinel-2 band order, not every TIFF.** Check band metadata before analysis.

## Requirements

- Python in a virtual environment on Windows; NVIDIA CUDA-capable GPU for **offline model inference**. The offline 4-bit configuration has run on a 4GB laptop GPU, but memory and compatibility can vary. A first run downloads model weights; after those files and Python packages are cached, the model can run without internet.
- For online mode, a running Kaggle/Colab GPU notebook serving the same model through Gradio with an `/ask` API, plus a live `https://...gradio.live` link. Share links are temporary and public: **do not send sensitive imagery to that endpoint.**
- Python packages used by the app: `gradio`, `gradio_client`, `numpy`, `pillow`, `rasterio`. Offline server also needs a CUDA build of `torch`, `transformers`, `accelerate`, `qwen-vl-utils`, `bitsandbytes`.

From PowerShell, with your terminal in `C:\Users\ASUS\Desktop\FPL\satquery`:

```powershell
cd C:\Users\ASUS\Desktop\FPL\satquery
.\.venv\Scripts\Activate.ps1
python -m pip install -U gradio gradio_client numpy pillow rasterio transformers accelerate qwen-vl-utils bitsandbytes
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Install a CUDA-compatible PyTorch build separately if `torch` is missing or prints `False`; follow the command for your machine at <https://pytorch.org/get-started/locally/>. Offline mode will not work until CUDA is available to PyTorch. Online mode does not require a GPU on the laptop.

## Run online

1. Start the GPU notebook on Kaggle or Colab with the remote-sensing model and a Gradio `ask(image, question)` interface. The notebook/server code is not currently included in this repository; the endpoint must expose `/ask` and remain running. Kaggle sessions may queue, stop or expire.
2. In `app.py`, set `COLAB_URL` near the top to the current `https://...gradio.live` link. The old link in that file is not permanent.
3. In the activated project terminal, run:

```powershell
python app.py
```

Open the local URL printed by Gradio (the app also attempts to open a browser tab). Your uploads are sent to the chosen online endpoint for model inference; local numerical analysis runs on your laptop.

## Run offline

Start the local VLM server first, then the app in a **second** PowerShell terminal. Keep both running:

```powershell
# Terminal 1, with the virtual environment active
python vqa_local.py
```

Wait for the server to load the 4-bit model and listen at `http://127.0.0.1:7870`. In another activated terminal:

```powershell
# Terminal 2
python app_offline.py
```

`app_offline.py` calls the server on your own laptop (`127.0.0.1:7870`). For a real offline demo, download the model and dependencies in advance and verify an answer with Wi-Fi disconnected. If an API error appears, check that Terminal 1 remains open and the model loaded successfully. First loading may take time and use substantial GPU memory.

## A demo flow

1. Choose **Scene caption** or **VQA**, upload an optical image, ask what is visible and inspect the answer. Model text can be wrong; compare it with the actual scene.
2. Choose **Change detection** for two aligned dates. Show the overlay, flagged-pixel percentage and before/after bright-surface proxy; do not call that percentage confirmed building growth.
3. Choose **Optical + SAR fusion** for paired images. Check the modalities and optical band order. Show blue water and red bright possible-built-up areas. The vegetation filter needs valid red and NIR TIFF bands.
4. Click **Download report and images (ZIP)**. Extract it and open `report.html` in a browser for the readable analysis; `report.json` holds the machine-readable trace.

## Screenshots

Add real screenshots from your tested app to an `assets/` folder, then uncomment these image lines. Do not use mock results as evidence.

<!-- ![Task selection and image upload](assets/input-ui.png) -->
<!-- ![Fusion overlay and numeric result](assets/fusion-result.png) -->
<!-- ![Readable HTML report](assets/report-html.png) -->

Suggested screenshots: (1) task selection and both uploads, (2) result percentages with the blue/red overlay visible, (3) extracted `report.html` showing the evidence image and analysis steps. Crop out private paths or account details before committing.

## Limits and reproducibility

The masks are heuristics, not trained building/water classifiers; bright farm soil may look built-up and low SAR return is not always water. Season, clouds, SAR preprocessing, sensors and image registration affect comparisons. The VLM can hallucinate and its answers have no calibrated confidence score. A `gradio.live` link will expire; update `COLAB_URL` when restarting online mode. Keep datasets and model files out of Git unless their licenses and sizes permit redistribution. Cite the [model card](https://huggingface.co/AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct) when presenting results.
