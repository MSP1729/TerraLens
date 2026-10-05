# TerraLens

TerraLens is a natural-language assistant for satellite imagery, built by Team Kairox for Smart India Hackathon 2026 (SIH26227, Ministry of Defence: semantic retrieval and multi-temporal change analysis).

Analysts ask a question in plain language and get an answer, a caption and supporting evidence, without GIS or remote-sensing expertise.

## Features

Built in this prototype:
- Visual question answering and captioning on optical and SAR images (AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct, run locally in 4-bit or on a remote GPU backend)
- Change analysis between two dates: change mask, changed-area percentage and a bright built-up proxy, with caveats
- Archive search: image embeddings stored in a FAISS index, searched by text query, fully local
- Cluster view: K-means over archive embeddings for exploring a large archive without labels

Proposed (not built yet):
- Trained change-detection model (Siamese / ChangeFormer)
- Validated optical + SAR fusion
- Visual grounding with verified boxes

## Important limits
- Change masks are heuristic screening, not verified land-cover transitions.
- The built-up value is a brightness proxy, not building detection.
- Clusters are exploratory, not verified land-use classes.
- Always verify geolocation and acquisition dates independently.

## Run locally (Windows, Python 3.13)

    cd satquery
    .venv\Scripts\activate

Main app (optical/SAR, VQA, change analysis):

    python app_offline.py        # http://127.0.0.1:7860

Local VQA model:

    python vqa_local.py          # http://127.0.0.1:7870

Archive search:

    python prepare_archive.py    # build the index once
    python archive_search.py

Cluster view:

    python cluster_view.py       # http://127.0.0.1:7862

## Models and tools
- AdaptLLM remote-sensing-Qwen2.5-VL-3B-Instruct: question answering and captioning
- RemoteCLIP: retrieval embeddings (https://github.com/ChenDelong1999/RemoteCLIP)
- FAISS: similarity search
- scikit-learn: K-means clustering
- Gradio: user interface
- PyTorch, Hugging Face Transformers

## Team
Team Kairox (AIT_KAIROX), Army Institute of Technology, Pune. Smart India Hackathon 2026.
