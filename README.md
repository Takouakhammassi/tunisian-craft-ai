# 🏺 Hirfatuna : Tunisian Craft Heritage AI Platform

> An AI-powered platform that identifies Tunisian handicrafts from a photo and reveals their history, region of origin, and craftsmanship preserving a cultural heritage that has no public digital footprint.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://hirfatuna.streamlit.app)

---

## The problem

Tunisian handicrafts represent centuries of cultural heritage. Yet this knowledge is scattered, undocumented online, and largely invisible to younger generations. No public dataset or digital tool existed to classify, explain, or explore it.

**Hirfatuna** solves this with a full AI pipeline: from a self-collected image dataset to a deployed, interactive platform.

## Features

### 1. VLM guard
A vision-language model checks the photo is actually a craft before classifying it

### 2. Visual classification
AI identifies the craft across 10 categories (ResNet-50, 90.8% accuracy)

### 3. Geographic origin
Pinpoints the craft's region on an interactive map of Tunisia

### 4. Cultural knowledge base
Reveals history, materials, techniques, and fun facts for each craft

### 5. Explainable AI
Grad-CAM visualizes exactly which parts of the image drove the prediction

### 6. RAG-powered chatbot
Ask free-form questions, answered from a curated knowledge base, no hallucinations

### 7. Session history
Browse every craft analyzed during your session

---

## Results

- **10 craft categories**, classified with **90.8% validation accuracy**
- **3,264 labeled images**, self-collected 
- **ResNet-50** selected after a comparative study across **5 architectures × 4 fine-tuning depths** (20 configurations) — see [`comparative_study/`](comparative_study/)
- Fully deployed, end-to-end application, no setup required to try it

```
                         precision   recall   f1-score
        bijoux_berberes      0.92     0.95      0.93
           bois_sculpte      0.94     0.88      0.91
    broderie_tunisienne      0.83     0.86      0.85
                 cuivre      0.93     0.90      0.91
                 djebba      0.85     0.92      0.88
              fer_forge      0.91     0.89      0.90
maroquinerie_tunisienne      0.92     0.90      0.91
         poterie_nabeul      0.96     0.92      0.94
         tapis_kairouan      0.95     0.97      0.96
          verre_souffle      0.86     0.91      0.88

               accuracy                          0.908
```

---

## How it works

### 1. Dataset — built from scratch

No public dataset exists for Tunisian handicrafts, so the first step was building one. `src/collect_dataset.py` queries multiple search phrases per category, covering different visual angles (macro close-up, object in use, market stall, isolated product shot) to avoid a dataset that is visually too uniform.

```bash
python src/collect_dataset.py    
python src/prepare_dataset.py   
```

### 2. Architecture selection — a comparative study, not a guess

Rather than picking a CNN architecture by convention, five candidates (ResNet-50, AlexNet, MobileNetV2, GoogLeNet, EfficientNet-B0) were each trained at four fine-tuning depths (10%, 30%, 50%, 100% of layers unfrozen), 20 configurations in total. Full methodology, results, and analysis live in [`comparative_study/`](comparative_study/).

**Key finding**: ResNet-50 at 30% unfrozen layers won (90.8%), narrowly ahead of ResNet-50 at 10% (90.2%). Fine-tuning beyond 50% consistently hurt accuracy across most architectures, a clear sign of overfitting on a dataset this size. AlexNet collapsed to near-random accuracy at both extremes (10% and 100%), likely due to its lack of batch normalization, making it unstable outside a narrow fine-tuning range.

```bash
python src/train_model.py        
```

### 3. VLM guard — catching the wrong kind of photo before it's classified

The classifier is only trained to choose among 10 known craft categories, it has no way to say "this isn't a craft at all." Before an uploaded photo reaches the classifier, it passes through a **vision-language model guard** (Qwen's VLM, accessed via Hugging Face Inference Providers) that judges whether the image actually shows a handcrafted object.

```
Uploaded photo → VLM guard → not a craft → "I can't identify this image"
                      ↓
                 looks like a craft
                      ↓
              ResNet-50 classification
```

### 4. Explainability — Grad-CAM

Every prediction is paired with a **Grad-CAM heatmap**, showing which regions of the image most influenced the model's decision, turning the classifier from a black box into something a user can actually inspect and trust.

### 5. Knowledge base & RAG chatbot

`craft_knowledge.json` is a hand-curated knowledge base covering each craft's history, materials, techniques, and cultural significance. The chatbot is a **structured Retrieval-Augmented Generation (RAG)** pipeline:

1. **Intent detection** : classifies the question (history / materials / techniques / time / region…)
2. **Keyword matching** : finds the relevant craft via a keyword index built from the knowledge base
3. **Semantic fallback** : if no keyword matches, falls back to FAISS + Sentence-Transformers embedding similarity
4. **Answer composition** : the response is composed directly from the knowledge base, never generated freely

This design choice trades the flexibility of a general-purpose LLM for **factual reliability**, the chatbot can never invent information about a craft it doesn't have data on.

### 6. Interface

A multi-page **Streamlit** application: Scan (upload & identify), Explore (full craft catalog), Map (interactive geographic view via Folium), AI Chat, and History.

---

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Deep learning | PyTorch, torchvision | Model training and inference |
| Architecture | ResNet-50 (30% fine-tuned) | Image classification, selected via comparative study |
| Vision-language guard | Qwen3.5-VL (via Hugging Face Inference Providers) | Filters non-craft images before classification |
| Explainability | Grad-CAM | Visual model interpretability |
| Embeddings | Sentence-Transformers | Semantic text representation for RAG |
| Vector search | FAISS | Fast similarity search for chatbot fallback |
| Data | Pandas, Pillow | Dataset manifest, image processing |
| Frontend | Streamlit | Web application interface |
| Maps | Folium | Interactive geographic visualization |

---

## Project structure

```
tunisian-craft-ai/
├── app/
│   ├── app.py                     
│   ├── vlm_guard.py                
│   └── craft_knowledge.json        
├── src/
│   ├── collect_dataset.py         
│   ├── prepare_dataset.py          
│   └── train_model.py             
├── comparative_study/
│   ├── compare_architectures.py   
│   ├── comparison_results.csv
│   └── comparison_summary.png
├── models/
│   ├── best_model.pth
│   └── metadata.json
└── requirements.txt
```

---

## Running locally
### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/tunisian-craft-ai.git
```

### 2. Create a Python virtual 
```bash
cd tunisian-craft-ai
python -m venv venv
```

### 3. Activate the virtual environment (Windows)
```bash
venv\Scripts\activate         
```

### 4. Install the required packages
```bash
pip install -r requirements.txt
```
### 5. Set your Hugging Face token (optional)
```bash
$env:HF_TOKEN="hf_your_token_here"
```
### 6. Run app
```bash
streamlit run app.py
```
Dataset collection, preprocessing, and model training scripts are available in the src/ directory if you want to rebuild the dataset and retrain the model from scratch.

When deploying on Streamlit Cloud, set `HF_TOKEN` under **Settings → Secrets** instead of as a local environment variable.


<p align="center"><em>حرفتنا — Hirfatuna</em><br>Preserving Tunisian craft heritage through AI</p>
