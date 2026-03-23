# 🎨 SpectraGen — Neural Style Transfer & Generative Art Laboratory

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pillow](https://img.shields.io/badge/Pillow-10.0%2B-3776AB?logo=python&logoColor=white)](https://python-pillow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88.svg)](LICENSE)

> **No API keys. No GPU. No cloud dependency.**  
> A self-contained laboratory for neural style transfer and procedural generative art, powered by PyTorch's pretrained VGG19.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SpectraGen App                           │
│                       (Streamlit UI)                            │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Neural Style      │   Generative Art  │    Visualiser         │
│   Transfer Tab      │   Lab Tab         │    (Plotly)           │
├─────────────────────┼───────────────────┼───────────────────────┤
│                     │                   │                       │
│  src/               │  src/             │  src/                 │
│  style_transfer.py  │  generative.py    │  visualizer.py        │
│                     │                   │                       │
│  ┌───────────────┐  │  ┌─────────────┐  │  ┌─────────────────┐  │
│  │ VGG19         │  │  │ Mandelbrot  │  │  │ RGB Histogram   │  │
│  │ (pretrained)  │  │  │ Julia Set   │  │  │ Feature Maps    │  │
│  │               │  │  │ Flow Field  │  │  │ (Dark theme)    │  │
│  │ Gram Matrix   │  │  │ Wave Interf.│  │  │                 │  │
│  │ Style Loss    │  │  │             │  │  │                 │  │
│  │ Content Loss  │  │  │ 4 Palettes  │  │  │                 │  │
│  └───────────────┘  │  └─────────────┘  │  └─────────────────┘  │
├─────────────────────┴───────────────────┴───────────────────────┤
│                    PyTorch · NumPy · Pillow                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🖌️ Neural Style Transfer
- **Gatys et al. (2016)** algorithm using pretrained **VGG19**
- Content layer: `conv4_2` — Style layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`
- Gram-matrix style loss with configurable weights
- Real-time progress bar during optimisation
- Side-by-side before / after comparison
- Runs entirely on **CPU** — no CUDA required

### 🎲 Generative Art Lab
Three procedural art engines, zero neural networks:

| Mode | Description |
|------|-------------|
| **🌀 Fractal Art** | Mandelbrot & Julia set fractals with smooth colouring |
| **🌊 Flow Field** | Perlin-noise-inspired particle flow fields |
| **🔊 Wave Interference** | Circular wave superposition patterns |

### 🎨 Four Colour Palettes

| Palette | Colours |
|---------|---------|
| **Cyber** | `#00ff88` `#00d4ff` `#0a0a0a` `#00b4d8` `#00ffc8` |
| **Sunset** | Warm reds, oranges, golds, rose, plum |
| **Ocean** | Deep blue, cerulean, ice tones |
| **Neon** | Hot pink, violet, green, cyan, yellow |

### 📊 Visualisation
- RGB colour histogram (Plotly, dark theme)
- VGG19 feature-map grid viewer
- Interactive charts with hover tooltips

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Neural-Style-Transfer-and-Generative-Art-Lab.git
cd Neural-Style-Transfer-and-Generative-Art-Lab
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit 1.31+ | Interactive web UI |
| **Style Transfer** | PyTorch 2.1+ / VGG19 | Feature extraction & optimisation |
| **Image Processing** | Pillow 10.0+ | Image I/O and manipulation |
| **Numerical** | NumPy 1.26+ | Fractal & noise computation |
| **Charts** | Plotly 5.18+ | Interactive dark-themed visualisations |
| **Pretrained Model** | `torchvision.models.vgg19` | ImageNet weights (auto-downloaded) |

---

## Project Structure

```
Neural-Style-Transfer-and-Generative-Art-Lab/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── style_transfer.py   # VGG19 neural style transfer engine
│   ├── generative.py       # Fractal, flow-field, wave art
│   └── visualizer.py       # Plotly chart utilities
├── data/                   # Sample input images (optional)
├── assets/                 # Static assets
└── gallery/                # Saved generated artwork
```

---

## How It Works

### Neural Style Transfer

1. **Load images** — content and style images are resized and normalised to ImageNet statistics.
2. **Extract features** — a frozen VGG19 extracts activations at content (`conv4_2`) and style layers.
3. **Gram matrices** — style is captured as the correlation between feature channels.
4. **Optimise** — the generated image is iteratively updated via Adam to minimise:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{content}} + \beta \cdot \mathcal{L}_{\text{style}}$$

5. **Output** — the optimised tensor is de-normalised and returned as a PIL Image.

### Generative Art

- **Fractals**: iterative escape-time algorithm with smooth colouring via logarithmic interpolation.
- **Flow Fields**: value-noise angle field drives thousands of particles; additive blending creates luminous trails.
- **Wave Interference**: superposition of circular waves from random sources produces moiré-like patterns.

---

## License

MIT License — **Yogesh Kuchimanchi**

```
MIT License

Copyright (c) 2026 Yogesh Kuchimanchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
