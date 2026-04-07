# SpectraGen -- Neural Style Transfer and Generative Art Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)](LICENSE)

A self-contained lab for neural style transfer and procedural generative art, powered by PyTorch's pretrained VGG19. No API keys, no GPU, no cloud dependency.

---

## Features

### Neural Style Transfer
- Gatys et al. (2016) algorithm using pretrained VGG19
- Content layer: `conv4_2` / Style layers: `conv1_1` through `conv5_1`
- Gram-matrix style loss with configurable weights
- Real-time progress bar during optimisation
- Side-by-side before/after comparison
- Runs entirely on CPU -- no CUDA required

### Generative Art Lab

Three procedural art engines, zero neural networks:

| Mode | Description |
|------|-------------|
| Fractal Art | Mandelbrot and Julia set fractals with smooth colouring |
| Flow Field | Perlin-noise-inspired particle flow fields |
| Wave Interference | Circular wave superposition patterns |

### Visualization
- RGB colour histogram (Plotly, dark theme)
- VGG19 feature-map grid viewer
- Interactive charts with hover tooltips

---

## Quick Start

```bash
git clone https://github.com/Zinga18018/Neural-Style-Transfer-and-Generative-Art-Lab.git
cd Neural-Style-Transfer-and-Generative-Art-Lab
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## How It Works

### Neural Style Transfer

1. Load images -- content and style images are resized and normalised to ImageNet statistics.
2. Extract features -- a frozen VGG19 extracts activations at content and style layers.
3. Gram matrices -- style is captured as the correlation between feature channels.
4. Optimise -- the generated image is iteratively updated via Adam to minimise total loss (content + style).
5. Output -- the optimised tensor is de-normalised and returned as a PIL Image.

### Generative Art

- **Fractals**: iterative escape-time algorithm with smooth colouring via logarithmic interpolation.
- **Flow Fields**: value-noise angle field drives particles; additive blending creates luminous trails.
- **Wave Interference**: superposition of circular waves from random sources produces moire-like patterns.

---

## Project Structure

```
Neural-Style-Transfer-and-Generative-Art-Lab/
|-- app.py                  # Streamlit entry point
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- style_transfer.py   # VGG19 neural style transfer engine
|   |-- generative.py       # Fractal, flow-field, wave art
|   +-- visualizer.py       # Plotly chart utilities
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web UI |
| Style Transfer | PyTorch / VGG19 | Feature extraction and optimisation |
| Image Processing | Pillow | Image I/O and manipulation |
| Numerical | NumPy | Fractal and noise computation |
| Charts | Plotly | Dark-themed visualisations |

---

## License

MIT License -- Yogesh Kuchimanchi
