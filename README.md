# SpectraGen -- Neural Style Transfer and Generative Art Lab

A Streamlit lab with two separate kinds of image generation:

- **Neural style transfer:** a Gatys et al. (2016)-style procedure uses frozen pretrained VGG19 features and optimizes an output image's pixels. It does not train a new neural model.
- **Procedural art:** NumPy/Pillow algorithms generate Mandelbrot or Julia fractals, value-noise flow fields and circular-wave interference patterns. These generators use no neural network.

## Run locally

Use Python 3.10+:

```bash
git clone https://github.com/Zinga18018/Neural-Style-Transfer-and-Generative-Art-Lab.git
cd Neural-Style-Transfer-and-Generative-Art-Lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead. Open the local URL printed by Streamlit, normally `http://localhost:8501`.

No API key or CUDA GPU is required. Style transfer runs on CPU and loads torchvision's `VGG19_Weights.IMAGENET1K_V1` weights when first used, downloading them if uncached. It is therefore not fully offline on first use. The procedural functions need no model weights, although the combined app imports the Torch-based style-transfer module.

## Neural style transfer

[src/style_transfer.py](src/style_transfer.py) resizes and normalizes content/style images, extracts VGG19 activations and updates the generated image with Adam. Content uses `conv4_2`; style compares Gram matrices at `conv1_1` through `conv5_1`.

The UI exposes loss weights and iteration settings, progress updates, before/after images and an RGB histogram. CPU cost depends on image size and iteration count; no timing or quality benchmark is committed. Use images you have permission to process and share.

## Procedural art

[src/generative.py](src/generative.py) implements:

| Mode | Method |
|---|---|
| Fractals | Escape-time iterations with smooth coloring |
| Flow fields | Particles following a value-noise angle field |
| Wave interference | Superposition of circular waves |

For repeatable flow/wave output, choose an explicit positive seed and retain the other parameters. In the UI, seed 0 means random rather than a fixed seed. These are algorithmic images, not learned-model samples.

## Current interface limits

[src/visualizer.py](src/visualizer.py) contains a feature-map-grid utility, but [app.py](app.py) only imports the color-histogram function; the feature-map viewer is not exposed in the interface. The sidebar mode selector also does not switch which tabs are displayed.

Dependencies use minimum versions without a lockfile, and no automated test suite is committed. Setup, model execution and browser flows were not rerun for this documentation update.

## License

MIT License -- Yogesh Kuchimanchi

The attribution and license statement above are retained from the existing README. The repository currently has no `LICENSE` file; no license text or new grant is supplied by this documentation.

