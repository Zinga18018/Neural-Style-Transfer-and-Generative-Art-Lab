"""
Procedural Generative Art Engine
=================================
Lightweight, pure-NumPy generative art — no GANs, no GPU required.
Produces fractal, flow-field, and wave-interference artwork.
"""

import math
import numpy as np
from PIL import Image
from typing import Optional

# ---------------------------------------------------------------------------
# Colour Palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "Cyber": [
        (0, 255, 136),   # #00ff88
        (0, 212, 255),   # #00d4ff
        (10, 10, 10),    # #0a0a0a
        (0, 180, 216),   # #00b4d8
        (0, 255, 200),   # #00ffc8
    ],
    "Sunset": [
        (255, 94, 77),   # warm red
        (255, 154, 0),   # orange
        (255, 206, 86),  # gold
        (200, 60, 100),  # rose
        (120, 40, 80),   # plum
    ],
    "Ocean": [
        (0, 63, 136),    # deep blue
        (0, 119, 182),   # ocean blue
        (0, 180, 216),   # cerulean
        (144, 224, 239), # light blue
        (202, 240, 248), # ice
    ],
    "Neon": [
        (255, 0, 110),   # hot pink
        (131, 56, 236),  # violet
        (0, 255, 136),   # green
        (0, 212, 255),   # cyan
        (255, 234, 0),   # yellow
    ],
}


def _palette_array(name: str) -> np.ndarray:
    """Return palette as (N, 3) uint8 array."""
    return np.array(PALETTES.get(name, PALETTES["Cyber"]), dtype=np.float64)


def _colorize(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map normalised float values [0, 1] → RGB via linear palette interpolation."""
    n = len(palette)
    scaled = np.clip(values, 0, 1) * (n - 1)
    idx = np.floor(scaled).astype(int)
    frac = scaled - idx
    idx_next = np.minimum(idx + 1, n - 1)

    r = palette[idx, 0] * (1 - frac) + palette[idx_next, 0] * frac
    g = palette[idx, 1] * (1 - frac) + palette[idx_next, 1] * frac
    b = palette[idx, 2] * (1 - frac) + palette[idx_next, 2] * frac

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Perlin-like noise (value noise with smoothstep interpolation)
# ---------------------------------------------------------------------------

def _value_noise(width: int, height: int, scale: float = 0.02,
                 seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth 2-D noise field in [0, 1]."""
    rng = np.random.default_rng(seed)
    xs = np.arange(width) * scale
    ys = np.arange(height) * scale
    gx, gy = np.meshgrid(xs, ys)

    ix = np.floor(gx).astype(int)
    iy = np.floor(gy).astype(int)
    fx = gx - ix
    fy = gy - iy

    # Smoothstep
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)

    max_i = int(np.max(ix)) + 2
    max_j = int(np.max(iy)) + 2
    grid = rng.random((max_j + 1, max_i + 1))

    n00 = grid[iy, ix]
    n10 = grid[iy, ix + 1]
    n01 = grid[iy + 1, ix]
    n11 = grid[iy + 1, ix + 1]

    nx0 = n00 * (1 - sx) + n10 * sx
    nx1 = n01 * (1 - sx) + n11 * sx
    noise = nx0 * (1 - sy) + nx1 * sy
    return noise


# ---------------------------------------------------------------------------
# Generative Art class
# ---------------------------------------------------------------------------

class GenerativeArt:
    """Produce procedural generative artwork — fractal, flow-field, and wave."""

    # ------------------------------------------------------------------ #
    # Fractal Art (Mandelbrot / Julia)
    # ------------------------------------------------------------------ #
    def fractal_art(
        self,
        width: int = 800,
        height: int = 800,
        palette: str = "Cyber",
        iterations: int = 100,
        fractal_type: str = "mandelbrot",
        c_real: float = -0.7,
        c_imag: float = 0.27015,
    ) -> Image.Image:
        """
        Generate a Mandelbrot or Julia set fractal with a custom colour palette.

        Parameters
        ----------
        width, height : int
            Output image dimensions.
        palette : str
            Name of the colour palette.
        iterations : int
            Maximum escape iterations (controls detail).
        fractal_type : str
            ``"mandelbrot"`` or ``"julia"``.
        c_real, c_imag : float
            Julia-set constant (ignored for Mandelbrot).
        """
        pal = _palette_array(palette)

        if fractal_type == "julia":
            x = np.linspace(-1.5, 1.5, width)
            y = np.linspace(-1.5, 1.5, height)
        else:
            x = np.linspace(-2.5, 1.0, width)
            y = np.linspace(-1.25, 1.25, height)

        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        if fractal_type == "julia":
            C = complex(c_real, c_imag)
        else:
            C = Z.copy()
            Z = np.zeros_like(Z)

        escape = np.full(Z.shape, iterations, dtype=float)
        mask = np.ones(Z.shape, dtype=bool)

        for i in range(iterations):
            Z[mask] = Z[mask] ** 2 + (C if fractal_type == "julia" else C[mask])
            diverged = mask & (np.abs(Z) > 2)
            # Smooth colouring
            escape[diverged] = i + 1 - np.log2(np.log2(np.abs(Z[diverged]) + 1e-10))
            mask[diverged] = False

        norm = np.clip(escape / iterations, 0, 1)
        rgb = _colorize(norm, pal)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------ #
    # Flow Field
    # ------------------------------------------------------------------ #
    def flow_field(
        self,
        width: int = 800,
        height: int = 800,
        palette: str = "Cyber",
        noise_scale: float = 0.005,
        n_particles: int = 3000,
        steps: int = 80,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a flow-field art piece driven by value noise.

        Parameters
        ----------
        width, height : int
            Output image dimensions.
        palette : str
            Colour palette name.
        noise_scale : float
            Controls frequency of the underlying noise.
        n_particles : int
            Number of particles to trace.
        steps : int
            How many steps each particle takes.
        seed : int, optional
            Reproducibility seed.
        """
        pal = _palette_array(palette)
        rng = np.random.default_rng(seed)

        angle_field = _value_noise(width, height, scale=noise_scale, seed=seed) * 2 * np.pi

        canvas = np.zeros((height, width, 3), dtype=np.float64)

        px = rng.uniform(0, width, n_particles)
        py = rng.uniform(0, height, n_particles)
        colors_idx = np.linspace(0, 1, n_particles)

        for _ in range(steps):
            ix = np.clip(px.astype(int), 0, width - 1)
            iy = np.clip(py.astype(int), 0, height - 1)
            angles = angle_field[iy, ix]

            px += np.cos(angles) * 1.5
            py += np.sin(angles) * 1.5

            # Draw live particles
            valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            ix2 = np.clip(px[valid].astype(int), 0, width - 1)
            iy2 = np.clip(py[valid].astype(int), 0, height - 1)

            n = len(pal)
            scaled = np.clip(colors_idx[valid], 0, 1) * (n - 1)
            ci = np.floor(scaled).astype(int)
            ci_next = np.minimum(ci + 1, n - 1)
            frac = scaled - ci
            r = pal[ci, 0] * (1 - frac) + pal[ci_next, 0] * frac
            g = pal[ci, 1] * (1 - frac) + pal[ci_next, 1] * frac
            b = pal[ci, 2] * (1 - frac) + pal[ci_next, 2] * frac

            canvas[iy2, ix2, 0] += r * 0.15
            canvas[iy2, ix2, 1] += g * 0.15
            canvas[iy2, ix2, 2] += b * 0.15

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        return Image.fromarray(canvas)

    # ------------------------------------------------------------------ #
    # Wave Interference
    # ------------------------------------------------------------------ #
    def wave_interference(
        self,
        width: int = 800,
        height: int = 800,
        palette: str = "Cyber",
        n_waves: int = 5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Create an interference pattern from multiple circular / planar waves.

        Parameters
        ----------
        width, height : int
            Output image dimensions.
        palette : str
            Colour palette name.
        n_waves : int
            Number of overlapping wave sources.
        seed : int, optional
            Reproducibility seed.
        """
        pal = _palette_array(palette)
        rng = np.random.default_rng(seed)

        xs = np.linspace(0, 1, width)
        ys = np.linspace(0, 1, height)
        X, Y = np.meshgrid(xs, ys)

        field = np.zeros((height, width), dtype=np.float64)

        for _ in range(n_waves):
            cx, cy = rng.uniform(0, 1, 2)
            freq = rng.uniform(20, 60)
            phase = rng.uniform(0, 2 * math.pi)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            field += np.sin(dist * freq + phase)

        # Normalise to [0, 1]
        fmin, fmax = field.min(), field.max()
        if fmax - fmin > 0:
            norm = (field - fmin) / (fmax - fmin)
        else:
            norm = np.zeros_like(field)

        rgb = _colorize(norm, pal)
        return Image.fromarray(rgb)
