"""
Neural Style Transfer Engine
=============================
Uses a pretrained VGG19 model from torchvision to perform artistic style transfer.
Gatys et al. (2016) approach — no API keys required, runs entirely on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTENT_LAYERS = ["conv4_2"]
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

# ImageNet normalisation stats expected by VGG
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Max dimension to keep processing tractable on CPU
MAX_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix for a batch of feature maps."""
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)


def _load_image(image: Image.Image, max_size: int = MAX_SIZE) -> torch.Tensor:
    """Resize, normalise and convert a PIL Image to a tensor."""
    w, h = image.size
    scale = max_size / max(w, h)
    new_size = (int(h * scale), int(w * scale))

    loader = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return loader(image).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """De-normalise a tensor and convert back to PIL Image."""
    img = tensor.clone().squeeze(0)
    # Undo ImageNet normalisation
    for ch, mean, std in zip(img, IMAGENET_MEAN, IMAGENET_STD):
        ch.mul_(std).add_(mean)
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)


# ---------------------------------------------------------------------------
# Feature extractor built on top of VGG19
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """Extract feature maps at specified VGG19 layers."""

    # Mapping from friendly layer names to VGG19 sequential indices
    _LAYER_MAP = {
        "conv1_1": 0, "conv1_2": 2,
        "conv2_1": 5, "conv2_2": 7,
        "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
        "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
        "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34,
    }

    def __init__(self, layers: list[str]):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        self.layer_names = layers

        indices = sorted([self._LAYER_MAP[l] for l in layers])
        prev = 0
        for idx in indices:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            prev = idx + 1

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        for name, sl in zip(self.layer_names, self.slices):
            x = sl(x)
            features[name] = x
        return features


# ---------------------------------------------------------------------------
# Style Transfer class
# ---------------------------------------------------------------------------

class StyleTransfer:
    """Gatys-style neural style transfer using VGG19 on CPU."""

    def __init__(self):
        all_layers = sorted(
            set(CONTENT_LAYERS + STYLE_LAYERS),
            key=lambda l: VGGFeatureExtractor._LAYER_MAP[l],
        )
        self.extractor = VGGFeatureExtractor(all_layers).eval()

    # ------------------------------------------------------------------ #
    def transfer(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        steps: int = 300,
        style_weight: float = 1e6,
        content_weight: float = 1.0,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Image.Image:
        """
        Run style transfer and return the stylised PIL Image.

        Parameters
        ----------
        content_image : PIL.Image
            The photograph / content image.
        style_image : PIL.Image
            The artistic style reference image.
        steps : int
            Number of optimisation iterations.
        style_weight : float
            Weight of the style loss term.
        content_weight : float
            Weight of the content loss term.
        progress_callback : callable, optional
            ``callback(current_step, total_steps, loss)`` — used to drive
            a Streamlit progress bar.

        Returns
        -------
        PIL.Image
            The stylised result.
        """
        content_tensor = _load_image(content_image.convert("RGB"))
        style_tensor = _load_image(style_image.convert("RGB"))

        # Initialise the generated image from the content
        generated = content_tensor.clone().requires_grad_(True)

        optimizer = optim.Adam([generated], lr=0.01)

        # Pre-compute targets
        with torch.no_grad():
            content_features = self.extractor(content_tensor)
            style_features = self.extractor(style_tensor)
            style_grams = {
                layer: _gram_matrix(style_features[layer])
                for layer in STYLE_LAYERS
                if layer in style_features
            }

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            gen_features = self.extractor(generated)

            # Content loss
            c_loss = torch.tensor(0.0)
            for layer in CONTENT_LAYERS:
                if layer in gen_features and layer in content_features:
                    c_loss += nn.functional.mse_loss(
                        gen_features[layer], content_features[layer]
                    )

            # Style loss
            s_loss = torch.tensor(0.0)
            for layer in STYLE_LAYERS:
                if layer in gen_features and layer in style_grams:
                    gen_gram = _gram_matrix(gen_features[layer])
                    s_loss += nn.functional.mse_loss(gen_gram, style_grams[layer])

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()
            optimizer.step()

            if progress_callback is not None:
                progress_callback(step, steps, total_loss.item())

        return _tensor_to_pil(generated.detach())
