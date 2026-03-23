"""
SpectraGen — Neural Style Transfer & Generative Art Laboratory
================================================================
A Streamlit application for neural style transfer (VGG19) and
procedural generative art.  No API keys required.
"""

import io
import time
import streamlit as st
from PIL import Image

from src.style_transfer import StyleTransfer
from src.generative import GenerativeArt, PALETTES
from src.visualizer import color_histogram

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SpectraGen | Generative Art",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom dark CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Global ---- */
:root {
    color-scheme: dark;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a0a 0%, #111118 50%, #0a0a0a 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: #0d0d14;
    border-right: 1px solid #1a1a2e;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* ---- Typography ---- */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
.main-title {
    background: linear-gradient(90deg, #00ff88, #00d4ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
    color: #0a0a0a;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.6rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,212,255,0.35);
}

/* ---- Download button ---- */
.stDownloadButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
    color: #fff;
    font-weight: 700;
    border: none;
    border-radius: 8px;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #161622;
    border-radius: 8px 8px 0 0;
    color: #aaa;
    padding: 10px 28px;
    border: 1px solid #1a1a2e;
}
.stTabs [aria-selected="true"] {
    background: #1a1a2e;
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88;
}

/* ---- Cards ---- */
.art-card {
    background: #12121e;
    border: 1px solid #1f1f35;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    border: 2px dashed #1f1f35;
    border-radius: 12px;
    padding: 1rem;
}

/* ---- Slider ---- */
.stSlider > div > div > div {
    color: #00d4ff;
}

/* ---- Progress bar ---- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ff88, #00d4ff);
}

/* ---- Metric ---- */
[data-testid="stMetric"] {
    background: #12121e;
    border: 1px solid #1f1f35;
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title"> SpectraGen</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Neural Style Transfer &amp; Generative Art Laboratory &mdash; '
    'powered by PyTorch VGG19</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def load_style_transfer():
    return StyleTransfer()


@st.cache_resource(show_spinner=False)
def load_generative():
    return GenerativeArt()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        [" Neural Style Transfer", " Generative Art Lab"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>SpectraGen v1.0 &bull; "
        "No API keys required</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_nst, tab_gen = st.tabs([" Neural Style Transfer", " Generative Art Lab"])

# ============================= TAB 1 — NST ================================
with tab_nst:
    st.markdown("### Upload Your Images")

    col_content, col_style = st.columns(2)

    with col_content:
        st.markdown('<div class="art-card">', unsafe_allow_html=True)
        st.markdown("**Content Image**")
        content_file = st.file_uploader(
            "Upload content image",
            type=["jpg", "jpeg", "png", "webp"],
            key="content_upload",
        )
        if content_file:
            content_img = Image.open(content_file)
            st.image(content_img, caption="Content", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_style:
        st.markdown('<div class="art-card">', unsafe_allow_html=True)
        st.markdown("**Style Image**")
        style_file = st.file_uploader(
            "Upload style image",
            type=["jpg", "jpeg", "png", "webp"],
            key="style_upload",
        )
        if style_file:
            style_img = Image.open(style_file)
            st.image(style_img, caption="Style", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # NST parameters ---------------------------------------------------------
    st.markdown("### Transfer Parameters")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        nst_steps = st.slider("Optimisation steps", 50, 800, 300, 25, key="nst_steps")
    with pcol2:
        style_weight_exp = st.slider(
            "Style weight (10^x)",
            3.0, 9.0, 6.0, 0.5,
            key="style_w",
        )
    with pcol3:
        content_weight = st.slider(
            "Content weight",
            0.1, 10.0, 1.0, 0.1,
            key="content_w",
        )

    style_weight = 10 ** style_weight_exp

    # Run button -------------------------------------------------------------
    run_nst = st.button(" Run Style Transfer", use_container_width=True, key="run_nst")

    if run_nst:
        if not content_file or not style_file:
            st.warning("Please upload both a content and a style image.")
        else:
            st.markdown("---")
            progress_bar = st.progress(0, text="Initialising VGG19…")
            status_text = st.empty()

            engine = load_style_transfer()

            start_time = time.time()

            def _progress(step, total, loss):
                pct = step / total
                progress_bar.progress(
                    pct,
                    text=f"Step {step}/{total}  —  loss: {loss:,.1f}",
                )

            result_img = engine.transfer(
                content_img,
                style_img,
                steps=nst_steps,
                style_weight=style_weight,
                content_weight=content_weight,
                progress_callback=_progress,
            )

            elapsed = time.time() - start_time
            progress_bar.progress(1.0, text="Done!")

            st.markdown("### Results")
            m1, m2 = st.columns(2)
            m1.metric("Total Time", f"{elapsed:.1f}s")
            m2.metric("Steps", str(nst_steps))

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.image(content_img, caption="Content (original)", use_container_width=True)
            with res_col2:
                st.image(style_img, caption="Style reference", use_container_width=True)
            with res_col3:
                st.image(result_img, caption="Stylised result", use_container_width=True)

            st.download_button(
                " Download Stylised Image",
                data=pil_to_bytes(result_img),
                file_name="spectra_stylised.png",
                mime="image/png",
                use_container_width=True,
            )

            # Histogram
            st.markdown("### Colour Analysis")
            fig = color_histogram(result_img)
            st.plotly_chart(fig, use_container_width=True)


# =========================== TAB 2 — Generative ===========================
with tab_gen:
    st.markdown("### Choose Your Art Style")

    gen_type = st.selectbox(
        "Art type",
        [" Fractal Art", " Flow Field", " Wave Interference"],
        key="gen_type",
    )

    # Common controls --------------------------------------------------------
    st.markdown("### Parameters")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        gen_width = st.slider("Width", 400, 1600, 800, 100, key="gen_w")
    with gc2:
        gen_height = st.slider("Height", 400, 1600, 800, 100, key="gen_h")
    with gc3:
        gen_palette = st.selectbox("Palette", list(PALETTES.keys()), key="gen_pal")

    # Show palette preview
    pal_cols = st.columns(len(PALETTES[gen_palette]))
    for i, (col, rgb) in enumerate(zip(pal_cols, PALETTES[gen_palette])):
        hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
        col.markdown(
            f'<div style="background:{hex_c};height:30px;border-radius:6px;'
            f'border:1px solid #333;"></div>'
            f'<small style="color:#888">{hex_c}</small>',
            unsafe_allow_html=True,
        )

    # Art-specific controls --------------------------------------------------
    if "Fractal" in gen_type:
        fc1, fc2 = st.columns(2)
        with fc1:
            fractal_type = st.selectbox("Fractal type", ["mandelbrot", "julia"], key="ft")
            frac_iters = st.slider("Max iterations", 30, 500, 100, 10, key="fi")
        with fc2:
            c_real = st.slider("Julia C (real)", -1.5, 1.5, -0.7, 0.05, key="cr")
            c_imag = st.slider("Julia C (imag)", -1.5, 1.5, 0.27, 0.05, key="ci")

    elif "Flow" in gen_type:
        fc1, fc2 = st.columns(2)
        with fc1:
            noise_scale = st.slider("Noise scale", 0.001, 0.05, 0.005, 0.001, key="ns")
            n_particles = st.slider("Particles", 500, 10000, 3000, 500, key="np")
        with fc2:
            flow_steps = st.slider("Trail length", 20, 200, 80, 10, key="fs")
            flow_seed = st.number_input("Seed (0 = random)", 0, 99999, 0, key="fseed")

    else:  # Wave
        wc1, wc2 = st.columns(2)
        with wc1:
            n_waves = st.slider("Number of waves", 2, 20, 5, 1, key="nw")
        with wc2:
            wave_seed = st.number_input("Seed (0 = random)", 0, 99999, 0, key="wseed")

    # Generate button --------------------------------------------------------
    gen_run = st.button(" Generate Art", use_container_width=True, key="gen_run")

    if gen_run:
        gen_engine = load_generative()

        with st.spinner("Generating artwork…"):
            t0 = time.time()

            if "Fractal" in gen_type:
                art = gen_engine.fractal_art(
                    width=gen_width,
                    height=gen_height,
                    palette=gen_palette,
                    iterations=frac_iters,
                    fractal_type=fractal_type,
                    c_real=c_real,
                    c_imag=c_imag,
                )
            elif "Flow" in gen_type:
                seed_val = flow_seed if flow_seed != 0 else None
                art = gen_engine.flow_field(
                    width=gen_width,
                    height=gen_height,
                    palette=gen_palette,
                    noise_scale=noise_scale,
                    n_particles=n_particles,
                    steps=flow_steps,
                    seed=seed_val,
                )
            else:
                seed_val = wave_seed if wave_seed != 0 else None
                art = gen_engine.wave_interference(
                    width=gen_width,
                    height=gen_height,
                    palette=gen_palette,
                    n_waves=n_waves,
                    seed=seed_val,
                )

            elapsed = time.time() - t0

        st.markdown("---")
        st.markdown("### Generated Artwork")

        m1, m2, m3 = st.columns(3)
        m1.metric("Generation Time", f"{elapsed:.2f}s")
        m2.metric("Resolution", f"{gen_width}×{gen_height}")
        m3.metric("Palette", gen_palette)

        st.image(art, caption="Generated artwork", use_container_width=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                " Download PNG",
                data=pil_to_bytes(art, "PNG"),
                file_name=f"spectra_{gen_type.split()[-1].lower()}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                " Download JPEG",
                data=pil_to_bytes(art, "JPEG"),
                file_name=f"spectra_{gen_type.split()[-1].lower()}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        # Histogram
        st.markdown("### Colour Analysis")
        fig = color_histogram(art)
        st.plotly_chart(fig, use_container_width=True)
