"""Lite Streamlit interface for the Fashion VAE and DCGAN."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.models import Discriminator, Generator, VAE


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
DATA_ROOT = ROOT / "data"
VAE_CHECKPOINT = OUTPUTS / "vae" / "vae.pt"
GAN_CHECKPOINT = OUTPUTS / "gan" / "dcgan.pt"

st.set_page_config(page_title="Fashion Forge", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17211b; --muted:#6d786f; --paper:#f5f7f1; --panel:#ffffff; --accent:#dd5b37; --line:#dfe5dc; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1320px; padding: 2rem 3rem 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.04em; color: var(--ink); }
    h1 { font-size: clamp(2.5rem, 5vw, 5.2rem); line-height: .95; margin-bottom: .8rem; }
    h2 { font-size: 2rem; margin-top: .4rem; }
    p, label, .stMarkdown { color: var(--ink); }
    .eyebrow { color: var(--accent); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
    .lede { max-width: 650px; color: var(--muted); font-size: 1rem; line-height: 1.7; }
    .hero { border-bottom: 1px solid var(--line); padding: 1rem 0 2.4rem; margin-bottom: 1.6rem; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); padding: 1rem 1.1rem; min-height: 105px; }
    .metric-label { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 800; margin-top: .4rem; }
    .section-note { color: var(--muted); font-size: .9rem; line-height: 1.6; }
    [data-testid='stSidebar'] { background: #e9eee5; border-right: 1px solid var(--line); }
    [data-testid='stFileUploader'] { background: var(--panel); border: 1px dashed #aebbae; }
    .stButton > button { border-radius: 2px; border: 1px solid var(--ink); background: var(--ink); color: white; font-weight: 700; }
    .stButton > button:hover { border-color: var(--accent); background: var(--accent); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_vae() -> tuple[VAE | None, str]:
    if not VAE_CHECKPOINT.exists():
        return None, "Checkpoint not found"
    checkpoint = torch.load(VAE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = VAE(int(checkpoint.get("latent_dim", 20)))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, "Ready"


@st.cache_resource(show_spinner=False)
def load_gan() -> tuple[Generator | None, str]:
    if not GAN_CHECKPOINT.exists():
        return None, "Checkpoint not found"
    checkpoint = torch.load(GAN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = Generator(int(checkpoint.get("noise_dim", 100)))
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, "Ready"


def upload_to_tensor(uploaded_file) -> torch.Tensor:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def display_tensor(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().squeeze().clamp(0, 1).numpy()


def display_grid(images: torch.Tensor, columns: int = 8) -> np.ndarray:
    """Tile a batch into one grayscale canvas that Streamlit can render."""
    images = images.detach().cpu().squeeze(1).clamp(0, 1).numpy()
    columns = min(columns, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros((rows * 28, columns * 28), dtype=np.float32)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28] = image
    return canvas


def metric_card(label: str, value: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def read_metrics(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return {}
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def vae_page(device: torch.device) -> None:
    model, status = load_vae()
    st.markdown('<div class="eyebrow">01 / representation lab</div>', unsafe_allow_html=True)
    st.header("Explore the latent wardrobe")
    st.markdown('<p class="section-note">Upload clothing images to see how the VAE reconstructs them, then move through the learned latent space between two looks.</p>', unsafe_allow_html=True)
    if model is None:
        st.error(f"VAE unavailable: {status}. Train it with `python train.py --model vae`.")
        return
    model = model.to(device)
    first, second = st.columns(2)
    with first:
        image_a = st.file_uploader("Source look A", type=["png", "jpg", "jpeg"], key="vae_a")
    with second:
        image_b = st.file_uploader("Source look B (optional)", type=["png", "jpg", "jpeg"], key="vae_b")

    if image_a is None:
        st.info("Add at least one image to start the reconstruction view.")
        return
    tensor_a = upload_to_tensor(image_a).to(device)
    tensor_b = upload_to_tensor(image_b).to(device) if image_b else tensor_a
    with torch.inference_mode():
        reconstruction_a, _, _ = model(tensor_a)
        mu_a, _ = model.encode(tensor_a)
        mu_b, _ = model.encode(tensor_b)
        interpolation = torch.cat([model.decode(torch.lerp(mu_a, mu_b, step / 7)) for step in range(8)])
    original_col, reconstruction_col = st.columns(2)
    with original_col:
        st.caption("INPUT / fitted to 28 × 28")
        st.image(display_tensor(tensor_a), clamp=True, use_container_width=True)
    with reconstruction_col:
        st.caption("VAE RECONSTRUCTION")
        st.image(display_tensor(reconstruction_a), clamp=True, use_container_width=True)
    st.divider()
    st.caption("LATENT WALK / eight decoded points between A and B")
    st.image(display_grid(interpolation, columns=8), clamp=True, width=620)


def gan_page(device: torch.device) -> None:
    model, status = load_gan()
    st.markdown('<div class="eyebrow">02 / synthetic studio</div>', unsafe_allow_html=True)
    st.header("Generate a new collection")
    st.markdown('<p class="section-note">Sample the trained DCGAN with fresh noise. Every click creates a new batch of synthetic clothing silhouettes.</p>', unsafe_allow_html=True)
    if model is None:
        st.error(f"DCGAN unavailable: {status}. Train it with `python train.py --model gan`.")
        return
    model = model.to(device)
    controls, output = st.columns([1, 2.2])
    with controls:
        count = st.slider("Number of looks", 4, 32, 16, step=4)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        generate = st.button("Generate collection", type="primary", use_container_width=True)
        st.markdown("**Model**  DCGAN\n\n**Output**  28 × 28 grayscale\n\n**Latent**  100 dimensions", unsafe_allow_html=True)
    if generate or "gan_samples" not in st.session_state:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(count, model.latent_dim, 1, 1, generator=generator).to(device)
        with torch.inference_mode():
            st.session_state.gan_samples = model(noise).cpu()
    with output:
        samples = (st.session_state.gan_samples + 1) / 2
        st.image(display_grid(samples, columns=8), clamp=True, width=700)


def metrics_page() -> None:
    st.markdown('<div class="eyebrow">03 / training signal</div>', unsafe_allow_html=True)
    st.header("Read the experiment")
    vae_metrics = read_metrics(OUTPUTS / "vae" / "metrics.csv")
    gan_metrics = read_metrics(OUTPUTS / "gan" / "metrics.csv")
    cards = st.columns(4)
    if vae_metrics:
        with cards[0]: metric_card("VAE total / final", f"{vae_metrics['loss'][-1]:.2f}")
        with cards[1]: metric_card("VAE KL / final", f"{vae_metrics['kl'][-1]:.2f}")
    if gan_metrics:
        with cards[2]: metric_card("GAN discriminator / final", f"{gan_metrics['discriminator'][-1]:.2f}")
        with cards[3]: metric_card("GAN generator / final", f"{gan_metrics['generator'][-1]:.2f}")
    left, right = st.columns(2)
    with left:
        st.subheader("VAE losses")
        if vae_metrics:
            st.line_chart({"total": vae_metrics["loss"], "reconstruction": vae_metrics["reconstruction"], "KL": vae_metrics["kl"]})
            st.image(str(OUTPUTS / "vae" / "random_samples.png"), caption="VAE random samples", use_container_width=True)
        else:
            st.info("No VAE metrics yet.")
    with right:
        st.subheader("GAN losses")
        if gan_metrics:
            st.line_chart({"discriminator": gan_metrics["discriminator"], "generator": gan_metrics["generator"]})
            st.image(str(OUTPUTS / "gan" / "samples.png"), caption="DCGAN samples", use_container_width=True)
        else:
            st.info("No GAN metrics yet.")


def main() -> None:
    device = choose_device()
    with st.sidebar:
        st.markdown("### FASHION FORGE")
        st.caption("A small studio for learned clothing representations")
        page = st.radio("Workspace", ["Generate", "Reconstruct", "Metrics"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Runtime**")
        st.code(str(device).upper())
        st.caption("Models are loaded from outputs/ and kept in memory for the session.")
    st.markdown('<div class="hero"><div class="eyebrow">fashion image generation / local studio</div><h1>Make the latent space tangible.</h1><p class="lede">A quiet interface for inspecting what the VAE represents and what the DCGAN invents.</p></div>', unsafe_allow_html=True)
    if page == "Generate":
        gan_page(device)
    elif page == "Reconstruct":
        vae_page(device)
    else:
        metrics_page()


if __name__ == "__main__":
    main()
