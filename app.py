import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO
import requests

from adavit_model import AdaViTDynamic

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PatchWise — Adaptive Sparse ViT",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts ────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e6e1;
}

.stApp {
    background-color: #0a0a0b;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56, 189, 248, 0.06) 0%, transparent 70%),
        linear-gradient(180deg, #0a0a0b 0%, #0c0d10 100%);
}

.block-container {
    max-width: 1240px;
    padding: 0 2rem 6rem 2rem;
    margin: 0 auto;
}

/* ── Hide streamlit chrome ────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── Divider line ─────────────────────────────────────────────────────────── */
.rule {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(248,248,242,0.08) 20%, rgba(248,248,242,0.08) 80%, transparent);
    margin: 0;
    border: none;
}

/* ── Nav bar ──────────────────────────────────────────────────────────────── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0 1.5rem 0;
    margin-bottom: 0;
}

.nav-wordmark {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    font-size: 1rem;
    letter-spacing: 0.18em;
    color: #f8f8f2;
    text-transform: uppercase;
}

.nav-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: rgba(248,248,242,0.35);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.hero-wrap {
    padding: 5rem 0 4.5rem 0;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 2rem;
    height: 1px;
    background: #38bdf8;
}

.hero-h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 5rem) !important;
    font-weight: 300 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    color: #f8f8f2 !important;
    margin-bottom: 1rem !important;
}

.hero-h1 em {
    font-style: italic;
    color: rgba(248,248,242,0.38);
}

.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: rgba(248,248,242,0.55);
    line-height: 1.75;
    max-width: 640px;
    margin-bottom: 3rem;
}

.hero-sub strong {
    color: rgba(248,248,242,0.85);
    font-weight: 400;
}

/* ── Stat strip ───────────────────────────────────────────────────────────── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(248,248,242,0.07);
    border: 1px solid rgba(248,248,242,0.07);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 5rem;
}

.stat-cell {
    background: #0c0d10;
    padding: 1.5rem 1.75rem;
    transition: background 0.2s;
}

.stat-cell:hover { background: #111215; }

.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #f8f8f2;
    line-height: 1;
    margin-bottom: 0.4rem;
    letter-spacing: -0.03em;
}

.stat-num span {
    font-size: 1.1rem;
    font-weight: 300;
    color: #38bdf8;
}

.stat-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.35);
    font-family: 'DM Mono', monospace;
}

/* ── Section heading ──────────────────────────────────────────────────────── */
.sec-head {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.75rem;
}

.sec-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: rgba(248,248,242,0.25);
    letter-spacing: 0.1em;
}

.sec-title {
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.45);
    font-family: 'DM Mono', monospace;
}

/* ── Upload zone ──────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: 1px solid rgba(248,248,242,0.09) !important;
    border-radius: 2px !important;
    padding: 0.5rem 1rem !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.35) !important;
}

[data-testid="stFileUploader"] * {
    color: rgba(248,248,242,0.55) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stFileUploader"] button {
    border: 1px solid rgba(56,189,248,0.4) !important;
    background: transparent !important;
    color: #38bdf8 !important;
    border-radius: 1px !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.35rem 1rem !important;
}

/* ── Sample thumbnails ────────────────────────────────────────────────────── */
.thumb-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.3);
    margin-top: 0.5rem;
    text-align: center;
}

/* ── Streamlit image border ───────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border: 1px solid rgba(248,248,242,0.08);
    border-radius: 2px;
}

/* ── Streamlit button ─────────────────────────────────────────────────────── */
.stButton button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid rgba(248,248,242,0.1) !important;
    color: rgba(248,248,242,0.45) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 1px !important;
    transition: all 0.15s !important;
    margin-top: 0.35rem !important;
}

.stButton button:hover {
    border-color: rgba(56,189,248,0.45) !important;
    color: #38bdf8 !important;
    background: rgba(56,189,248,0.04) !important;
}

/* ── Result panel ─────────────────────────────────────────────────────────── */
.result-panel {
    border: 1px solid rgba(248,248,242,0.08);
    border-radius: 2px;
    padding: 2rem;
    background: #0c0d10;
    height: 100%;
}

.result-class {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.3);
    margin-bottom: 0.5rem;
}

.result-label {
    font-size: 2.6rem;
    font-weight: 300;
    letter-spacing: -0.03em;
    color: #f8f8f2;
    line-height: 1;
    margin-bottom: 0.25rem;
    text-transform: capitalize;
}

.result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 3.5rem;
    font-weight: 500;
    letter-spacing: -0.04em;
    color: #38bdf8;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.result-conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.25);
    margin-bottom: 2rem;
}

.metric-row {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(248,248,242,0.06);
}

.metric-line {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}

.metric-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.3);
}

.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: rgba(248,248,242,0.75);
}

.bar-wrap {
    height: 2px;
    background: rgba(248,248,242,0.07);
    border-radius: 1px;
    overflow: hidden;
    margin-top: 0.25rem;
}

.bar-fill {
    height: 100%;
    background: #38bdf8;
    border-radius: 1px;
    transition: width 0.6s ease;
}

.bar-fill-warn {
    background: #fb923c;
}

/* ── Patch map label ──────────────────────────────────────────────────────── */
.map-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.25);
    margin-bottom: 0.5rem;
}

/* ── Architecture table ───────────────────────────────────────────────────── */
.arch-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}

.arch-table tr {
    border-bottom: 1px solid rgba(248,248,242,0.05);
}

.arch-table tr:last-child {
    border-bottom: none;
}

.arch-table td {
    padding: 0.8rem 0;
    line-height: 1.4;
}

.arch-table td:first-child {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.3);
    width: 42%;
    padding-right: 1.5rem;
}

.arch-table td:last-child {
    color: rgba(248,248,242,0.75);
    font-weight: 300;
}

/* ── Contribution items ───────────────────────────────────────────────────── */
.contrib-item {
    display: flex;
    gap: 1.25rem;
    align-items: flex-start;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(248,248,242,0.05);
}

.contrib-item:last-child { border-bottom: none; }

.contrib-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: rgba(56,189,248,0.6);
    letter-spacing: 0.05em;
    padding-top: 0.1rem;
    min-width: 1.5rem;
}

.contrib-text {
    font-size: 0.88rem;
    font-weight: 300;
    color: rgba(248,248,242,0.65);
    line-height: 1.65;
}

.contrib-text strong {
    color: rgba(248,248,242,0.88);
    font-weight: 400;
}

/* ── Phase timeline ───────────────────────────────────────────────────────── */
.phase-wrap {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.phase-item {
    display: grid;
    grid-template-columns: 7rem 1fr;
    gap: 1.5rem;
    padding: 1.5rem 0;
    border-bottom: 1px solid rgba(248,248,242,0.05);
    align-items: start;
}

.phase-item:last-child { border-bottom: none; }

.phase-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.2);
    padding-top: 0.1rem;
}

.phase-name {
    font-size: 0.88rem;
    font-weight: 400;
    color: rgba(248,248,242,0.85);
    margin-bottom: 0.3rem;
}

.phase-desc {
    font-size: 0.8rem;
    font-weight: 300;
    color: rgba(248,248,242,0.4);
    line-height: 1.6;
}

.phase-pill {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.2rem 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 1px;
}

.pill-resolved {
    background: rgba(56,189,248,0.08);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
}

.pill-limited {
    background: rgba(251,146,60,0.08);
    color: #fb923c;
    border: 1px solid rgba(251,146,60,0.2);
}

/* ── Upload box placeholder ───────────────────────────────────────────────── */
.upload-placeholder {
    border: 1px solid rgba(248,248,242,0.07);
    border-radius: 2px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: #0c0d10;
}

.upload-placeholder p {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.2);
    line-height: 2;
    margin: 0;
}

/* ── Footer ───────────────────────────────────────────────────────────────── */
.footer {
    margin-top: 6rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(248,248,242,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(248,248,242,0.2);
}

.footer-right {
    font-size: 0.75rem;
    color: rgba(248,248,242,0.18);
    font-weight: 300;
}

/* ── Streamlit image override ─────────────────────────────────────────────── */
[data-testid="stImage"] {
    border-radius: 2px;
    overflow: hidden;
}

/* ── Hide streamlit label gaps ────────────────────────────────────────────── */
.stFileUploader > label { display: none !important; }
div[data-testid="stVerticalBlock"] > div:empty { display: none; }

/* ── Expander ─────────────────────────────────────────────────────────────── */
details {
    border: 1px solid rgba(248,248,242,0.07) !important;
    border-radius: 2px !important;
    background: #0c0d10 !important;
}

summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(248,248,242,0.4) !important;
    padding: 1rem 1.25rem !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE + MODEL
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = AdaViTDynamic(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
    )
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

model = load_model()

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(image: Image.Image):
    """Returns (predicted_class, confidence, keep_rate, flops_saved, patch_mask_8x8)."""
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)

    # Unpack — model may return (logits, extras) tuple or a dict
    if isinstance(outputs, tuple):
        raw, extras = outputs[0], outputs[1] if len(outputs) > 1 else {}
    else:
        raw = outputs

    if isinstance(raw, dict):
        logits = raw["logits"]
        mask_flat = raw.get("mask", None)
    else:
        logits = raw
        mask_flat = extras.get("mask", None) if isinstance(extras, dict) else None

    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()

    if mask_flat is not None:
        mask = mask_flat[0].cpu().numpy()
        keep_rate = float(mask.mean()) * 100
        patch_grid = mask.reshape(8, 8)
    else:
        keep_rate = 100.0
        patch_grid = np.ones((8, 8))

    flops_saved = 100.0 - keep_rate
    return CLASSES[pred_idx], confidence * 100, keep_rate, flops_saved, patch_grid


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE IMAGES
# ─────────────────────────────────────────────────────────────────────────────

# Handpicked samples — clear, colorful, subject-centered images.
# Using Wikimedia Commons with proper headers — reliably public domain.
# Four demo images — multiple URL fallbacks per class so if one CDN
# blocks, the next is tried. All are subject-centered, colorful images
# that produce interesting patch routing maps.
DEMO_SAMPLES = [
    ("Frog", [
        "https://images.pexels.com/photos/2062316/pexels-photo-2062316.jpeg",
        "https://images.pexels.com/photos/70083/frog-macro-amphibian-green-70083.jpeg",
        "https://images.pexels.com/photos/145683/pexels-photo-145683.jpeg",
        "https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg",
    ]),
    ("Automobile", [
        "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg",
        "https://images.pexels.com/photos/1149831/pexels-photo-1149831.jpeg",
        "https://images.pexels.com/photos/707046/pexels-photo-707046.jpeg",
        "https://images.pexels.com/photos/3802510/pexels-photo-3802510.jpeg",
    ]),
]

@st.cache_resource(show_spinner=False)
def load_demo_samples():
    """For each demo class, try each fallback URL until one succeeds.
    Downscales to 32x32 for inference, upscales to 256x256 for display."""
    import requests as _req

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.pexels.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }

    samples = []
    for label, urls in DEMO_SAMPLES:
        img_32 = None
        for url in urls:
            try:
                resp = _req.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    img_32 = img.resize((32, 32), Image.LANCZOS)
                    break
            except Exception:
                continue

        if img_32 is None:
            # All URLs failed — make a synthetic fallback that's still recognisable
            img_32 = _make_synthetic(label)

        img_256 = img_32.resize((256, 256), Image.NEAREST)
        samples.append((label, img_32, img_256))

    return samples  # [(label, img_32, img_256), ...]


def _make_synthetic(label: str) -> Image.Image:
    """Generates a simple recognisable 32x32 stand-in if all URLs fail."""
    from PIL import ImageDraw, ImageFilter
    img = Image.new("RGB", (32, 32), (20, 20, 28))
    draw = ImageDraw.Draw(img)
    if label == "Dog":
        draw.rectangle([0, 0, 32, 32], fill=(200, 170, 120))
        draw.ellipse([8, 8, 24, 24], fill=(160, 120, 70))
        draw.ellipse([10, 5, 22, 15], fill=(170, 130, 80))
        draw.ellipse([7, 4, 13, 12], fill=(130, 90, 50))
        draw.ellipse([19, 4, 25, 12], fill=(130, 90, 50))
    elif label == "Frog":
        draw.rectangle([0, 0, 32, 32], fill=(60, 120, 50))
        draw.ellipse([6, 10, 26, 28], fill=(50, 160, 45))
        draw.ellipse([8, 4, 24, 18], fill=(60, 175, 55))
        draw.ellipse([7, 3, 14, 11], fill=(220, 220, 60))
        draw.ellipse([18, 3, 25, 11], fill=(220, 220, 60))
        draw.ellipse([8, 4, 13, 10], fill=(20, 70, 20))
        draw.ellipse([19, 4, 24, 10], fill=(20, 70, 20))
    elif label == "Ship":
        draw.rectangle([0, 0, 32, 18], fill=(100, 150, 210))
        draw.rectangle([0, 18, 32, 32], fill=(40, 90, 160))
        draw.polygon([(3, 18), (29, 18), (27, 25), (5, 25)], fill=(230, 230, 230))
        draw.rectangle([8, 11, 24, 18], fill=(200, 200, 200))
        draw.rectangle([12, 6, 20, 11], fill=(210, 210, 210))
        draw.rectangle([15, 2, 17, 6], fill=(180, 50, 50))
    elif label == "Automobile":
        draw.rectangle([0, 0, 32, 32], fill=(160, 185, 160))
        draw.rectangle([0, 24, 32, 32], fill=(70, 70, 70))
        draw.rectangle([3, 15, 29, 24], fill=(200, 40, 40))
        draw.polygon([(8, 9), (24, 9), (28, 15), (4, 15)], fill=(175, 30, 30))
        draw.rectangle([9, 10, 15, 14], fill=(140, 200, 230))
        draw.rectangle([17, 10, 23, 14], fill=(140, 200, 230))
        draw.ellipse([4, 20, 12, 28], fill=(25, 25, 25))
        draw.ellipse([20, 20, 28, 28], fill=(25, 25, 25))
    return img.filter(ImageFilter.GaussianBlur(0.3))


# selected_sample_idx: int index into demo_samples list, or None
if "selected_sample_idx" not in st.session_state:
    st.session_state.selected_sample_idx = None


# ─────────────────────────────────────────────────────────────────────────────
# NAV
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="nav">
  <span class="nav-wordmark">PatchWise</span>
  <span class="nav-tag">UCE Osmania University &nbsp;·&nbsp; CSE 2025–26</span>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Adaptive Sparse Vision Transformers</div>
  <h1 class="hero-h1">
    Not every patch<br><em>deserves</em> attention.
  </h1>
  <p class="hero-sub">
    PatchWise uses a reinforcement-learned policy to decide, per image, 
    which patches are worth computing — and physically drops the rest.
    <strong>Same accuracy. Half the FLOPs.</strong>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stat strip ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-strip">
  <div class="stat-cell">
    <div class="stat-num">54.2<span>%</span></div>
    <div class="stat-label">Attention FLOPs saved</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num">79.4<span>%</span></div>
    <div class="stat-label">CIFAR-10 accuracy</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num">2033<span>fps</span></div>
    <div class="stat-label">Jetson AGX Orin</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num">0.49<span>ms</span></div>
    <div class="stat-label">Inference latency</div>
  </div>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 01 — INFERENCE DEMO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">
  <span class="sec-num">01</span>
  <span class="sec-title">Live Sparse Inference</span>
</div>
""", unsafe_allow_html=True)

# ── Load demo samples (cached after first fetch) ─────────────────────────────
with st.spinner("Fetching samples…"):
    demo_samples = load_demo_samples()  # [(label, img_32, img_256), ...]

# ── Upload row ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Sample thumbnails row (full width, 4 equal columns) ──────────────────────
st.markdown("""
<p style="font-family:'DM Mono',monospace;font-size:0.65rem;
letter-spacing:0.15em;text-transform:uppercase;
color:rgba(248,248,242,0.25);margin-bottom:0.85rem;">
Or try a sample — processed at CIFAR-10 resolution (32×32)
</p>
""", unsafe_allow_html=True)

s_cols = st.columns(len(demo_samples), gap="medium")

for i, (label, img_32, img_256) in enumerate(demo_samples):
    with s_cols[i]:
        st.image(img_256, use_container_width=True)

        if st.button(label, key=f"sample_{i}"):
            st.session_state.selected_sample_idx = i
            uploaded = None

# ── Resolve active image ──────────────────────────────────────────────────────

active_image = None
active_label = None

if uploaded is not None:
    active_image = Image.open(uploaded).convert("RGB")
    active_label = "uploaded"
    st.session_state.selected_sample_idx = None
elif st.session_state.selected_sample_idx is not None:
    idx = st.session_state.selected_sample_idx
    active_label = demo_samples[idx][0]
    active_image = demo_samples[idx][1]   # use the 32×32 version for inference

# ── Inference output ──────────────────────────────────────────────────────────

st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)

if active_image is not None:
    pred_class, confidence, keep_rate, flops_saved, patch_grid = run_inference(active_image)

    out_img_col, out_patch_col, out_result_col = st.columns([1, 1, 1.4], gap="large")

    with out_img_col:
        st.markdown('<p class="map-label">Input image</p>', unsafe_allow_html=True)
        st.image(active_image, use_container_width=True)

    with out_patch_col:
        st.markdown('<p class="map-label">Active patch map</p>', unsafe_allow_html=True)
        # Render patch grid as a colour-coded image
        patch_vis = np.zeros((patch_grid.shape[0], patch_grid.shape[1], 3), dtype=np.uint8)
        patch_vis[patch_grid == 1] = [56, 189, 248]   # cyan-ish = kept
        patch_vis[patch_grid == 0] = [18, 18, 22]      # near-black = pruned
        patch_img = Image.fromarray(patch_vis).resize((128, 128), Image.NEAREST)
        st.image(patch_img, use_container_width=True)

    with out_result_col:
        st.markdown(f"""
        <div class="result-panel">
          <div class="result-class">Prediction</div>
          <div class="result-label">{pred_class}</div>

          <div style="height:1.25rem"></div>

          <div class="result-class">Confidence</div>
          <div class="result-conf">{confidence:.1f}<span style="font-size:1.4rem;color:rgba(56,189,248,0.6)">%</span></div>

          <div class="metric-row">
            <div>
              <div class="metric-line">
                <span class="metric-key">Patches kept</span>
                <span class="metric-val">{keep_rate:.1f}%</span>
              </div>
              <div class="bar-wrap">
                <div class="bar-fill" style="width:{keep_rate:.1f}%"></div>
              </div>
            </div>
            <div>
              <div class="metric-line">
                <span class="metric-key">FLOPs saved</span>
                <span class="metric-val">{flops_saved:.1f}%</span>
              </div>
              <div class="bar-wrap">
                <div class="bar-fill bar-fill-warn" style="width:{flops_saved:.1f}%"></div>
              </div>
            </div>
            <div>
              <div class="metric-line">
                <span class="metric-key">Routing policy</span>
                <span class="metric-val">A3C · EMA-controlled</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-placeholder">
      <p>Upload an image or select a sample above<br>to observe adaptive patch routing live.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 02 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <span class="sec-num">02</span>
  <span class="sec-title">Three-Phase Development</span>
</div>

<div class="phase-wrap">

  <div class="phase-item">
    <div class="phase-tag">Phase 01</div>
    <div>
      <div class="phase-name">Gumbel-Softmax differentiable approximation</div>
      <div class="phase-desc">
        Embedded a lightweight decision network into DeiT-Small. Used the Gumbel-Softmax 
        reparameterization trick to allow gradient flow through discrete keep/drop decisions. 
        Dual-objective loss: cross-entropy + MSE usage penalty toward a target keep rate.
      </div>
      <span class="phase-pill pill-limited">Training–inference gap · soft probs vs hard actions</span>
    </div>
  </div>

  <div class="phase-item">
    <div class="phase-tag">Phase 02</div>
    <div>
      <div class="phase-name">Hybrid single-agent A2C with Bernoulli policy</div>
      <div class="phase-desc">
        Transitioned to true discrete decisions. Actor outputs one logit per patch sampled 
        via a Bernoulli distribution. Patches physically removed before attention blocks using 
        <code style="font-family:'DM Mono',monospace;font-size:0.78em;color:rgba(248,248,242,0.55)">x.detach()</code> 
        gating. Critic reduces variance via advantage estimation.
      </div>
      <span class="phase-pill pill-limited">High variance · correlated sequential data · gradient conflict</span>
    </div>
  </div>

  <div class="phase-item">
    <div class="phase-tag">Phase 03</div>
    <div>
      <div class="phase-name">Asynchronous A3C · Categorical policy · EMA controller</div>
      <div class="phase-desc">
        Four parallel Hogwild! workers decorrelate training data. Categorical distribution 
        with two independent logits per patch eliminates gradient tug-of-war. EMA controller 
        enforces keep-rate budget. Attention-guided dense rewards from early transformer layers. 
        Curriculum warmup stabilises joint backbone + policy training.
      </div>
      <span class="phase-pill pill-resolved">Stable convergence · 79.4% accuracy @ ≈47.7% keep rate</span>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 03 — CONTRIBUTIONS + ARCHITECTURE (side by side)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)

contrib_col, arch_col = st.columns([1, 1], gap="large")

with contrib_col:
    st.markdown("""
    <div class="sec-head">
      <span class="sec-num">03</span>
      <span class="sec-title">Contributions</span>
    </div>

    <div class="contrib-item">
      <span class="contrib-num">01</span>
      <span class="contrib-text">
        <strong>Attention-Guided Reward Mechanism</strong> — dense, spatially-aware 
        feedback extracted from early transformer layers, replacing sparse end-of-episode signals.
      </span>
    </div>
    <div class="contrib-item">
      <span class="contrib-num">02</span>
      <span class="contrib-text">
        <strong>EMA Controller</strong> — exponential moving average of batch-wise keep rate 
        enforces a strict computational budget, eliminating keep-rate collapse.
      </span>
    </div>
    <div class="contrib-item">
      <span class="contrib-num">03</span>
      <span class="contrib-text">
        <strong>Curriculum Warmup</strong> — RL policy disabled during early epochs so the 
        backbone learns stable representations before pruning activates.
      </span>
    </div>
    <div class="contrib-item">
      <span class="contrib-num">04</span>
      <span class="contrib-text">
        <strong>Categorical → Bernoulli upgrade</strong> — dual-logit formulation decouples 
        accuracy and sparsity gradients, resolving the Phase 2 tug-of-war.
      </span>
    </div>
    <div class="contrib-item">
      <span class="contrib-num">05</span>
      <span class="contrib-text">
        <strong>Hogwild! A3C parallelisation</strong> — four asynchronous workers update a 
        shared global model, decorrelating data and accelerating convergence.
      </span>
    </div>
    <div class="contrib-item">
      <span class="contrib-num">06</span>
      <span class="contrib-text">
        <strong>Edge validation on NVIDIA Jetson AGX Orin</strong> — 2033 FPS, 0.49 ms 
        latency, confirming real-world deployment viability.
      </span>
    </div>
    """, unsafe_allow_html=True)

with arch_col:
    st.markdown("""
    <div class="sec-head">
      <span class="sec-num">04</span>
      <span class="sec-title">Model Configuration</span>
    </div>

    <table class="arch-table">
      <tr><td>Backbone</td><td>Custom Vision Transformer (DeiT-inspired)</td></tr>
      <tr><td>RL Controller</td><td>Asynchronous A3C · 4 workers</td></tr>
      <tr><td>Policy distribution</td><td>Categorical (Softmax, 2 logits / patch)</td></tr>
      <tr><td>Dataset</td><td>CIFAR-10 · CIFAR-100 stress test</td></tr>
      <tr><td>Input resolution</td><td>32 × 32 px</td></tr>
      <tr><td>Patch size</td><td>4 × 4 → 64 tokens total</td></tr>
      <tr><td>Embedding dim</td><td>256</td></tr>
      <tr><td>Depth</td><td>6 transformer layers</td></tr>
      <tr><td>Attention heads</td><td>8</td></tr>
      <tr><td>Target keep rate</td><td>50% (EMA-enforced)</td></tr>
      <tr><td>Edge hardware</td><td>NVIDIA Jetson AGX Orin</td></tr>
      <tr><td>CIFAR-100 stability</td><td>~57% accuracy @ 300 epochs</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rule" style="margin-top:5rem"></div>
<div class="footer">
  <span class="footer-left">PatchWise &nbsp;·&nbsp; Adaptive Sparse Vision Transformers for Edge AI</span>
  <span class="footer-right">
    Srivarsha S · Aila Kaushik · Adla Sahithi &nbsp;—&nbsp; 
    UCE Osmania University · CSE · May 2026
  </span>
</div>
""", unsafe_allow_html=True)