# =========================================================
# PATCHWISE — CLEAN RESEARCH UI
# REPLACE ENTIRE app.py WITH THIS
# =========================================================
from io import BytesIO
import os
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO
import requests


from adavit_model import AdaViTDynamic

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PatchWise",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* APP BACKGROUND */
.stApp {
    background-color: #f5f7fb;
}

/* MAIN WIDTH */
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb;
    width: 320px !important;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* HERO */
.hero {
    background: white;
    border-radius: 30px;
    padding: 4rem;
    border: 1px solid #e7eaf0;
    box-shadow: 0 10px 30px rgba(15,23,42,0.04);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
    letter-spacing: -3px;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #475569;
    margin-top: 1.5rem;
    line-height: 1.9;
    max-width: 750px;
}

.hero-highlight {
    color: #2563eb;
    font-weight: 700;
}

/* METRIC CARDS */
.metric-card {
    background: white;
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 18px rgba(15,23,42,0.03);
    text-align: center;
    transition: 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    color: #111827;
}

.metric-label {
    margin-top: 0.7rem;
    color: #64748b;
    font-size: 0.95rem;
}

/* SECTIONS */
.section {
    background: white;
    border-radius: 28px;
    padding: 2.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 18px rgba(15,23,42,0.03);
    margin-top: 2rem;
}

.section-title {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 1.5rem;
    letter-spacing: -1px;
}

/* UPLOAD */
.upload-box {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 22px;
    padding: 4rem;
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
}

/* PREDICTION BOX */
.pred-box {
    background: #f8fafc;
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    color: #111827 !important;
}

.pred-box p {
    color: #374151 !important;
    font-size: 1rem;
    line-height: 1.8;
}

.pred-box b {
    color: #111827 !important;
}

.pred-box h1 {
    color: #2563eb !important;
}

.pred-box h2 {
    color: #111827 !important;
}

/* TABLES */
table {
    width: 100%;
    border-collapse: collapse;
}

td {
    padding: 18px;
    border-bottom: 1px solid #e5e7eb;
}

td:first-child {
    width: 38%;
    font-weight: 600;
    color: #111827;
}

td:last-child {
    color: #475569;
}

/* REMOVE STREAMLIT WEIRD GAPS */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
}

[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* BUTTONS */
.stButton button {
    border-radius: 12px;
    border: none;
    background: #2563eb;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1rem;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 5rem;
    font-size: 0.9rem;
}
            
/* Research contribution text fix */
.body-text {
    color: #1e293b !important;
    line-height: 1.9;
    font-size: 1.05rem;
}

.body-text li {
    color: #1e293b !important;
    margin-bottom: 0.9rem;
}
            
/* File uploader text visibility */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
}

[data-testid="stFileUploader"] * {
    color: #111827 !important;
}
            
/* Sample image cards */
button[kind="secondary"] {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #dbe4f0;
    background: white;
    color: #111827;
    font-weight: 600;
}

button[kind="secondary"]:hover {
    border-color: #2563eb;
    color: #2563eb;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cpu")

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = AdaViTDynamic(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512
    )

    checkpoint = torch.load(
        "best_model.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.eval()

    return model

model = load_model()

# =========================================================
# CLASSES
# =========================================================

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
PatchWise
</div>

<div class="hero-subtitle">

A reinforcement-learned sparse Vision Transformer framework
that dynamically decides which image patches deserve computation.

Traditional Vision Transformers process every token equally.
<span class="hero-highlight">PatchWise learns adaptive computation policies per image.</span>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("54.2%", "Attention FLOPs Saved"),
    ("80.1%", "Validation Accuracy"),
    ("62.9 ms", "Jetson Inference"),
    ("44.2%", "Mask Diversity")
]

for col, metric in zip([col1,col2,col3,col4], metrics):

    value, label = metric

    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("""
<div class="section">
<div class="section-title">
Upload or Try Demo Images
</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    key="main_uploader"
)



# =========================================================
# SAMPLE IMAGES
# =========================================================

st.markdown("""
<div class="section">
<div class="section-title">
Quick Demo Samples
</div>
</div>
""", unsafe_allow_html=True)

sample_cols = st.columns(4)

sample_paths = {
    "Airplane": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Boeing_747-8_first_flight_Everett%2C_WA.jpg",

    "Dog": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",

    "Frog": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Red_eyed_tree_frog_edit2.jpg",

    "Ship": "https://upload.wikimedia.org/wikipedia/commons/2/26/Queen_Mary_2_in_Long_Beach.jpg"
}

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

for idx, (col, (label, path)) in enumerate(
    zip(sample_cols, sample_paths.items())
):

    with col:

        if os.path.isfile(path):

            st.image(path, use_container_width=True)

            if st.button(
                label,
                key=f"sample_{idx}",
                use_container_width=True
            ):

                st.session_state.selected_sample = path

                st.markdown("""
                <script>
                const demoSection = window.parent.document.getElementById("live-demo");

                if (demoSection) {
                    demoSection.scrollIntoView({
                        behavior: "smooth"
                    });
                }
                </script>
                """, unsafe_allow_html=True)


# =========================================================
# LIVE DEMO
# =========================================================

st.markdown("""
    <div id="live-demo"></div>

    <div class="section">
    <div class="section-title">
    Live Sparse Inference Demo
    </div>
    """, unsafe_allow_html=True)

if uploaded_file or st.session_state.selected_sample:

    st.markdown("""
    <script>
    const demoSection = window.parent.document.getElementById("live-demo");
    if (demoSection) {
        demoSection.scrollIntoView({ behavior: "smooth" });
    }
    </script>
    """, unsafe_allow_html=True)

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

    else:

        response = requests.get(
            st.session_state.selected_sample
        )

        image = Image.open(
            BytesIO(response.content)
            ).convert("RGB")
        tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

    # Handle tuple outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]

    # Extract logits tensor from dict
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        probs = F.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        confidence = probs[0][pred].item()

        mask = outputs["mask"][0].cpu().numpy()
        keep_rate = mask.mean() * 100
        flops_saved = 100 - keep_rate

    with col2:

        st.markdown(f"""
        <div class="pred-box">

        <h2 style="margin-bottom:0.5rem;">
        {classes[pred].capitalize()}
        </h2>

        <p style="color:#6b7280;">
        Prediction Confidence
        </p>

        <h1 style="font-size:3rem;color:#2563eb;">
        {confidence*100:.2f}%
        </h1>

        <hr style="margin-top:2rem;margin-bottom:2rem;">

        <p><b>Keep Rate:</b> {keep_rate:.1f}%</p>
        <p><b>FLOPs Saved:</b> {flops_saved:.1f}%</p>
        <p><b>Inference Policy:</b> Adaptive Sparse Routing</p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Active Patch Map")

        patch_grid = mask.reshape(8, 8)

        st.image(
            patch_grid,
            clamp=True,
            use_container_width=True
        )

else:

    st.markdown("""
    <div class="upload-box">

    Upload an image to observe adaptive patch routing
    and sparse transformer inference in real time.

    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# WHY PATCHWISE
# =========================================================

st.markdown("""
<div class="section">

<div class="section-title">
Why PatchWise?
</div>

<table>

<tr>
<td>Traditional Vision Transformers</td>
<td>PatchWise</td>
</tr>

<tr>
<td>Processes all patches equally</td>
<td>Adaptive patch routing</td>
</tr>

<tr>
<td>Static computation</td>
<td>Dynamic compute allocation</td>
</tr>

<tr>
<td>Dense attention inference</td>
<td>Sparse transformer inference</td>
</tr>

<tr>
<td>High computational cost</td>
<td>Reduced FLOPs</td>
</tr>

<tr>
<td>Weak edge deployment feasibility</td>
<td>Edge-ready architecture</td>
</tr>

</table>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL CONFIG
# =========================================================

st.markdown("""
<div class="section">

<div class="section-title">
Model Configuration
</div>

<table>

<tr>
<td>Backbone</td>
<td>Custom Vision Transformer</td>
</tr>

<tr>
<td>RL Controller</td>
<td>A3C</td>
</tr>

<tr>
<td>Dataset</td>
<td>CIFAR-10</td>
</tr>

<tr>
<td>Input Resolution</td>
<td>32 × 32</td>
</tr>

<tr>
<td>Patch Size</td>
<td>4 × 4</td>
</tr>

<tr>
<td>Embedding Dimension</td>
<td>256</td>
</tr>

<tr>
<td>Transformer Depth</td>
<td>6 Layers</td>
</tr>

<tr>
<td>Attention Heads</td>
<td>8</td>
</tr>

<tr>
<td>Deployment Hardware</td>
<td>NVIDIA Jetson AGX Orin</td>
</tr>

</table>

</div>
""", unsafe_allow_html=True)

# =========================================================
# CONTRIBUTIONS
# =========================================================

st.markdown("""
<div class="section">

<div class="section-title">
Research Contributions
</div>

<div class="body-text">

<ul>

<li>
Reinforcement-learned adaptive token pruning for Vision Transformers
</li>

<li>
Dynamic sparse routing based on image complexity
</li>

<li>
Compute-aware policy optimization using A3C
</li>

<li>
Real-time edge deployment validation on NVIDIA Jetson AGX Orin
</li>

<li>
Interpretable patch selection behavior analysis
</li>

<li>
Accuracy-efficiency tradeoff evaluation across multiple keep-rate budgets
</li>

</ul>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

PatchWise • Adaptive Sparse Vision Transformers for Edge AI

</div>
""", unsafe_allow_html=True)