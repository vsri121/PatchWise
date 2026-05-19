import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from adavit_model import AdaViTDynamic

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PatchWise",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        color: white;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    }

    h1, h2, h3 {
        color: white !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 15px;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 28px;
        font-weight: 700;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(168,85,247,0.2));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 30px;
    }

    .small-text {
        color: #cbd5e1;
        font-size: 16px;
    }

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# DEVICE
# =====================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =====================================================
# CLASSES
# =====================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = AdaViTDynamic(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=8,
        heads=8,
        mlp_dim=512
    ).to(DEVICE)

    checkpoint = torch.load(
        "best_model.pth",
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model

model = load_model()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🚀 PatchWise</h1>
        <p class="small-text">
        Reinforcement-Learned Adaptive Vision Transformer for Edge AI
        </p>
        <p class="small-text">
        Dynamic patch routing • Sparse inference • Intelligent compute allocation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

show_heatmap = st.sidebar.toggle("Show Patch Heatmap", value=True)

show_overlay = st.sidebar.toggle("Show Patch Overlay", value=True)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    ### 💻 Runtime
    **Device:** `{DEVICE}`
    """
)

# =====================================================
# MAIN
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    start = time.time()

    with torch.no_grad():

        output = model(
            input_tensor,
            is_training=False,
            policy_active=True
        )

    latency_ms = (time.time() - start) * 1000

    logits = output["logits"]

    probs = torch.softmax(logits, dim=1)

    pred = probs.argmax(dim=1).item()

    confidence = probs[0, pred].item()

    keep_prob = output["keep_prob"][0].cpu().numpy()

    mask = output["mask"][0].cpu().numpy()

    keep_rate = mask.mean() * 100

    flops_saved = 100 - keep_rate

    # =====================================================
    # TOP LAYOUT
    # =====================================================

    left, right = st.columns([1, 1])

    with left:

        st.subheader("🖼️ Input Image")

        st.image(image, use_container_width=True)

    with right:

        st.subheader("📊 Inference Metrics")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Prediction</div>
                    <div class="metric-value">{CIFAR10_CLASSES[pred]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Confidence</div>
                    <div class="metric-value">{confidence*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        c3, c4 = st.columns(2)

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Keep Rate</div>
                    <div class="metric-value">{keep_rate:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Latency</div>
                    <div class="metric-value">{latency_ms:.1f}ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Estimated FLOPs Saved</div>
                <div class="metric-value">{flops_saved:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # OVERLAY
    # =====================================================

    if show_overlay:

        st.subheader("🧠 Adaptive Patch Routing")

        fig, ax = plt.subplots(figsize=(6, 6))

        img_np = np.array(image.resize((32, 32)))

        ax.imshow(img_np)

        patch_size = 4
        grid_size = 8

        for row in range(grid_size):
            for col in range(grid_size):

                idx = row * grid_size + col

                if mask[idx] < 0.5:

                    rect = patches.Rectangle(
                        (col * patch_size, row * patch_size),
                        patch_size,
                        patch_size,
                        linewidth=0,
                        facecolor='black',
                        alpha=0.85
                    )

                    ax.add_patch(rect)

        ax.axis("off")

        st.pyplot(fig)

    # =====================================================
    # HEATMAP
    # =====================================================

    if show_heatmap:

        st.subheader("🔥 Policy Keep Probability Heatmap")

        heatmap = keep_prob.reshape(8, 8)

        fig2, ax2 = plt.subplots(figsize=(6, 6))

        im = ax2.imshow(
            heatmap,
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        ax2.axis("off")

        plt.colorbar(im)

        st.pyplot(fig2)

    # =====================================================
    # INSIGHT BOX
    # =====================================================

    st.subheader("⚡ Adaptive Compute Insight")

    if keep_rate < 40:

        st.success(
            "Simple image detected → computation aggressively reduced."
        )

    elif keep_rate < 70:

        st.info(
            "Balanced semantic complexity → moderate compute allocation."
        )

    else:

        st.warning(
            "Complex visual scene detected → retaining more patches for accuracy."
        )

else:

    st.markdown(
        """
        ## 👈 Upload an image from the sidebar

        Try images like:
        - airplanes in clear skies
        - cluttered dog scenes
        - frogs
        - ships

        and observe how PatchWise dynamically changes transformer computation.
        """
    )