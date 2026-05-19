import torch
import time
import numpy as np
import ssl
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report
from fpdf import FPDF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adavit_a3c_tb import AdaViTDynamic

# --- SSL FIX FOR JETSON DOCKER CONTAINERS ---
ssl._create_default_https_context = ssl._create_unverified_context
# --------------------------------------------

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def setup_output_directory():
    """Creates a timestamped folder inside 'outputs' and sets up logging."""
    base_dir = "outputs"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}_master")
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "deployment.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Created master output directory: {run_dir}")
    return run_dir

def generate_extensive_graphs(latencies, keep_rates, all_targets, all_preds, run_dir):
    """Generates a 4-panel dashboard of edge deployment metrics."""
    logging.info("Generating extensive visualization dashboard...")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Jetson AGX Orin - AdaViT Edge Deployment Analysis', fontsize=20, fontweight='bold')

    # 1. Latency Distribution
    axs[0, 0].hist(latencies, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    axs[0, 0].set_title('Batch Processing Latency Distribution', fontsize=14)
    axs[0, 0].set_xlabel('Latency (ms)', fontsize=12)
    axs[0, 0].set_ylabel('Frequency', fontsize=12)
    axs[0, 0].grid(axis='y', alpha=0.3)
    axs[0, 0].axvline(np.mean(latencies), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(latencies):.1f}ms')
    axs[0, 0].legend()

    # 2. Keep Rate vs Latency Scatter
    axs[0, 1].scatter(keep_rates, latencies, color='darkorange', alpha=0.6, edgecolors='k')
    axs[0, 1].set_title('Dynamic Routing Efficiency (Keep Rate vs Latency)', fontsize=14)
    axs[0, 1].set_xlabel('Average Batch Keep Rate (%)', fontsize=12)
    axs[0, 1].set_ylabel('Latency (ms)', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    if len(keep_rates) > 1 and len(latencies) > 1:
        z = np.polyfit(keep_rates, latencies, 1)
        p = np.poly1d(z)
        axs[0, 1].plot(keep_rates, p(keep_rates), "r--", alpha=0.8, label="Trendline")
        axs[0, 1].legend()

    # 3. Confusion Matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
    im = axs[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Inference Confusion Matrix', fontsize=14)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CIFAR10_CLASSES))
    axs[1, 0].set_xticks(tick_marks)
    axs[1, 0].set_yticks(tick_marks)
    axs[1, 0].set_xticklabels(CIFAR10_CLASSES, rotation=45)
    axs[1, 0].set_yticklabels(CIFAR10_CLASSES)
    axs[1, 0].set_ylabel('True Label', fontsize=12)
    axs[1, 0].set_xlabel('Predicted Label', fontsize=12)

    # 4. Per-Class Accuracy
    class_correct = cm.diagonal()
    class_totals = cm.sum(axis=1)
    class_acc = (class_correct / class_totals) * 100
    bars = axs[1, 1].bar(CIFAR10_CLASSES, class_acc, color='seagreen', edgecolor='black')
    axs[1, 1].set_title('Per-Class Accuracy (%)', fontsize=14)
    axs[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axs[1, 1].set_ylim(0, 100)
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_path = os.path.join(run_dir, "dashboard.png")
    plt.savefig(dash_path, dpi=300, bbox_inches='tight')
    plt.close()
    return class_acc.tolist(), cm.tolist(), dash_path

def visualize_dropped_patches(images, targets, preds, patch_policies, run_dir, num_samples=8):
    """Creates an X-Ray visualization showing exactly which patches the model dropped."""
    logging.info("Generating dynamic patch 'X-Ray' visualizations...")
    inv_normalize = transforms.Normalize(
        mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
        std=[1/0.2023, 1/0.1994, 1/0.2010]
    )
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('AdaViT Dynamic Routing: Dropped vs. Kept Patches', fontsize=18, fontweight='bold')
    axs = axs.flatten()
    samples_to_draw = min(num_samples, len(images))
    
    for i in range(samples_to_draw):
        img = images[i].cpu()
        img = inv_normalize(img)
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        
        true_label = CIFAR10_CLASSES[targets[i]]
        pred_label = CIFAR10_CLASSES[preds[i]]
        color = 'green' if true_label == pred_label else 'red'
        axs[i].set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=10)
        axs[i].axis('off')
        
        try:
            policy = patch_policies[i].cpu().numpy().flatten()
            patch_size = 4
            grid_size = 32 // patch_size
            for row in range(grid_size):
                for col in range(grid_size):
                    patch_idx = row * grid_size + col
                    if patch_idx < len(policy) and policy[patch_idx] < 0.5:
                        rect = patches.Rectangle(
                            (col * patch_size, row * patch_size), 
                            patch_size, patch_size, 
                            linewidth=0, edgecolor='none', facecolor='black', alpha=0.85
                        )
                        axs[i].add_patch(rect)
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    patch_path = os.path.join(run_dir, "patch_xray.png")
    plt.savefig(patch_path, dpi=300, bbox_inches='tight')
    plt.close()
    return patch_path

def generate_pdf_report(run_dir, dash_path, xray_path, results_dict):
    """Compiles everything into a professional PDF document."""
    logging.info("Compiling final Executive PDF Report...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Title & Metrics
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AdaViT Edge Deployment Report - Jetson AGX Orin", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "System & Hardware Profile", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Timestamp: {results_dict['metadata']['timestamp']}", ln=True)
    pdf.cell(0, 8, f"Hardware: NVIDIA Jetson AGX Orin ({results_dict['metadata']['hardware'].upper()})", ln=True)
    pdf.cell(0, 8, f"Batch Size Evaluated: {results_dict['metadata']['batch_size']}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "High-Level Performance Summary", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Overall Accuracy: {results_dict['model_performance']['overall_accuracy_percent']}%", ln=True)
    pdf.cell(0, 8, f"Average Patch Keep Rate: {results_dict['model_performance']['average_keep_rate_percent']}% (Lower is faster)", ln=True)
    pdf.cell(0, 8, f"System Throughput: {results_dict['system_metrics']['throughput_fps']} FPS", ln=True)
    pdf.cell(0, 8, f"Average Latency per Batch: {results_dict['system_metrics']['avg_latency_per_batch_ms']} ms", ln=True)
    pdf.cell(0, 8, f"Average Latency per Image: {results_dict['system_metrics']['avg_latency_per_image_ms']} ms", ln=True)
    
    # Page 2: Dashboard
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Visual Metrics Dashboard", ln=True, align='C')
    pdf.image(dash_path, x=10, w=190)
    
    # Page 3: Patch X-Rays
    if xray_path and os.path.exists(xray_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. Dynamic Routing X-Ray (Dropped Patches)", ln=True, align='C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "Black squares indicate image patches the model dynamically chose to ignore to save compute.", ln=True, align='C')
        pdf.image(xray_path, x=10, w=190)

    pdf_path = os.path.join(run_dir, "Executive_Deployment_Report.pdf")
    pdf.output(pdf_path)
    logging.info(f"🎉 PDF Report successfully generated at: {pdf_path}")

def profile_model(model_path, batch_size=64):
    run_dir = setup_output_directory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Deploying on: {device}")

    model = AdaViTDynamic(
        image_size=32, patch_size=4, num_classes=10, 
        dim=256, depth=8, heads=8, mlp_dim=512
    ).to(device)
    
    logging.info(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    logging.info("Loading CIFAR-10 validation data...")
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_ds = datasets.CIFAR10("./data", train=False, download=True, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    logging.info("Warming up Orin GPU (20 dummy batches)...")
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input, is_training=False, policy_active=True)

    logging.info(f"Starting detailed metrics profiling (Batch Size: {batch_size})...")
    latencies, keep_rates, all_targets, all_preds = [], [], [], []
    sample_images, sample_policies = None, None

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = model(data, is_training=False, policy_active=True)
            end_event.record()
            
            torch.cuda.synchronize() 
            
            latencies.append(start_event.elapsed_time(end_event))
            keep_rates.append(output["keep_prob"].mean().item() * 100) 
            
            preds = output["logits"].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if i == 0:
                sample_images = data.clone()
                if "policy" in output: sample_policies = output["policy"].clone()
                elif "keep_prob" in output: sample_policies = output["keep_prob"].clone()

    avg_latency_ms = np.mean(latencies)
    throughput_fps = (1000.0 / avg_latency_ms) * batch_size
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    total_correct = (all_preds == all_targets).sum()
    accuracy = (total_correct / len(all_targets)) * 100
    avg_keep = np.mean(keep_rates)

    logging.info("\n" + "="*50)
    logging.info(" 🚀 ORIN EDGE DEPLOYMENT SUMMARY")
    logging.info("="*50)
    logging.info(f"Throughput: {throughput_fps:.2f} FPS | Keep Rate: {avg_keep:.2f}% | Acc: {accuracy:.2f}%")
    logging.info("="*50)

    # Generate Dashboards
    class_acc, conf_matrix, dash_path = generate_extensive_graphs(latencies, keep_rates, all_targets, all_preds, run_dir)
    
    xray_path = None
    if sample_images is not None and sample_policies is not None:
        xray_path = visualize_dropped_patches(sample_images, all_targets[:batch_size], all_preds[:batch_size], sample_policies, run_dir)

    logging.info("Calculating Precision, Recall, and F1-Scores...")
    class_report_dict = classification_report(all_targets, all_preds, target_names=CIFAR10_CLASSES, output_dict=True)

    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": str(device), "batch_size": batch_size, "total_images_processed": len(all_targets)
        },
        "system_metrics": {
            "throughput_fps": round(throughput_fps, 2), "avg_latency_per_batch_ms": round(avg_latency_ms, 2),
            "avg_latency_per_image_ms": round(avg_latency_ms / batch_size, 2)
        },
        "model_performance": {
            "overall_accuracy_percent": round(accuracy, 2), "average_keep_rate_percent": round(avg_keep, 2),
            "detailed_classification_report": class_report_dict
        }
    }

    json_path = os.path.join(run_dir, "extensive_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # GENERATE THE FINAL PDF
    generate_pdf_report(run_dir, dash_path, xray_path, results)
    logging.info(f"✅ Run complete. Check {run_dir} for your PDF and metrics!\n")

if __name__ == "__main__":
    profile_model("best_model.pth", batch_size=128)
