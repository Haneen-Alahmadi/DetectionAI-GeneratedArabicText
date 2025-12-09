import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def save_pairplot(df, features, save_path="../reports/figures/FeaturePairplot.png"):
    if os.path.exists(save_path):
        print(f"Pairplot already exists at {save_path}")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.pairplot(df[features], corner=True, diag_kind="kde", plot_kws={"alpha": 0.5})
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/confusion_{model_name}.png"
    if os.path.exists(save_path):
        print(f"Confusion Matrix already exists: {save_path}")
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_scores, model_name="Model", save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/roc_{model_name}.png"
    if os.path.exists(save_path):
        print(f"ROC Curve already exists: {save_path}")
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
