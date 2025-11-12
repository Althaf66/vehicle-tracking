import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
import json
import os
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ModelEvaluator:
    def __init__(self, model_path, classes_json_path, data_dir, model_name):
        """
        Initialize the evaluator for a specific model

        Args:
            model_path: Path to saved model weights (.pth file)
            classes_json_path: Path to JSON file containing class names
            data_dir: Directory containing train/val subdirectories
            model_name: Name for this model (e.g., 'color_classifier')
        """
        self.model_path = model_path
        self.model_name = model_name
        self.data_dir = data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load class names
        with open(classes_json_path, 'r') as f:
            self.class_names = json.load(f)

        self.num_classes = len(self.class_names)

        # Define transforms (same as training validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load validation dataset
        val_dir = os.path.join(data_dir, 'val')
        self.val_dataset = datasets.ImageFolder(val_dir, self.transform)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )

        # Load model
        self.model = self._load_model()

        print(f"Loaded {model_name}")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)

    def _load_model(self):
        """Load the trained ResNet-18 model"""
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        # Load saved weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        return model

    def get_predictions(self):
        """Get predictions and ground truth labels for validation set"""
        all_preds = []
        all_labels = []
        all_probs = []

        print(f"Getting predictions for {self.model_name}...")

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # Get predicted class
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(max(10, len(self.class_names)), max(8, len(self.class_names) * 0.8)))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
        plt.close()

    def plot_normalized_confusion_matrix(self, y_true, y_pred, save_dir):
        """Generate and save normalized confusion matrix (percentages)"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(max(10, len(self.class_names)), max(8, len(self.class_names) * 0.8)))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Greens',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title(f'Normalized Confusion Matrix - {self.model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_confusion_matrix_normalized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved normalized confusion matrix to {save_path}")
        plt.close()

    def plot_metrics_per_class(self, y_true, y_pred, save_dir):
        """Plot precision, recall, and F1-score for each class"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.class_names))
        )

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(12, len(self.class_names) * 0.8), 6))
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')

        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Precision, Recall, and F1-Score by Class - {self.model_name}',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)

        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{self.model_name}_metrics_per_class.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics per class to {save_path}")
        plt.close()

    def plot_accuracy_by_class(self, y_true, y_pred, save_dir):
        """Plot accuracy for each class"""
        accuracies = []
        for i in range(len(self.class_names)):
            class_mask = (y_true == i)
            class_correct = np.sum((y_true[class_mask] == y_pred[class_mask]))
            class_total = np.sum(class_mask)
            accuracy = class_correct / class_total if class_total > 0 else 0
            accuracies.append(accuracy)

        plt.figure(figsize=(max(10, len(self.class_names) * 0.6), 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.class_names)))
        bars = plt.bar(self.class_names, accuracies, color=colors, edgecolor='black', linewidth=1.2)

        plt.xlabel('Classes', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'Accuracy by Class - {self.model_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{self.model_name}_accuracy_by_class.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved accuracy by class to {save_path}")
        plt.close()

    def save_classification_report(self, y_true, y_pred, save_dir):
        """Generate and save detailed classification report"""
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )

        # Save to text file
        report_path = os.path.join(save_dir, f'{self.model_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n")

            # Add overall accuracy
            overall_acc = accuracy_score(y_true, y_pred)
            f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
            f.write(f"Total samples: {len(y_true)}\n")

        print(f"✓ Saved classification report to {report_path}")
        print("\n" + "=" * 60)
        print(report)
        print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        print("=" * 60 + "\n")

    def plot_top_k_accuracy(self, y_true, probs, save_dir, max_k=5):
        """Plot top-k accuracy curve"""
        k_values = range(1, min(max_k + 1, len(self.class_names) + 1))
        top_k_accuracies = []

        for k in k_values:
            # Get top k predictions
            top_k_preds = np.argsort(probs, axis=1)[:, -k:]
            # Check if true label is in top k
            correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
            top_k_acc = np.mean(correct)
            top_k_accuracies.append(top_k_acc)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, top_k_accuracies, marker='o', linewidth=2, markersize=8, color='#E63946')
        plt.xlabel('k (Top-k predictions)', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'Top-k Accuracy - {self.model_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])

        # Add value labels
        for i, (k, acc) in enumerate(zip(k_values, top_k_accuracies)):
            plt.text(k, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{self.model_name}_top_k_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved top-k accuracy plot to {save_path}")
        plt.close()

    def plot_pr_curves(self, y_true, probs, save_dir):
        """
        Plot Precision-Recall curves with both one-vs-rest and micro/macro averaging
        """
        n_classes = len(self.class_names)

        # Binarize the labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute PR curve and average precision for each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], probs[:, i])

        # Compute micro-average PR curve
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), probs.ravel()
        )
        average_precision["micro"] = average_precision_score(y_true_bin, probs, average="micro")

        # Compute macro-average PR curve
        average_precision["macro"] = average_precision_score(y_true_bin, probs, average="macro")

        # Plot all PR curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        # Plot micro-average curve (thick line)
        plt.plot(recall["micro"], precision["micro"],
                label=f'Micro-average (AP = {average_precision["micro"]:.3f})',
                color='deeppink', linestyle='--', linewidth=3)

        # Plot individual class curves
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AP = {average_precision[i]:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title(f'Precision-Recall Curves (One-vs-Rest) - {self.model_name}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_pr_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Precision-Recall curves to {save_path}")
        plt.close()

        # Create a second plot with macro/micro averages only (cleaner view)
        plt.figure(figsize=(10, 6))
        plt.plot(recall["micro"], precision["micro"],
                label=f'Micro-average (AP = {average_precision["micro"]:.3f})',
                color='deeppink', linestyle='-', linewidth=3)

        # Calculate macro-average curve by averaging all class curves
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= n_classes

        plt.plot(all_recall, mean_precision,
                label=f'Macro-average (AP = {average_precision["macro"]:.3f})',
                color='navy', linestyle='-', linewidth=3)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title(f'Precision-Recall Curves (Averaged) - {self.model_name}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_pr_curves_averaged.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved averaged Precision-Recall curves to {save_path}")
        plt.close()

    def plot_training_history(self, history_path, save_dir):
        """
        Plot training and validation accuracy/loss over epochs
        """
        # Check if history file exists
        if not os.path.exists(history_path):
            print(f"⚠ Training history not found at {history_path}")
            print(f"  Skipping training history plots. Train the model first to generate history.")
            return

        # Load training history
        with open(history_path, 'r') as f:
            history = json.load(f)

        epochs = range(1, len(history['train_acc']) + 1)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot accuracy
        ax1.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
        ax1.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'Training and Validation Accuracy - {self.model_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Add best accuracy annotation
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        ax1.annotate(f'Best: {best_val_acc:.4f}\n(Epoch {best_epoch})',
                    xy=(best_epoch, best_val_acc),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2),
                    fontsize=10, fontweight='bold')

        # Plot loss
        ax2.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax2.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title(f'Training and Validation Loss - {self.model_name}', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Add best loss annotation
        best_val_loss = min(history['val_loss'])
        best_loss_epoch = history['val_loss'].index(best_val_loss) + 1
        ax2.annotate(f'Best: {best_val_loss:.4f}\n(Epoch {best_loss_epoch})',
                    xy=(best_loss_epoch, best_val_loss),
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=2),
                    fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{self.model_name}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history plot to {save_path}")
        plt.close()

    def plot_roc_curves(self, y_true, probs, save_dir):
        """
        Plot ROC curves with both one-vs-rest and micro/macro averaging
        """
        n_classes = len(self.class_names)

        # Binarize the labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and AUC
        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        # Plot micro-average curve (thick line)
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle='--', linewidth=3)

        # Plot individual class curves
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves (One-vs-Rest) - {self.model_name}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {save_path}")
        plt.close()

        # Create a second plot with macro/micro averages only (cleaner view)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle='-', linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle='-', linewidth=3)

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves (Averaged) - {self.model_name}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{self.model_name}_roc_curves_averaged.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved averaged ROC curves to {save_path}")
        plt.close()

    def evaluate(self, save_dir, history_path=None):
        """Run complete evaluation and generate all visualizations"""
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}\n")

        # Get predictions
        y_true, y_pred, probs = self.get_predictions()

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f"\n{'='*60}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"{'='*60}\n")

        # Generate all visualizations
        print("Generating visualizations...\n")

        # Plot training history if available
        if history_path:
            self.plot_training_history(history_path, save_dir)

        self.plot_confusion_matrix(y_true, y_pred, save_dir)
        self.plot_normalized_confusion_matrix(y_true, y_pred, save_dir)
        self.plot_metrics_per_class(y_true, y_pred, save_dir)
        self.plot_accuracy_by_class(y_true, y_pred, save_dir)
        self.plot_top_k_accuracy(y_true, probs, save_dir)
        self.plot_pr_curves(y_true, probs, save_dir)
        self.plot_roc_curves(y_true, probs, save_dir)
        self.save_classification_report(y_true, y_pred, save_dir)

        print(f"\n{'='*60}")
        print(f"Evaluation complete for {self.model_name}!")
        print(f"{'='*60}\n")


def main():
    """Main evaluation function for both models"""
    print("\n" + "="*60)
    print("Vehicle Tracking Model Evaluation")
    print("="*60)

    # Create evaluation output directory
    eval_dir = 'evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create subdirectories for each model
    color_eval_dir = os.path.join(eval_dir, f'color_classifier_{timestamp}')
    carname_eval_dir = os.path.join(eval_dir, f'carname_classifier_{timestamp}')
    os.makedirs(color_eval_dir, exist_ok=True)
    os.makedirs(carname_eval_dir, exist_ok=True)

    # Check if models exist
    if not os.path.exists('models/color_classifier.pth'):
        print("ERROR: Color classifier model not found at models/color_classifier.pth")
        print("Please train the model first using train_color_classifier.py")
        return

    if not os.path.exists('models/car_name_classifier.pth'):
        print("ERROR: Car name classifier model not found at models/car_name_classifier.pth")
        print("Please train the model first using train_carname_classifier.py")
        return

    try:
        # Evaluate Color Classifier
        print("\n" + "🎨 " * 20)
        color_evaluator = ModelEvaluator(
            model_path='models/color_classifier.pth',
            classes_json_path='models/color_classes.json',
            data_dir='data/training_data/color',
            model_name='color_classifier'
        )
        color_evaluator.evaluate(
            save_dir=color_eval_dir,
            history_path='models/color_classifier_history_new.json'
        )

        # Evaluate Car Name Classifier
        print("\n" + "🚗 " * 20)
        carname_evaluator = ModelEvaluator(
            model_path='models/car_name_classifier.pth',
            classes_json_path='models/carname_classes.json',
            data_dir='data/training_data/car_name',
            model_name='carname_classifier'
        )
        carname_evaluator.evaluate(
            save_dir=carname_eval_dir,
            history_path='models/car_name_classifier_history_new.json'
        )

        # Final summary
        print("\n" + "="*60)
        print("✓ ALL EVALUATIONS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to:")
        print(f"  - Color Classifier: {color_eval_dir}")
        print(f"  - Car Name Classifier: {carname_eval_dir}")
        print("\nGenerated files for each model:")
        print("  • training_history.png (accuracy & loss vs epoch)")
        print("  • confusion_matrix.png (raw counts)")
        print("  • confusion_matrix_normalized.png (percentages)")
        print("  • metrics_per_class.png (precision/recall/F1)")
        print("  • accuracy_by_class.png")
        print("  • top_k_accuracy.png")
        print("  • pr_curves.png (Precision-Recall curves - all classes)")
        print("  • pr_curves_averaged.png (Precision-Recall curves - micro/macro avg)")
        print("  • roc_curves.png (ROC curves - all classes)")
        print("  • roc_curves_averaged.png (ROC curves - micro/macro avg)")
        print("  • classification_report.txt")
        print("\n" + "="*60 + "\n")

    except Exception as e:
        print(f"\nERROR during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
