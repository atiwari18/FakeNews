from __future__ import annotations
import sys
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    accuracy_score, 
    auc, 
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_recall_curve, 
    roc_auc_score, 
    roc_curve
)

from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    set_seed
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.roberta_data import RobertaDataConfig, RobertaDatasetBuilder

@dataclass
class EvaluationResults:
    accuracy: float
    macro_f1: float
    roc_auc: float
    pr_auc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray

def softmax_numpy(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def extract_logits(predictions):
    #Check if predictions is tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = np.asarray(predictions)

    #If there is an extra singleton dimension squeeze it out
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = np.squeeze(logits, axis=1)

    return logits

#Compute metrics
def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]):
    predictions, labels = eval_pred

    logits = extract_logits(predictions)
    
    #convert logits into probabilities for the positive class.
    prob_matrix= softmax_numpy(logits)
    probs = prob_matrix[:, 1]
    preds = np.argmax(logits, axis=1)

    print("logits shape:", logits.shape)
    print("prob_matrix shape:", prob_matrix.shape)
    print("probs shape:", probs.shape)
    print("labels shape:", np.asarray(labels).shape)

    #Metrics
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

#Allows us to resuse the same plotting/reporting functions across
#validation, test and future retraining rounds. 
def evaluate_split(trainer: Trainer, dataset):
    predictions = trainer.predict(dataset)
    logits = extract_logits(predictions.predictions)
    labels = predictions.label_ids

    prob_matrix = softmax_numpy(logits)
    probs = prob_matrix[:, 1]
    preds = np.argmax(logits, axis=1)

    #Computing metrics
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    return EvaluationResults(
        accuracy=accuracy,
        macro_f1=macro_f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        y_true=labels,
        y_pred=preds,
        y_prob=probs,
    )

def plot_training_history(trainer: Trainer, save_path: Optional[str] = None):
    log_history = trainer.state.log_history

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))

    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss")

    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("RoBERTa Training History")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_roc_curve(results: EvaluationResults, save_path: Optional[str] = None):
    fpr, tpr, _ = roc_curve(results.y_true, results.y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {results.roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RoBERTa ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_pr_curve(results: EvaluationResults, save_path: Optional[str] = None):
    precision, recall, _ = precision_recall_curve(results.y_true, results.y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RoBERTa Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_confusion_matrix(results: EvaluationResults, save_path: Optional[str] = None):
    cm = confusion_matrix(results.y_true, results.y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("RoBERTa Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


def print_metrics(name: str, results: EvaluationResults) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {results.accuracy:.4f}")
    print(f"Macro F1 : {results.macro_f1:.4f}")
    print(f"ROC AUC  : {results.roc_auc:.4f}")
    print(f"PR AUC   : {results.pr_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            results.y_true,
            results.y_pred,
            target_names=["FAKE", "REAL"],
        )
    )


#Keeps the training configuration in one place so it is easier to tune
#and easier to reuse for future retraining with synthetic augmentation.
def build_training_arguments(output_dir: str):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        #logging_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )


def main():
    set_seed(42)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("Training will run on CPU.")

    # Build tokenized train/validation/test splits for RoBERTa.
    data_config = RobertaDataConfig(
        model_name="roberta-base",
        max_length=512,
        label_scheme="binary",
        include_metadata=True,
    )

    builder = RobertaDatasetBuilder(data_config)
    tokenized_datasets = builder.build_tokenized_dataset_dict()
    data_collator = builder.get_data_collator()

    # Standard binary classification RoBERTa head.
    # This baseline uses the combined text field only.
    model = AutoModelForSequenceClassification.from_pretrained(
        data_config.model_name,
        num_labels=2,
    )

    training_args = build_training_arguments(output_dir="roberta_baseline_output")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=builder.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on validation and locked test splits.
    validation_results = evaluate_split(trainer, tokenized_datasets["validation"])
    test_results = evaluate_split(trainer, tokenized_datasets["test"])

    print_metrics("Validation Results", validation_results)
    print_metrics("Test Results", test_results)

    # Plot the optimization behavior and final test diagnostics.
    plot_training_history(trainer, save_path="roberta_training_history.png")
    plot_roc_curve(test_results, save_path="roberta_test_roc.png")
    plot_pr_curve(test_results, save_path="roberta_test_pr.png")
    plot_confusion_matrix(test_results, save_path="roberta_test_confusion.png")


if __name__ == "__main__":
    main()