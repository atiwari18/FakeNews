from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.dataset import LIAR2Dataset

@dataclass
class DatasetBundle:
    train_texts: list[str]
    train_credit_vectors: np.ndarray
    train_labels: np.ndarray
    val_texts: list[str]
    val_credit_vectors: np.ndarray
    val_labels: np.ndarray
    test_texts: list[str]
    test_credit_vectors: np.ndarray
    test_labels: np.ndarray

@dataclass
class EvaluationResults:
    accuracy: float
    macro_f1: float
    roc_auc: float
    pr_auc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray

#Convert one dataset split into the three core pieces the logistic
#regression model needs:
#   1. text input
#   2. numeric credit vector
#   3. labels
def extract_split(dataset: LIAR2Dataset) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts = []
    credit_vectors = []
    labels = []

    for i in range(len(dataset)):
        sample = dataset[i]
        texts.append(sample["text"])
        credit_vectors.append(sample["credit_vector"].numpy())
        labels.append(sample["label"])

    return texts, np.array(credit_vectors), np.array(labels)

#Build baseline real-only dataset bundle
def build_real_only_bundle():
    train_ds = LIAR2Dataset(
        split="train",
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    val_ds = LIAR2Dataset(
        split="validation",
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    test_ds = LIAR2Dataset(
        split="test",
        task_type="classification",
        label_scheme="binary",
        include_metadata=True,
    )

    train_texts, train_credit_vectors, train_labels = extract_split(train_ds)
    val_texts, val_credit_vectors, val_labels = extract_split(val_ds)
    test_texts, test_credit_vectors, test_labels = extract_split(test_ds)

    return DatasetBundle(
        train_texts=train_texts,
        train_credit_vectors=train_credit_vectors,
        train_labels=train_labels,
        val_texts=val_texts,
        val_credit_vectors=val_credit_vectors,
        val_labels=val_labels,
        test_texts=test_texts,
        test_credit_vectors=test_credit_vectors,
        test_labels=test_labels,
    )

#Class for wrapping the logistic regression pipeline.
#   - TF-IDF for text
#   - scaling for numeric credit vector
#   - logistic regression classifier
class LogisticRegressionDetector:
    def __init__(self, max_features: int=20000, C: float=1.0, random_state: int = 42):
        #TF-IDF converts the text into numeric features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True,
        )

        #Credit vector is numeric and low-dimensional, so wescale it 
        self.credit_scaler = StandardScaler()

        #Model
        self.model = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="liblinear",
            random_state=random_state,
        )

    #We convert the numeric features to sparse format so they can be
    #stacked efficiently with the sparse text representation.
    def _build_features(self, texts: list[str], credit_vectors: np.ndarray, fit: bool=False):
        if fit:
            text_features = self.vectorizer.fit_transform(texts)
            scaled_credit = self.credit_scaler.fit_transform(credit_vectors)
        else:
            text_features = self.vectorizer.transform(texts)
            scaled_credit = self.credit_scaler.transform(credit_vectors)

        credit_sparse = csr_matrix(scaled_credit)
        return hstack([text_features, credit_sparse])
    
    def fit(self, texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray,):
        X = self._build_features(texts, credit_vectors, fit=True)
        self.model.fit(X, labels)
    
    #Predict hard class labels: fake (0) or real (1).
    def predict(self, texts: list[str], credit_vectors: np.ndarray) -> np.ndarray:
        X = self._build_features(texts, credit_vectors, fit=False)
        return self.model.predict(X)
    
    #Predict class probabilities for the positive class (REAL = 1).
    #These probabilities are what we use for ROC and precision-recall plots.
    def predict_proba(self, texts: list[str], credit_vectors: np.ndarray) -> np.ndarray:
        X = self._build_features(texts, credit_vectors, fit=False)
        return self.model.predict_proba(X)[:, 1]
    
#Evaluate a trained model on one dataset split and compute the metrics
def evaluate_model(model: LogisticRegressionDetector, texts: list[str], credit_vectors: np.ndarray, labels: np.ndarray):
    y_pred = model.predict(texts, credit_vectors)
    y_prob = model.predict_proba(texts, credit_vectors)

    accuracy = accuracy_score(labels, y_pred)
    macro_f1 = f1_score(labels, y_pred, average="macro")
    roc_auc = roc_auc_score(labels, y_prob)

    precision, recall, _ = precision_recall_curve(labels, y_prob)
    pr_auc = auc(recall, precision)

    return EvaluationResults(
        accuracy=accuracy,
        macro_f1=macro_f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        y_true=labels,
        y_pred=y_pred,
        y_prob=y_prob,
    )

def plot_roc_curve(results: EvaluationResults, save_path: Optional[str] = None):
    fpr, tpr, _ = roc_curve(results.y_true, results.y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {results.roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    #plt.show()


def plot_pr_curve(results: EvaluationResults, save_path: Optional[str] = None):
    precision, recall, _ = precision_recall_curve(results.y_true, results.y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Logistic Regression Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    #plt.show()


def plot_confusion_matrix(results: EvaluationResults, save_path: Optional[str] = None):
    cm = confusion_matrix(results.y_true, results.y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    #plt.show()


#Print a readable summary of the model performance.
def print_metrics(name: str, results: EvaluationResults) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {results.accuracy:.4f}")
    print(f"Macro F1 : {results.macro_f1:.4f}")
    print(f"ROC AUC  : {results.roc_auc:.4f}")
    print(f"PR AUC   : {results.pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(results.y_true, results.y_pred, target_names=["FAKE", "REAL"]))


#Main baseline run:
#   1. load real-only LIAR2 data
#   2. train logistic regression
#   3. evaluate on validation and test
#   4. save/display useful diagnostic plots
def main() -> None:
    data = build_real_only_bundle()

    model = LogisticRegressionDetector(
        max_features=20000,
        C=1.0,
        random_state=42,
    )

    model.fit(
        texts=data.train_texts,
        credit_vectors=data.train_credit_vectors,
        labels=data.train_labels,
    )

    val_results = evaluate_model(
        model,
        texts=data.val_texts,
        credit_vectors=data.val_credit_vectors,
        labels=data.val_labels,
    )

    test_results = evaluate_model(
        model,
        texts=data.test_texts,
        credit_vectors=data.test_credit_vectors,
        labels=data.test_labels,
    )

    print_metrics("Validation Results", val_results)
    print_metrics("Test Results", test_results)

    plot_roc_curve(test_results, save_path="logreg_test_roc.png")
    plot_pr_curve(test_results, save_path="logreg_test_pr.png")
    plot_confusion_matrix(test_results, save_path="logreg_test_confusion.png")

if __name__ == "__main__":
    main()