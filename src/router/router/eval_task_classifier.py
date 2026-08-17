import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from tqdm import tqdm

from src.router.router.classification import TaskClassifier


def print_stats(accuracy, precision, recall, f1, roc_auc, pr_auc):
    table = PrettyTable(float_format="4.3f")

    table.add_column(
        "Metric", ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
    )
    table.add_column(
        "Score",
        [
            f"{accuracy:.3f}",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{roc_auc:.3f}",
            f"{pr_auc:.3f}",
        ],
    )
    print(table)


datasets = [
    ("squad", "qa"),
    ("cnn_dailymail", "summarization"),
    ("xsum", "summarization"),
    ("glue_mnli", "classification"),
    ("glue_qqp", "classification"),
    ("glue_sst2", "classification"),
    ("mbpp", "coding"),
    ("gsm8k", "reasoning"),
    ("natural_questions", "qa"),
]

number_of_evaluation_samples = 2000

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

classifier = TaskClassifier(model, "../../hf_datasets/")
classifier.train(datasets)
tasks = classifier.tasks


base_path = "../../hf_datasets/difficulty"
dfs = []
for ds in datasets:
    df = pd.read_pickle(f"{base_path}/prepared-{ds[0]}.pkl")
    dfs.append(df.sample(n=500))


concat_df = pd.concat(dfs)
samples_per_task = number_of_evaluation_samples / len(tasks)

print(f"Tasks: {tasks}")

df_samples = []
for task in tasks:
    df_samples.append(
        concat_df[concat_df["task"] == task].sample(n=int(samples_per_task))
    )

rows = pd.concat(df_samples)

preds = []
probs = []
for index, row in tqdm(rows.iterrows(), "Making predictions...", total=len(rows)):
    pred, prob = classifier.predict(row["prompt"])
    preds.append(pred)
    probs.append(prob)

le = LabelEncoder()
y_true = rows["task"]
y_pred = preds

le.fit(y_true)

y_true = le.transform(y_true)
y_pred = le.transform(y_pred)

y_prob = np.vstack(probs)


# ---- Basic metrics ----
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

# ---- Confusion matrix ----
cm = confusion_matrix(y_true, y_pred)

# ---- ROC-AUC ----
roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

# ---- Precision-Recall AUC ----
pr_auc = average_precision_score(y_true, y_prob, average="macro")

print_stats(accuracy, precision, recall, f1, roc_auc, pr_auc)

n_classes = y_prob.shape[1]

# convert labels to one-hot
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

# Create the figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


# -------------------- ROC curves --------------------
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{tasks[i]} (AUC={roc_auc:.2f})")

axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right", fontsize=8)

# ---------------- Precision-Recall curves ----------------
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
    axes[1].plot(recall, precision, label=f"{tasks[i]} (AP={ap:.2f})")

axes[1].set_title("Precision-Recall Curves")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="lower left", fontsize=8)

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=tasks,
    yticklabels=tasks,
    ax=axes[2],
    cmap="Blues",
)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted Class")
axes[2].set_ylabel("True Class")

plt.tight_layout()
plt.show()
