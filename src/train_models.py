# ==========================================
# TRAIN_MODELS.PY
# ==========================================

# IMPORT LIBRARIES

import pandas as pd

# SPLIT + NORMALIZATION
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Support Vector Machine
from sklearn.svm import SVC

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Artificial Neural Network
from sklearn.neural_network import MLPClassifier

# ==========================================
# METRICS
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    cohen_kappa_score,
)

# ==========================================
# VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

import seaborn as sns

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...\n")

df = pd.read_csv(
    "data/dataset_8bands.tsv",
    sep="\t"
)

# ==========================================
# FEATURES AND LABELS
# ==========================================

X = df[
    [
        "BAND_1",
        "BAND_2",
        "BAND_3",
        "BAND_4",
        "BAND_5",
        "BAND_6",
        "BAND_7",
        "BAND_8",
    ]
]

y = df["class"]

# ==========================================
# NORMALIZATION
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# ==========================================
# CREATE MODELS
# ==========================================

models = {

    # ======================================
    # DECISION TREE
    # ======================================

    "DecisionTree": DecisionTreeClassifier(
        max_depth=10
    ),

    # ======================================
    # SUPPORT VECTOR MACHINE
    # ======================================

    "SVM": SVC(
        kernel="rbf"
    ),

    # ======================================
    # K-NEAREST NEIGHBOR
    # ======================================

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    # ======================================
    # NAIVE BAYES
    # ======================================

    "NaiveBayes": GaussianNB(),

    # ======================================
    # ARTIFICIAL NEURAL NETWORK
    # ======================================

    "ANN": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
    ),
}

# ==========================================
# RESULTS LIST
# ==========================================

results = []

# ==========================================
# TRAINING LOOP
# ==========================================

for name, model in models.items():

    print(f"\n==============================")
    print(f"TRAINING: {name}")
    print(f"==============================\n")

    # ======================================
    # TRAIN MODEL
    # ======================================

    model.fit(
        X_train,
        y_train
    )

    # ======================================
    # PREDICTIONS
    # ======================================

    y_pred = model.predict(
        X_test
    )

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    # ======================================
    # SAVE RESULTS
    # ======================================

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "F1-Score": f1,
            "Kappa": kappa,
        }
    )

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # ======================================
    # PLOT MATRIX
    # ======================================

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    # ======================================
    # SAVE IMAGE
    # ======================================

    plt.savefig(
        f"outputs/confusion_matrices/{name}_cm.png"
    )

    plt.close()

    # ======================================
    # PRINT REPORT
    # ======================================

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

# ==========================================
# FINAL RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n====================================")
print("FINAL RESULTS")
print("====================================\n")

print(results_df)

# ==========================================
# SAVE METRICS CSV
# ==========================================

results_df.to_csv(
    "outputs/metrics/model_metrics.csv",
    index=False
)

print("\nMetrics saved successfully.\n")