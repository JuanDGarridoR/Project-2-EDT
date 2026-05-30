# Project 2 - EDT

Machine Learning classification of Sentinel-2 satellite imagery using:

- Decision Tree (DT)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)
- Artificial Neural Network (ANN)

---

# Requirements

- Python 3.10 or higher
- VS Code (recommended)

---

# Project Structure

```text
Project-2-EDT/
│
├── data/
│   └── dataset_8bands.csv
│
├── outputs/
│
├── src/
│   └── train_models.py
│
├── requirements.txt
└── README.md
```

---

# Clone Repository

```bash
git clone <repository-url>
cd Project-2-EDT
```

---

# Create Virtual Environment

### Windows

```powershell
python -m venv venv
```

---

# Activate Virtual Environment

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see something like:

```text
(venv) PS C:\Project-2-EDT>
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Place the dataset inside the `data` folder.

Example:

```text
data/dataset_8bands.csv
```

The dataset must contain:

- Spectral bands (B2, B3, B4, B5, B6, B7, B8, B11)
- Coordinates (X, Y)
- Class label (`class`)

---

# Run Training

```bash
python src/train_models.py
```

---

# Generated Outputs

The project automatically generates:

- Confusion matrices (.png)
- Performance metrics (.csv)

Location:

```text
outputs/
```

Example:

```text
outputs/
├── ANN_cm.png
├── DecisionTree_cm.png
├── KNN_cm.png
├── NaiveBayes_cm.png
├── SVM_cm.png
└── metrics_summary.csv
```

---

# Deactivate Virtual Environment

```bash
deactivate
```

---

# Notes

- The `venv` folder is not included in the repository and must be created on each machine.
- If dependencies are missing, run:

```bash
pip install -r requirements.txt
```

- Ensure the dataset path matches the one configured in `src/train_models.py`.