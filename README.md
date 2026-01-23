# mHolmes
A Transformer-based digital twin for forensic microbiology. Powered by transfer learning, it solves sparse sampling limits for high-precision PMI estimation (MAE &lt; ±2 days). Enables daily microbial prediction and cross-anatomical matching with SHAP interpretability.
![mHolmes Overview](mHolmes_overview.png)

### Key Features
* **🧩 Cross-Anatomical Transfer**: Successfully transfers knowledge from stable sites (Hip) to environmentally exposed sites (Face).
* **🤖 Digital Twin Architecture**: Utilizes Transformer-based time-series forecasting (PatchTST) to model microbial succession at daily resolution.
* **🔍 Interpretability**: Integrated **SHAP (Shapley Additive exPlanations)** analysis to identify key necro-microbiome biomarkers (e.g., *Gammaproteobacteria* dynamics).
* **📊 Robustness**: Validated on 34 human cadavers and externally tested on cross-species (mouse) datasets.

---

## 🛠️ System Requirements

mHolmes is built on Python and PyTorch. We recommend running on a Linux environment with CUDA support.

* **OS**: Linux (Ubuntu 20.04+ recommended) or Windows (WSL2)
* **Python**: 3.8+
* **GPU**: NVIDIA GPU (RTX 3090 or higher recommended for training)

## 📦 Installation

We recommend using **Conda** to manage the environment, similar to other bioinformatics tools.

```bash
# 1. Clone the repository
git clone [https://github.com/HUST-NingKang-Lab/mHolmes.git](https://github.com/HUST-NingKang-Lab/mHolmes.git)
cd mHolmes

# 2. Create Conda environment
conda create -n mHolmes python=3.8
conda activate mHolmes

# 3. Install dependencies
pip install -r requirements.txt

```
Or you can download the offline installation package from the GitHub release page and install MicroProphet using the following command:

```bash
pip install mHolmes-1.1.0-py3-none-any.whl

```

## 📖Usage

mHolmes is accessed via a command-line interface (CLI) with various modes. The general syntax is:

```bash
mhm <mode> [options]
```


### 📂 Data Preparation
#### 1. Input Data Format (`input.csv`)
Please structure your input CSV file as shown below. 

| ID | day | Gammaproteobacteria | Bacilli | Bacteroidia | Clostridia | Actinobacteria | ... | Deinococci | others |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CMU_16_12_face | 21 | 0.033 | 0.151 | 0.040 | 0.102 | 0.633 | ... | 0.007 | 0.007 |
| CMU_16_12_face | 20 | 0.027 | 0.302 | 0.012 | 0.023 | 0.600 | ... | 0.015 | 0.015 |
| CMU_16_12_face | 19 | 0.032 | 0.474 | 0.007 | 0.009 | 0.427 | ... | 0.021 | 0.021 |

> **📝 Note:** Ensure all abundance values in a row sum to approximately **1.0 (100%)**.

#### 2. Column Descriptions

* **`ID`**: Unique identifier for each sample (e.g., `CMU_16_12_face`).
* **`day`**: The Postmortem Interval (PMI) or time point associated with the sample (in days). This serves as the target variable for training or validation.
* **Feature Columns (11 Classes)**: Relative abundance of the top 11 dominant bacterial classes used by the mHolmes model (values are numerical):
    * *Gammaproteobacteria*
    * *Bacilli*
    * *Bacteroidia*
    * *Clostridia*
    * *Actinobacteria*
    * *Alphaproteobacteria*
    * *Erysipelotrichia*
    * *Fusobacteriia*
    * *Thermoleophilia*
    * *Negativicutes*
    * *Deinococci*
* **`others`**: Aggregated relative abundance of all remaining low-prevalence taxa not listed above.

### 🔮 Predict Future Microbial Time-Series Abundance (No Transfer Learning)

This mode allows you to predict microbial abundance using a 5-fold cross-validation strategy. The model automatically partitions the input data into training and testing sets to validate performance without requiring external transfer learning datasets.

#### 1. Command Syntax

Run the prediction using the `predict` mode. You must specify your input CSV and the desired output directory.

```bash
mhm predict ./data/example.csv --export_path "./result"
```

#### 2. Workflow & Output
The tool performs a 5-fold cross-validation. For every fold, `mHolmes` generates specific output files in the export path.

**📂 Output Files Breakdown:**

The execution will generate **10 files in total** (2 types of files × 5 folds).

| File Type | Count | Description | Key Content |
| :--- | :--- | :--- | :--- |
| **Microbial Abundance Table** | 5 | Comparison of predicted vs. actual values for each fold. | • **`pred`**: Predicted abundance values<br>• **`label`**: Ground truth (actual) abundance |
| **Evaluation Metrics Table** | 5 | Statistical performance metrics for the model. | • **Mean MSE** (Mean Squared Error)<br>• **$R^2$** (Coefficient of Determination)<br>• **Pearson Correlation** |

> **📊 Logic:** The input data is split into 5 subsets. The model trains on 4 subsets and tests on the remaining one, repeating this process 5 times until each subset has served as the test set.

### 🚀 Predict Future Microbial Time-Series Abundance (With Transfer Learning)

This mode leverages transfer learning to improve prediction accuracy on a specific target dataset. It utilizes a source dataset for pre-training and a subset of the target dataset for fine-tuning.

#### 1. Command Syntax

Run the transfer learning prediction using the `tlpredict` mode. You must provide both the source and target data files.

```bash
mhm tlpredict ./data/source.csv ./data/target.csv --export_path "./result"
```

#### 2. Workflow & Methodology

The model employs a **Pre-training + Fine-tuning** strategy integrated with 5-fold cross-validation on the target data.

* **Step 1: Data Inputs**
    * **Source Data** (`source.csv`): Used entirely for model **pre-training**. Must follow the standard `input.csv` format.
    * **Target Data** (`target.csv`): Used for **fine-tuning** and **testing**. Must follow the standard `input.csv` format.

* **Step 2: Split Strategy**
    The target data is divided into 5 folds. For each fold:
    1.  **Pre-training:** The model is initialized using **all** source data.
    2.  **Fine-tuning:** From the training portion of the target data, **60% (0.6) of the subjects** are randomly selected to form the fine-tuning dataset.
    3.  **Testing:** The model is evaluated on the remaining held-out test fold of the target data.

#### 3. Output Files

The output format is identical to the non-transfer learning mode. The execution will generate **10 files in total** in the export path.

* **5 Microbial Abundance Tables:** Contains `pred` (prediction) and `label` (ground truth) columns for each fold.
* **5 Evaluation Metrics Tables:** Contains Mean MSE, $R^2$, and Pearson correlation for each fold.

### 🧠 SHAP Analysis (Feature Importance)

This module calculates SHAP (SHapley Additive exPlanations) values to interpret model predictions. It quantifies the contribution of each microbial feature to the model's output.

#### 1. Analysis Modes

You can perform SHAP analysis in two contexts: standard (non-transfer) and transfer learning.

**Option A: Standard SHAP Analysis (Non-Transfer)**
Calculates feature importance based on the standard training model.

```bash
mhm shap ./data/example.csv --export_path "./result"
```

**Option B: Transfer Learning SHAP Analysis**
Calculates feature importance specifically for the target domain after fine-tuning.

```bash
mhm tlshap ./data/source.csv ./data/target.csv --export_path "./result"
```
#### 2. Output File Format

Both modes generate a CSV file (`shap_feature_importance.csv`) summarizing the feature importance. Below is an example of the file content:

| Feature | SHAP Importance |
| :--- | :--- |
| Gammaproteobacteria | 6.194648e-06 |
| others | 9.256655e-07 |
| Clostridia | 5.751579e-07 |
| Fusobacteriia | 4.939387e-07 |
| Thermoleophilia | 1.520035e-07 |
| ... | ... |

**Column Descriptions:**

* **`Feature`**: The name of the microbial feature (e.g., *Gammaproteobacteria*, *Bacilli*).
* **`SHAP Importance`**: The average SHAP value associated with the feature.
    * **Magnitude**: Indicates the strength of the feature's influence on the predicted PMI.
    * **Sign (+/-)**: Indicates the direction of the effect (positive values increase the predicted PMI, negative values decrease it).
    * *> For transfer learning, these values reflect feature importance within the **target domain**.*
