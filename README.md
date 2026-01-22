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

## 📂 Data Preparation

mHolmes requires input data in a specific CSV format. The file should contain sample identifiers, temporal metadata, and the relative abundances of 12 feature categories (11 specific bacterial classes + 1 aggregated "others" category).

### 1. Input Data Format (`input.csv`)
Please structure your input CSV file as shown below. Rows represent samples, and columns represent the top 11 dominant bacterial classes ranked by mean relative abundance.

| ID | day | Gammaproteobacteria | Bacilli | Bacteroidia | Clostridia | Actinobacteria | ... | Deinococci | others |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CMU_16_12_face | 21 | 0.033 | 0.151 | 0.040 | 0.102 | 0.633 | ... | 0.007 | 0.007 |
| CMU_16_12_face | 20 | 0.027 | 0.302 | 0.012 | 0.023 | 0.600 | ... | 0.015 | 0.015 |
| CMU_16_12_face | 19 | 0.032 | 0.474 | 0.007 | 0.009 | 0.427 | ... | 0.021 | 0.021 |

> **📝 Note:** Ensure all abundance values in a row sum to approximately **1.0 (100%)**.

### 2. Column Descriptions

* **`ID`**: Unique identifier for each sample (e.g., `CMU_16_12_face`).
* **`day`**: The Postmortem Interval (PMI) or time point associated with the sample (in days). This serves as the target variable for training or validation.
* **Feature Columns (11 Classes)**: Relative abundance of the top 11 dominant bacterial classes used by the mHolmes model:
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
