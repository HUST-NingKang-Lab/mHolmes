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

We recommend using **Conda** to manage the environment, similar to other bioinformatics tools. Installation takes approximately 10-20 minutes

```bash
# 1. Clone the repository
git clone [https://github.com/HUST-NingKang-Lab/mHolmes.git](https://github.com/HUST-NingKang-Lab/mHolmes.git)
cd mHolmes

# 2. Create Conda environment
conda create -n mHolmes python=3.10
conda activate mHolmes

# 3. Install dependencies
pip install -r requirements.txt

```
Or you can download the offline installation package from the GitHub release page and install MicroProphet using the following command:

```bash
pip install --prefer-binary mHolmes-1.1.0-py3-none-any.whl 

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

(Demo run time: 5-10 minutes)

```bash
mhm predict ./corpse_data/example.csv --export_path="./result"
```

#### 2. Workflow & Output
The tool performs a 5-fold cross-validation. For every fold, `mHolmes` generates specific output files in the export path.

**📂 Output Files Breakdown:**

The execution will generate **10 files in total** (2 types of files × 5 folds).

| File Type | Count | Description | Key Content |
| :--- | :--- | :--- | :--- |
| **Microbial Abundance Table** | 5 | Comparison of predicted vs. actual values for each fold. | • **`pred`**: Predicted abundance values<br>• **`label`**: Ground truth (actual) abundance |
| **Evaluation Metrics Table** | 1 | Statistical performance metrics for the model. | • **Mean MSE** (Mean Squared Error)<br>• **$R^2$** (Coefficient of Determination)<br>• **Pearson Correlation** |

> **📊 Logic:** The input data is split into 5 subsets. The model trains on 4 subsets and tests on the remaining one, repeating this process 5 times until each subset has served as the test set.

**📝 Output Examples & Column Definitions:**

Below are examples of the generated files based on actual outputs.

**A. Microbial Abundance Tables**
*Files: `fold_n_pred.csv` (Prediction) and `fold_n_label.csv` (Ground Truth)*

These files present the microbial abundance profile in a matrix format. Two corresponding files are generated for each fold: one containing the model's predictions (`pred`) and one containing the actual ground truth values (`label`).

| subject_id | time | Actinobacteria | Alphaproteobacteria | Bacilli | ... | others |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SHSU_2016_64 | 15 | 0.045452 | 0.022228 | 0.543553 | ... | 0.021950 |
| SHSU_2016_64 | 16 | 0.050529 | 0.024506 | 0.501234 | ... | 0.024189 |
| SHSU_2016_64 | 17 | 0.055563 | 0.029841 | 0.452508 | ... | 0.029516 |
| SHSU_2016_64 | 18 | 0.050999 | 0.024593 | 0.495412 | ... | 0.024274 |

**Column Definitions:**
* **`subject_id`**: The unique identifier for the subject or sample series (e.g., `SHSU_2016_64`).
* **`time`**: The specific time point or longitudinal sequence indicator for the sample.
* **Taxa Columns** (e.g., *Actinobacteria*, *Alphaproteobacteria*, *Bacilli*): The abundance values for each specific taxon in the sample. In the `pred` file, these are the model's estimates; in the `label` file, these are the actual biological measurements.

**B. Evaluation Metrics Table**
*File: `results.csv`*

This table summarizes the statistical performance metrics for each fold of the cross-validation.

| fold_num | MSE | R2 | Corr |
| :--- | :--- | :--- | :--- |
| 1 | 0.016687 | 0.350591 | 0.607779 |
| 2 | 0.014608 | 0.401445 | 0.634046 |
| 3 | 0.012432 | 0.545991 | 0.746215 |
| 4 | 0.014164 | 0.414067 | 0.665841 |
| 5 | 0.022978 | 0.209675 | 0.464288 |

**Column Definitions:**
* **`fold_num`**: The fold number (e.g., 1-5) corresponding to the cross-validation iteration.
* **`MSE`**: **Mean Squared Error**. The average squared difference between the predicted values and the actual values (Lower is better).
* **`R2`**: **Coefficient of Determination ($R^2$)**. Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable (Closer to 1 is better).
* **`Corr`**: **Pearson Correlation Coefficient**. Measures the linear correlation between the predicted and actual abundance values (Range: -1 to 1; closer to 1 indicates strong positive correlation).

### 🚀 Predict Future Microbial Time-Series Abundance (With Transfer Learning)

This mode leverages transfer learning to improve prediction accuracy on a specific target dataset. It utilizes a source dataset for pre-training and a subset of the target dataset for fine-tuning.

#### 1. Command Syntax

Run the transfer learning prediction using the `tlpredict` mode. You must provide both the source and target data files.

(Demo run time: 5-10 minutes)

```bash
mhm tlpredict ./corpse_data/source_hip.csv ./corpse_data/target_face.csv --export_path="./result"
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
* **1 Evaluation Metrics Tables:** Contains Mean MSE, $R^2$, and Pearson correlation for each fold.

### 🧠 SHAP Analysis (Feature Importance)

This module calculates SHAP (SHapley Additive exPlanations) values to interpret model predictions. It quantifies the contribution of each microbial feature to the model's output.

#### 1. Analysis Modes

You can perform SHAP analysis in two contexts: standard (non-transfer) and transfer learning.

**Option A: Standard SHAP Analysis (Non-Transfer)**
Calculates feature importance based on the standard training model.

(Demo run time: <5 minutes)

```bash
mhm shap ./corpse_data/example.csv --export_path="./result"
```

**Option B: Transfer Learning SHAP Analysis**
Calculates feature importance specifically for the target domain after fine-tuning.

(Demo run time: <5 minutes)

```bash
mhm tlshap ./corpse_data/source_hip.csv ./corpse_data/target_face.csv --export_path="./result"
```
#### 2. Output File Format

Both modes generate a CSV file (`foundation/transfer_fold_shap.csv`) summarizing the feature importance. Below is an example of the file content:

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

### 📉 MAE Validation (Forecast Quality Assessment)

This module evaluates the quality of the forecasted microbial data by using a proxy task: predicting the Postmortem Interval (PMI or "day") based on the abundance profiles. Lower MAE (Mean Absolute Error) indicates that the forecasted data preserves realistic biological signals.

#### 1. Analysis Modes

**Option A: Standard MAE Validation (Non-Transfer)**
Evaluates forecasting quality on a single dataset using 5-fold cross-validation.

(Demo run time: <5 minutes)

```bash
mhm mae ./corpse_data/example.csv --export_path="./result"
```

**Option B: Transfer Learning MAE Validation**
Evaluates forecasting quality on the target domain (e.g., hip) after pre-training on a source domain (e.g., face).

(Demo run time: <5 minutes)

```bash
mhm tlmae ./corpse_data/source_hip.csv ./corpse_data/target_face.csv --export_path="./result"
```

#### 2. Methodology

The validation process follows a rigorous 4-step workflow, repeated for each fold in a 5-fold cross-validation setup:

**Step 1: Pre-training (Source Domain)**
* **Model**: A `PatchTST` Transformer-based model is trained on the source domain.
* **Task**: Learn to predict the next 4 days of microbial abundance using the previous 7 days as context.
* **Goal**: Capture general microbiome temporal patterns.

**Step 2: Fine-Tuning (Target Domain)**
* **Action**: The pre-trained model is fine-tuned on a small subset (60%) of the target domain's training data.
* **Goal**: Adapt the model to the specific dynamics of the target site while retaining general knowledge.

**Step 3: Evaluation via "Day Prediction" Proxy Task**
To measure forecast realism, we use a separate `RandomForestRegressor` trained to predict the *day* from abundance data.
1.  **Train Day Predictor**: A Random Forest is trained on the target domain's training data.
2.  **Generate Forecasts**: The fine-tuned `PatchTST` predicts future abundances for the validation set.
3.  **Compare Scenarios**: The Day Predictor estimates the "day" for samples in two scenarios:
    * **`past-only` (Baseline)**: Uses only actual historical data (Days 1-7).
    * **`past+generated` (Evaluation)**: Uses historical data + the model's forecasted data (Days 8-11).

**Step 4: Aggregation**
We calculate the Mean Absolute Error (MAE) between the *predicted day* and the *true day*. Comparing the MAE of the forecasted period in `past+generated` against the baseline reveals the quality of the generated data.

#### 3. Output File Format

The tool generates a CSV file (e.g., `transfer_1_metrics.csv`) containing the per-day MAE metrics. Below is a sample:

| day | mae | scenario | fold |
| :--- | :--- | :--- | :--- |
| 1 | 9.144 | past-only | 1 |
| 2 | 9.183 | past-only | 1 |
| ... | ... | ... | ... |
| 11 | 1.746 | past-only | 2 |
| 12 | 1.392d | past+generate | 2 |
| 13 | 2.221 | past+generated | 3 |

**Column Descriptions:**

* **`day`**: The actual day (PMI) of the sample collection.
* **`mae`**: The Mean Absolute Error between the predicted day and the actual day. Lower values represent better performance.
* **`scenario`**: The evaluation context.
    * **`past-only`**: Baseline accuracy using only real, historical abundance data.
    * **`past+generated`**: Evaluation accuracy when including the model's forecasted abundance data. A low MAE here suggests the forecast is realistic enough to accurately guide PMI estimation.

### 🎭 Mask Mode (Robustness Evaluation)

This mode evaluates the model's robustness and generalization capabilities by introducing masking mechanisms. It is particularly useful for assessing performance on external validation datasets with potential irregularities or missing data.

#### 1. Command Syntax

Run the mask evaluation mode by providing source, target, and external validation datasets.

(Demo run time: 5-15 minutes)

```bash
mhm mask ./corpse_data/source_hip_keytaxa.csv ./corpse_data/target_face_keytaxa.csv ./corpse_data/val_head.csv --export_path="./result"
```

#### 2. Methodology: Masking Mechanisms

This mode employs two strategic masking techniques to enhance model training and handle time-series complexities:

**A. Random Input Masking**
* **What it is**: A data augmentation technique where random portions of the input time-series (`past_values`) are intentionally set to zero during training.
* **Why it's used**: Acts as regularization, forcing the model to learn robust, generalizable patterns from incomplete data rather than memorizing sequences.
* **Implementation**: Controlled via `do_mask_input: True` in the configuration.

**B. Padding Mask**
* **What it is**: A binary tensor (`past_observed_mask`) generated alongside input data to handle varying sequence lengths.
* **Why it's used**: It explicitly marks data points as real (`1`) or padding (`0`). The model's attention mechanism uses this to ignore padding, ensuring it doesn't skew loss calculations or predictions.
* **Implementation**: Automatically generated by the dataset loader during both training and inference.

#### 3. Output File Format

The tool generates a CSV file (e.g., `mask_val_head.csv`) summarizing the performance metrics on the external validation dataset.

**Example Output:**

| fold | MSE | R2 | Corr |
| :--- | :--- | :--- | :--- |
| 1 | 0.025747 | 0.600024 | 0.799217 |
| 2 | 0.024797 | 0.614769 | 0.808516 |
| 3 | 0.025238 | 0.607928 | 0.805296 |
| 4 | 0.025562 | 0.602888 | 0.802708 |
| 5 | 0.025619 | 0.602000 | 0.804590 |

**Column Descriptions:**

* **`fold`**: The cross-validation fold number (1-5). Each fold uses a unique subset of subjects for training vs. testing.
* **`mse`** (**Mean Squared Error**): The average squared difference between predicted and actual abundance. A lower value indicates better model fit. The value in this file is the *average MSE* across all sliding time windows for that specific fold.
* **`r2`** (**$R^2$ / Coefficient of Determination**): Indicates the proportion of the variance in the dependent variable (actual abundance) that is predictable from the independent variable (predicted abundance). Values closer to **1.0** indicate higher explanatory power.
* **`corr`** (**Pearson Correlation Coefficient**): Measures the linear relationship between the flattened arrays of predicted and actual values. Values closer to **1.0** indicate a strong positive linear relationship, meaning the model's predictions trend in the same direction as the actual values.

## Model Configuration

The core of the forecasting model is a `PatchTST` architecture configured with the following key parameters.

### Main Parameters

| Parameter               | Value  | Description                                                                                             |
| ----------------------- | :----: | ------------------------------------------------------------------------------------------------------- |
| `prediction_length`     |   3    | The model predicts 3 future time steps.                                                                 |
| `context_length`        |   5    | The model uses the past 5 time steps as input context. (Derived from `WINDOW_LEN` 8 - `prediction_length` 3) |
| `patch_length`          |   1    | Each patch consists of a single time step.                                                              |
| `patch_stride`          |   1    | Patches are extracted with a stride of 1, meaning no time steps are skipped.                            |
| `d_model`               |   64   | The embedding dimension size within the Transformer.                                                    |
| `nhead`                 |   4    | The number of attention heads in the multi-head attention mechanism.                                    |
| `num_hidden_layers`     |   3    | The number of Transformer encoder layers.                                                               |
| `dropout`               |  0.2   | Dropout rate applied for regularization.                                                                |
| `do_mask_input`         | `True` | Enables random input masking during training for data augmentation and regularization.                  |

### Training Parameters

-   **Optimizer**: AdamW (default in `transformers.Trainer`).
-   **Learning Rate Scheduler**: Linear decay.
-   **Epochs**: Up to 1000, with early stopping patience of 3 epochs based on validation loss.
-   **Batch Size**: 8.

## Maintainer

| Name | Email | Organization |
| :--- | :--- | :--- |
| **Jin Han** | `jinhan@hust.edu.cn` | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| **Kang Ning** | `ningkang@hust.edu.cn` | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |
