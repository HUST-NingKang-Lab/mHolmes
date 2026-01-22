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

