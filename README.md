# Targeting AAV vectors to the CNS via de novo engineered capsid-receptor interactions

Now published in PLOS Biology! [https://doi.org/10.1371/journal.pbio.3002112](https://doi.org/10.1371/journal.pbio.3002112)

Code and documentation supporting "Targeting AAV vectors to the CNS via de novo engineered capsid-receptor interactions", including data, SVAE-based variant generation method, and figure-generation code.

## Contents

# Installation
Code is provided as a collection of [Jupyter Notebooks](https://jupyter.org/) and requires `python3.8` (TensorFlow 2.5 does not support newer interpreters). Dependency management is now handled by [uv](https://github.com/astral-sh/uv), which also creates a local virtual environment and lockfile.

```
# Install uv if needed (once per workstation)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Set up the project environment
cd AAV_capsid_receptor
uv sync  # creates .venv using the pinned dependencies from pyproject/uv.lock

# Launch tooling through uv (examples)
uv run jupyter lab  # start an interactive notebook server
uv run python some_script.py
```

> The legacy `requirements.txt` is still available for reference, but `uv sync` is now the authoritative way to install dependencies.

## 配置 ESM C（ESM Cambrian）

为了在本项目中复用 EvolutionaryScale 发布的「ESM Cambrian (ESM C)」蛋白表示模型，需要单独创建一个 Python 3.12+ 环境（`tensorflow==2.5` 仅支持 3.8，无法与 `esm` 的官方依赖共存）。以下步骤默认你已经安装了 [uv](https://github.com/astral-sh/uv)。

1. **创建独立解释器与虚拟环境**（目的：确保 `esm` 的 Python 3.12 运行时不影响现有 SVAE 环境）。

   ```bash
   uv toolchain install 3.12                # 安装 Python 3.12 解释器
   uv venv --python 3.12 .venv-esmc         # 在仓库根目录创建单独虚拟环境
   source .venv-esmc/bin/activate          # 激活环境（Windows 请使用 .venv-esmc\Scripts\activate）
   python -m pip install --upgrade pip
   ```

2. **安装 PyTorch 与 ESM C**（目的：拉取 GPU/CPU 依赖并从 GitHub 安装最新的 esm 仓库）。根据硬件选择合适的 PyTorch 轮子；以下示例为 CUDA 12.1，若仅 CPU 可改为官方 CPU 索引。

   ```bash
   # CUDA 12.1 示例，如需 CPU 版本请改用 https://download.pytorch.org/whl/cpu
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install "esm @ git+https://github.com/evolutionaryscale/esm.git@main"

   # （可选）安装 flash-attn 以启用 Flash Attention
   pip install flash-attn --no-build-isolation
   ```

3. **快速验证 & 生成嵌入**（目的：确认可以加载公开权重并获得表征，可将输出 `.npy` 文件供 SVAE/下游脚本使用）。示例脚本：

   ```bash
   python - <<'PY'
   from pathlib import Path

   import numpy as np
   from esm.models.esmc import ESMC
   from esm.sdk.api import ESMProtein, LogitsConfig

   client = ESMC.from_pretrained("esmc_300m").to("cuda")  # 无 GPU 可改为 "cpu"
   protein = ESMProtein(sequence="AAAAAGGTTTCCCAA" )       # 替换为你的氨基酸序列

   protein_tensor = client.encode(protein)
   logits_output = client.logits(
       protein_tensor,
       LogitsConfig(sequence=True, return_embeddings=True)
   )

   embeddings = logits_output.embeddings.cpu().numpy()
   Path("data/esmc_embeddings").mkdir(parents=True, exist_ok=True)
   np.save("data/esmc_embeddings/example_esmc_300m.npy", embeddings)
   print("Saved:", embeddings.shape)
   PY
   ```

将生成的嵌入（或其他派生特征）写入 CSV/NPY 后即可在原有 `python3.8` 环境里读取，无需在老环境中安装 `esm`。

## Experiment Tracking (Weights & Biases)

```
# Ensure dependencies are installed
uv sync

# Authenticate once (or set WANDB_API_KEY)
uv run wandb login

# Inside your training script / notebook
from svae.train import train

model, preds_df, run_dir = train(
    model,
    train_batches,
    val_batches,
    train_df,
    log_wandb=True,
    wandb_project="your-project-name",
    wandb_config={"notes": "optional"},
    wandb_watch=True,
)
```

When `log_wandb=True`, the trainer logs per-epoch losses, a preview of `preds_df`, and uploads `loss_log.csv`/`preds.csv` so the run is reproducible. Pass an existing `wandb_run` if you prefer to manage the run context manually (e.g., inside notebooks).


### SVAE model and training data

1. **Training data processing** - [`AAV_capsid_receptor/notebooks/pulldown_assay_data_processing.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/notebooks/pulldown_assay_data_processing.ipynb)
   
   - Starting from read counts, compute reads per million (RPM) and $\log_2$ enrichment for LY6A-Fc and LY6C1-Fc; export a CSV of `mean_RPM`, `cv_RPM` (coefficient of variation), and `log2enr` values for each of LY6A-Fc and LY6C1-Fc.

2. **SVAE model and variant generation** (for LY6C1-Fc) - [`AAV_capsid_receptor/notebooks/SVAE_variant_generation.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/notebooks/SVAE_variant_generation.ipynb)
    
    1. Starting from the CSV exported by `pulldown_assay_data_processing.ipynb`, format LY6C1-Fc data into TensorFlow-compatible training batches. 
    
    2. Initialize and train an SVAE model.
    
    3. Cluster and sample the trained SVAE model's latent space to generate novel variants.

### Paper figures

**Note:** all figure-generation notebooks assume figure data is contained in `AAV_capsid_receptor/data` (see [Data](#data) for more details).

1. Figure 1 and 1S (supplemental) panels - [`AAV_capsid_receptor/figures/fig1.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig1.ipynb)
2. Figure 2 and 2S (supplemental) panels - [`AAV_capsid_receptor/figures/fig2.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig2.ipynb)
3. Figure 3 panels - [`AAV_capsid_receptor/figures/fig3.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig3.ipynb)
4. Figure 4 and 4S (supplemental) panels - [`AAV_capsid_receptor/figures/fig4.ipynb`](https://github.com/vector-engineering/AAV_capsid_receptor/blob/main/figures/fig4.ipynb)


# Data

All relevant data is stored on Zenodo at [DOI 10.5281/zenodo.8222089](https://doi.org/10.5281/zenodo.8222089). Once downloaded, data files should be put into `AAV_capsid_receptor/data` - by default, figure-generation notebooks will search for data there.

