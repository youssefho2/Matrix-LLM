# Matrix LLM (85M Parameters)

This repository contains the training and generation code for a custom 85M parameter language model designed to run on Consumer/Free-tier GPUs, specifically optimized for Google Colab using a Tesla T4 GPU.

## Model Configuration
- **Vocabulary Size:** 32,000
- **Embedding Dimensions ($d_{model}$):** 640
- **Attention Heads:** 8
- **Layers:** 9
- **Block Size (Context Length):** 256
- **Parameters:** ~85.3 Million

## Repository Structure
- `model.py`: Core architecture of the model.
- `config.py`: Training hyperparameters, model size, and dataset configurations.
- `train.py`: Training pipeline with automated Google Colab + Google Drive checkpoint integration.
- `tokenizer.py`: BPE Tokenizer training and loading script.
- `dataset.py` / `dataloader.py`: Dataset utilities.
- `requirements.txt`: Project dependencies.

## Setup & Local Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd matrix-llm
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training on Google Colab (Tesla T4)

The training script automatically detects if it is running in Google Colab, mounts your Google Drive, and saves checkpoints to `/content/drive/MyDrive/matrix_llm_checkpoints` so your progress is never lost.

1. Upload the files to your Google Drive or clone the repository directly inside Colab.
2. Select **T4 GPU** as your Hardware Accelerator (**Runtime** -> **Change runtime type**).
3. Run the training script:
   ```python
   python train.py
   ```
