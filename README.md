# Music Genre Classification with Vision Transformer (ViT)

This repository provides a Jupyter notebook (`Vit_Genre_Classification.ipynb`) showcasing the fine-tuning of a Vision Transformer (ViT) model for music genre classification. The demonstration uses the egtzan_plus dataset available from Hugging Face.

## Dataset

The dataset used in this project is egtzan_plus (available on Hugging Face Datasets platform). The egtzan_plus dataset consists of mel spectrograms generated with librosa, serving as valuable representations for audio data.

- **Dataset Name:** egtzan_plus
- **Hugging Face Datasets Link:** [egtzan_plus](https://huggingface.co/datasets/ghermoso/egtzan_plus)

## Cuda

The code is set to run on GPU or CPU. If you want to run it on a GPU make sur you have CUDA installed, and add the link to the PyTorch wheel that matches your CUDA version. For example, for CUDA 12.1 add `--extra-index-url https://download.pytorch.org/whl/cu121` to the `requirements.txt` file.

## Usage

To run the notebook and fine-tune the ViT model on the music genre classification task:

1. Clone this repository:

    ```bash
    git clone https://github.com/GuillaumeHERMOSO/Vision-Transformer-Music-Genre-Classification.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Open and run the `Vit_Genre_Classification.ipynb` Jupyter notebook. Follow the step-by-step instructions provided in the notebook for fine-tuning the ViT model on the egtzan_plus dataset.

