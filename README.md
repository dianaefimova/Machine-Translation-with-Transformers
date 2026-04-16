# Machine_Translation_with_Transformers
The project implements a sequence-to-sequence Neural Machine Translation (NMT) model using a Transformer architecture built from scratch in PyTorch. The model is created to translate from French to English using a small parallel corpus.

## Overview
The goal of this project is to build and train a Transformer-based translation model without relying on high-level libraries. The implementation includes:

- Custom tokenization pipeline
- Vocabulary construction with special tokens
- Positional encoding
- PyTorch nn.Transformer-based architecture
- Teacher forcing training loop
- Greedy decoding for inference

## Training setup
Loss function: CrossEntropyLoss
Optimizer: Adam
lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9
Batch size: 256
Epochs: 15
Train/Validation split: 80/20

## Training results
After 15 epochs:

- Train Loss: 0.0679
- Validation Loss: 0.0683

The model demonstrates effective optimization and stable training behavior.

## Model's results

## Limitations
While the model achieves low training and validation loss, it is important to note:

- Dataset is very small and synthetically limited
- Vocabulary coverage is restricted
- Model struggles with rare words and complex sentence structures


Therefore, despite strong numerical performance, real-world translation quality is not perfect.

## How to run
git clone https://github.com/dianaefimova/Machine_Translation_with_Transformers.git
cd Machine_Translation_with_Transformers
pip install -r requirements.txt
python model.py



