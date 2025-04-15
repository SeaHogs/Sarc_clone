# Sarcasm Detection and Explanation Evaluation

This repository contains the experimental code for CS4248 (Natural Language Processing) AY24/25 Semester 2 project on sarcasm detection, classification, and explanation evaluation. The project focuses on:

- Building and training BERT-based models for sarcasm detection in news headlines
- Implementing and comparing different explanation methods (LIME, Integrated Gradients) for model interpretability
- Evaluating the effectiveness of explanations in understanding model predictions

The codebase provides a framework for:
- Data preprocessing and model training
- Generating and visualizing model explanations
- Comparing different explanation methods
- Evaluating explanation quality and effectiveness

## Requirements
- python 3.12
- [uv](https://docs.astral.sh/uv/) (0.6.10)

```bash
# Install dependencies
uv install
source .venv/bin/activate  # or .venv\Scripts\activate in Windows

# Install PyTorch with CUDA support (adjust based on your system)
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Getting Started

1. Open `experiments.ipynb` in Jupyter Notebook
2. Run the cells to:
   - Train and evaluate the model
   - Generate LIME / Integrated Gradient explanations
   - Compare explanations

## Features
- BERT-based sarcasm detection
- LIME explanations for model predictions
- Sarcasm removal capabilities (experiment)

## Notes
- Pretrained models are cached in the `data/` directory
- Kaggle dataset is cached in the `data/` directory
- Training checkpoints are saved in the `checkpoints/` directory

### Side-track: LLM as an Expert

See appendix A of the project report for more details. Raw code is archieved in `appendix_laae` folder.

Main entrypoint: `appendix_laae/explain_model.ipynb`

