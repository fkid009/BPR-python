# BPR-MF: Bayesian Personalized Ranking with Matrix Factorization

This repository implements the core idea of the [BPR (Bayesian Personalized Ranking)](https://arxiv.org/abs/1205.2618) paper by Rendle et al., using NumPy-based matrix factorization.

BPR is designed for learning personalized item rankings from implicit feedback (e.g., clicks, purchases), and optimizes pairwise ranking using stochastic gradient descent.

## How to Run

### 1. Prepare interaction data

The input file should be a `.csv` file with at least two columns user_id, item_id.

A sample file (`ex_data.csv`) is provided in the `data/` folder.

### 2. Train the model

Run `main.py` using the following command:

```bash
python main.py --data_file ex_data \
               --latent_dim 32 \
               --reg 0.01 \
               --learning_rate 0.01 \
               --epochs 100 \
               --num_samples 1000