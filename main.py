import os
import argparse

from utils import create_leave_one_out_split, txt2dict
from path import Path
from model import BPR_MF

parser = argparse.ArgumentParser(description="Run BPR Matrix Factorization model.")
parser.add_argument('--data_file', type=str, default='ex_data',
                    help='Path to the interaction data CSV file (e.g., user_id,item_id).')
# BPR_MF class arguments
parser.add_argument('--latent_dim', type=int, default=32,
                    help='Dimension of the latent factors (K).')
parser.add_argument('--reg', type=float, default=0.01,
                    help='Regularization strength (lambda).')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Learning rate for SGD.')
# Training arguments
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='Number of samples (updates) per epoch.')
args = parser.parse_args()



if __name__ == "__main__":
    # --- Data Loading ---
    file_name = f"{args.data_file}.csv"
    data_path = Path.DATA_PATH
    file_path = os.path.join(data_path, file_name)

    try:
        data, user_num, item_num = txt2dict(file_path)

        train_dict, test_dict = create_leave_one_out_split(data)

        print(f"Successfully loaded data from '{file_name}' using load_interactions_to_dict.")
        print(f"Inferred number of users: {user_num}")
        print(f"Inferred number of items: {item_num}")

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please ensure the CSV file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading or processing data: {e}")
        exit()

    # --- Model Initializing ---
    model = BPR_MF(
        num_users = user_num,
        num_items = item_num,
        latent_dim = args.latent_dim,
        reg = args.reg,
        learning_rate = args.learning_rate
    )

    model.set_train_data(train_dict)
    model.set_test_data(test_dict)

    # --- Model Training ---
    print("\nStarting BPR-MF model training...")
    model.train(epochs=args.epochs, num_samples=args.num_samples)
    print("\nTraining complete.")