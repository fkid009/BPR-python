import numpy as np
import random

class BPR_MF:
    """
    A Bayesian Personalized Ranking (BPR) based Matrix Factorization (MF) model.
    It is optimized to provide personalized rankings for items using implicit feedback.
    """
    def __init__(
            self,
            num_users: int,
            num_items: int,
            latent_dim: int,
            reg: float,
            learning_rate: float
    ):
        """
        Initializes the BPR_MF model with random user and item latent factor matrices.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.K = latent_dim
        self.reg = reg
        self.lr = learning_rate

        self.U = np.random.normal(0, 0.1, (num_users, latent_dim)) # User latent factor matrix
        self.I = np.random.normal(0, 0.1, (num_items, latent_dim)) # Item latent factor matrix
        
    def set_train_data(self, user_train_dict: defaultdict[int, Set[int]]) -> None:
        """
        Sets the training data for the model.
        """
        self.user_train_dict = user_train_dict

    def set_test_data(self, user_test_dict: Dict[int, int]) -> None:
        """
        Sets the test data for model evaluation.
        """
        self.user_test_dict = user_test_dict

    def train(self, epochs: int, num_samples: int) -> None:
        """
        Trains the BPR-MF model using stochastic gradient descent (SGD).
        Prints the AUC performance at the end of each epoch.
        """
        for epoch in range(epochs):
            for _ in range(num_samples):
                if not self.user_train_dict:
                    continue
                u = random.choice(list(self.user_train_dict.keys()))
                
                if not self.user_train_dict[u]:
                    continue

                i = random.choice(list(self.user_train_dict[u]))
                j = random.randint(0, self.num_items - 1)
                # Sample a negative item j that is not in the user's training set and not the positive item i
                while j in self.user_train_dict[u] or j == i:
                    j = random.randint(0, self.num_items - 1)
                self._update(u, i, j) # Update model parameters
            auc = self._get_auc() # Calculate AUC for the current epoch
            print(f"Epoch {epoch + 1} done... AUC (LOO): {auc:.4f}")

    def _update(self, u: int, i: int, j: int) -> None:
        """
        Updates model parameters using stochastic gradient descent based on the BPR loss function.
        Performs an update for user u, positive item i, and negative item j.
        """
        u_vec = self.U[u]
        i_vec = self.I[i]
        j_vec = self.I[j]

        # x_uij = x_ui - x_uj = U[u] * I[i] - U[u] * I[j]
        x_uij = np.dot(u_vec, i_vec - j_vec)
        sigmoid = self._sigmoid(x_uij)
        
        # Calculate gradients for the BPR loss function (minimization objective)
        grad_u = (sigmoid - 1) * (i_vec - j_vec) + self.reg * u_vec
        grad_i = (sigmoid - 1) * u_vec + self.reg * i_vec
        grad_j = - (sigmoid - 1) * u_vec + self.reg * j_vec # The - sign is correct as it's -d(x_uj)/d(I[j])

        # Update parameters using gradient descent
        self.U[u] -= self.lr * grad_u
        self.I[i] -= self.lr * grad_i
        self.I[j] -= self.lr * grad_j

    def _sigmoid(self, x: float) -> float:
        """
        Computes the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def _get_auc(self, num_neg: int = 100) -> float:
        """
        Estimates and returns the AUC (Area Under the Curve) of the model.
        Compares one positive item with 'num_neg' randomly sampled negative items for each test user,
        and then averages the AUC across all users.
        """
        user_aucs: List[float] = []
        for u, i_pos_test in self.user_test_dict.items():
            if u not in self.user_train_dict: # Skip if user has no training data
                continue
            
            num_correct_predictions = 0
            
            # Set of items the user has interacted with (training + test)
            user_interacted_items = self.user_train_dict[u].union({i_pos_test})

            negative_samples_count = 0
            temp_negative_items: Set[int] = set()
            max_attempts = self.num_items * 2 # Max attempts to prevent infinite loops during sampling

            # Sample num_neg valid negative items
            while negative_samples_count < num_neg and len(temp_negative_items) < self.num_items - len(user_interacted_items) and max_attempts > 0:
                j_neg = random.randint(0, self.num_items - 1)
                if j_neg not in user_interacted_items and j_neg not in temp_negative_items:
                    temp_negative_items.add(j_neg)
                    negative_samples_count += 1
                max_attempts -= 1

            if not temp_negative_items: # If no valid negative samples could be found
                continue
            
            # Calculate score for the positive test item
            x_ui = np.dot(self.U[u], self.I[i_pos_test])
            
            # Compare scores with sampled negative items
            for j_neg in temp_negative_items:
                x_uj = np.dot(self.U[u], self.I[j_neg])
                if x_ui > x_uj: # Correct prediction if positive item score is higher
                    num_correct_predictions += 1
            
            # Calculate and add AUC for the current user
            if negative_samples_count > 0:
                user_aucs.append(num_correct_predictions / negative_samples_count)

        # Return the average AUC across all users
        return np.mean(user_aucs) if user_aucs else 0.0