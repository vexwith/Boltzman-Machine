import torch
from PIL import Image
import numpy as np

class Network:

    def __init__(self, num_values):
        self.num_values = num_values
        self.X = torch.ones(num_values, 1)
        self.X[torch.rand(num_values, 1) > 0.5] = -1

        # self.W = torch.zeros(num_values, num_values)
        self.W = torch.ones(num_values, num_values)
        self.W[torch.rand(num_values, num_values) > 0.5] = -1
        self.W.fill_diagonal_(0)

    def embed(self, png_path):
        img = Image.open(png_path).convert('L')  # 'L' mode for grayscale
        img_array = np.array(img)  # Shape: (H, W), dtype: uint8

        # Convert to PyTorch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        binary_tensor = torch.where(img_tensor == 1, -1.0, 1.0)

        # Ensure shapes match (safety check)
        assert binary_tensor.shape == self.X.shape, \
            f"Shape mismatch: expected {self.X.shape}, got {binary_tensor.shape}"

        # Copy values into self.X (in-place to avoid reallocation)
        self.X.copy_(binary_tensor)

    def update_weights(self):
        for i, row in enumerate(self.W):
            for j, col in enumerate(row):
                if i == j: continue
                self.W[i][j] += self.X[i] * self.X[j]

    def get_neuron_neighbour_energy(self, idx):
        return self.W[idx] @ self.X

    def get_system_energy(self, wei): # we want it to be the highest
        return ((wei @ self.X).T @ self.X).item() # sum of every (weight_ij * x_i * x_j)

    def generate(self, max_iters):
        wei = self.W.triu()
        counter = 0
        best_energy = self.get_system_energy(wei)
        while counter < max_iters:
            indices = np.random.permutation(self.num_values)
            prev_energy = self.get_system_energy(wei)
            cur_energy = 0.0
            for idx in indices:
                nn_energy = self.get_neuron_neighbour_energy(idx)
                self.X[idx] = 1 if nn_energy > 0 else -1

                cur_energy = self.get_system_energy(wei)
                if cur_energy > best_energy:
                    best_energy = cur_energy

                counter += 1
                if counter % 100 == 0:
                    print(f"system energy: {cur_energy}  counter: {counter}  best_se: {best_energy}  max_se: {self.num_values * (self.num_values - 1) / 2}")

                if counter >= max_iters: break
            if prev_energy - cur_energy == 0: break


nn = Network(100)
print(torch.count_nonzero(nn.X == -1))
# print(nn.W)
print(torch.count_nonzero(nn.W == -1) / torch.count_nonzero(nn.W == 1))
print(nn.get_system_energy(nn.W.triu()))
nn.generate(1000)

