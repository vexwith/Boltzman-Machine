import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# '''
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(mnist_pytorch, batch_size=1, shuffle=False)

class Network:

    def __init__(self, num_values):
        self.num_values = num_values
        self.num_minimas = 0
        self.X = torch.ones(num_values, 1, device=device)
        self.W = torch.zeros(num_values, num_values, device=device)

    def random_values(self):
        self.X = torch.ones(self.num_values, 1, device=device)
        self.X[torch.rand(self.num_values, 1, device=device) > 0.5] = -1

    def random_weights(self):
        W = torch.ones(self.num_values, self.num_values, device=device)
        W[torch.rand(self.num_values, self.num_values, device=device) > 0.5] = -1
        W.fill_diagonal_(0)
        W = W.tril() + W.T.triu()
        return W

    def embed_img(self, num_images=1, num_labels=10, use_all=False, info=False):
        labels = dict()
        max_for_label = num_images // num_labels
        for i, (image, label) in enumerate(train_loader_pytorch):
            label_name = label.item()
            if not use_all and i >= num_images: break
            # Making sure the number of pictures in each label is the same
            if use_all or labels.get(label_name, 0) < max_for_label:
                labels[label_name] = labels.get(label_name, 0) + 1
                # Image embedding
                self.X = image.squeeze(0).flatten().to(device)
                self.X = torch.where(self.X > 0, torch.tensor(1, device=device), torch.tensor(-1, device=device))
                if info: print(f"idx: {i}  label: {label_name}")
                self.update_weights()
            else: # If skipped due to repeating we need to take more images into account
                num_images += 1
        print(labels)

    def tensor_to_img(self):
        image_tensor = self.X.reshape(28, 28)
        image_tensor = torch.where(image_tensor == -1, 0, 255)  # scaling
        image_np = image_tensor.cpu().numpy().astype('uint8')
        return image_np

    def decode_img(self):
        image_np = self.tensor_to_img()

        plt.imshow(image_np, cmap='gray')
        plt.show()

    def init_plot(self):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        image_np = self.tensor_to_img()

        img_display = ax.imshow(image_np, cmap='gray')
        return fig, img_display

    def update_plot(self, img_display, delay_time=0.5):
        # Update plot in place
        new_image = self.tensor_to_img()
        img_display.set_data(new_image)
        plt.gcf().canvas.draw_idle()  # redraw
        plt.pause(delay_time)

    def embed_test(self, png_path):
        img = Image.open(png_path).convert('L')  # 'L' mode for grayscale
        img_array = np.array(img)  # Shape: (H, W), dtype: uint8, normalize to [0, 1]

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_array).float().to(device)
        binary_tensor = torch.where(img_tensor == 0, -1.0, 1.0)
        binary_tensor = binary_tensor.reshape(self.num_values, 1)

        # Ensure shapes match (safety check)
        assert binary_tensor.shape == self.X.shape, \
            f"Shape mismatch: expected {self.X.shape}, got {binary_tensor.shape}"

        # Copy values into self.X (in-place to avoid reallocation)
        self.X.copy_(binary_tensor)

    def update_weights(self):
        # Makes [num_values x num_values] matrix of every x_i * x_j
        self.W += torch.outer(self.X.squeeze(), self.X.squeeze())
        self.W.fill_diagonal_(0)
        self.num_minimas += 1

    def get_neuron_neighbour_energy(self, idx):
        return self.W[idx] @ self.X

    def get_system_energy(self, wei): # we want it to be the highest
        return ((wei @ self.X).T @ self.X).item() # sum of every (weight_ij * x_i * x_j)

    def generate(self, max_iters=1000, find_minimum=False, info=True):
        if info:
            fig, img_display = self.init_plot()

        wei = self.W.triu().sign() # to not repeat weights use triu only and normalize to -1, 0, 1
        counter = 0
        temperature = 5
        tcr = 0.005 # temperature change rate
        best_energy = self.get_system_energy(wei)
        while find_minimum or counter < max_iters:
            indices = np.random.permutation(self.num_values)
            prev_energy = best_energy
            for idx in indices:
                nn_energy = self.get_neuron_neighbour_energy(idx)
                neuron_energy = nn_energy * self.X[idx] # We need to take neuron value into account
                probability = 1/(1 + torch.exp(-neuron_energy/temperature)) # sigmoid - probability of saying the same
                r = torch.rand(1, device=device)
                if r > probability:
                    self.X[idx] = -self.X[idx]

                cur_energy = self.get_system_energy(wei)
                if cur_energy > best_energy:
                    best_energy = cur_energy

                counter += 1
                temperature = max(0, temperature - tcr)
                if info:
                    if counter % 100 == 0:
                        print(f"system energy: {cur_energy}  counter: {counter}  best_se: {best_energy}  "
                              f"max_se: {self.num_values * (self.num_values - 1) / 2}")
                    if counter % 10 == 0:
                        self.update_plot(img_display, 0.03)

                # Rate of checking for no energy change
                if counter % (self.num_values // 2) == 0 \
                    and prev_energy == best_energy: break
                if not find_minimum and counter >= max_iters: break # break earlier
            if prev_energy == best_energy: break
        if info:
            plt.close(fig)
            plt.ioff()

    def save_model(self, filepath):
        torch.save({
            'W': self.W,
            'num_minimas': self.num_minimas,
            'num_values': self.num_values
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model = cls(checkpoint['num_values'])
        model.W = checkpoint['W'].to(device)
        model.num_minimas = checkpoint['num_minimas']
        return model

    def train(self, filepath, info=False):
        self.random_noise(10)
        self.embed_img(num_images=10, info=info)
        self.save_model(filepath)

    def random_noise(self, repetitions):
        temp_wei = torch.zeros(self.num_values, self.num_values, device=device)
        for i in range(repetitions):
            self.random_values()
            self.W = self.random_weights()
            self.generate(find_minimum=True, info=False)
            self.update_weights()
            temp_wei -= self.W # subtracting random noise
        self.W = temp_wei


if __name__ == '__main__':
    model_path = 'boltzman-model.pth'
    train_again = False
    if train_again or not os.path.exists(model_path):
        print(f"Model not found at {model_path}\nTraining new model...")
        nn = Network(784)
        nn.train(model_path, info=True)
    loaded_nn = Network.load_model(model_path)
    # from specified
    loaded_nn.embed_test('test_1.png')
    loaded_nn.decode_img()
    loaded_nn.generate(max_iters=10000, find_minimum=True)
    loaded_nn.decode_img()
    # from random
    loaded_nn.random_values()
    loaded_nn.generate(find_minimum=True)
    loaded_nn.decode_img()
    print(torch.count_nonzero(loaded_nn.X == -1))
# '''
