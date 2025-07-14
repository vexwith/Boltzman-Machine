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
        self.num_hidden = num_values * 4
        self.num_minimas = 0
        self.X = torch.ones(num_values, 1, device=device)
        self.hidden = torch.ones(self.num_hidden, 1, device=device)
        self.W = torch.zeros(num_values, self.num_hidden, device=device) # Simple linear connection from X to hidden

    def random_values(self):
        self.X = torch.ones(self.num_values, 1, device=device)
        self.X[torch.rand(self.num_values, 1, device=device) > 0.5] = -1
        self.random_hidden()

    def random_hidden(self):
        with torch.no_grad():
            # Random projection (adjust weights as needed)
            W_init = torch.randn(self.num_hidden, self.num_values, device=device) * 0.1  # Small random weights
            self.hidden = torch.sigmoid(W_init @ self.X)  # Non-linear transform
            rand_hidden = torch.randn(self.num_hidden, 1, device=device)
            self.hidden = torch.where(rand_hidden > self.hidden, 1.0, -1.0)
        # self.hidden = torch.ones(self.num_hidden, 1, device=device)
        # self.hidden[torch.rand(self.num_hidden, 1, device=device) > 0.5] = -1

    def random_weights(self):
        W = torch.ones(self.num_values, self.num_hidden, device=device)
        W[torch.rand(self.num_values, self.num_hidden, device=device) > 0.5] = -1
        # W.fill_diagonal_(0)
        # W = W.tril() + W.T.triu() # Weights need to be symmetrical
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
                self.X = image.squeeze(0).flatten().unsqueeze(1).to(device)
                self.X = torch.where(self.X > 0, 1.0, -1.0)
                if info: print(f"idx: {i}  label: {label_name}")
                self.random_hidden()
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

        # Ensure shapes match (safety check)
        shape = binary_tensor.shape
        num_pixels = shape[0] * shape[1]
        assert num_pixels == self.num_values, \
            f"Shape mismatch: expected {self.X.shape}, got {num_pixels}"

        binary_tensor = binary_tensor.reshape(self.num_values, 1) # [num_values, 1]
        # Copy values into self.X (in-place to avoid reallocation)
        self.X.copy_(binary_tensor)
        self.random_hidden()

    def update_weights(self):
        # Makes [num_values x num_hidden] matrix of every x * h
        self.W += torch.outer(self.X.squeeze(), self.hidden.squeeze())
        # self.W.fill_diagonal_(0)
        self.num_minimas += 1

    def get_neuron_energy(self, idx, get_hidden):
        if get_hidden: # Get the energy of a hidden node
            return (self.W[:, idx] @ self.X) * self.hidden[idx]
        else:
            return (self.W[idx] @ self.hidden) * self.X[idx]

    def get_system_energy(self): # we want it to be the highest
        return ((self.W.sign() @ self.hidden).T @ self.X).item() # sum of every (weight_ij * x * h)

    def update_node(self, values, idx, get_hidden, temperature):
        neuron_energy = self.get_neuron_energy(idx, get_hidden)
        probability = 1 / (1 + torch.exp(-neuron_energy / temperature))  # sigmoid - probability of saying the same
        r = torch.rand(1, device=device)
        if r > probability:
            values[idx] = -values[idx]

    def generate(self, max_iters=1000, find_minimum=False, info=True):
        if info:
            fig, img_display = self.init_plot()

        counter = 0
        temperature = 5
        tcr = 0.005 # temperature change rate
        best_energy = self.get_system_energy()
        while find_minimum or counter < max_iters:
            prev_energy = best_energy
            for idx in range(self.num_hidden + self.num_values):
                if idx >= self.num_hidden:
                    self.update_node(self.X, idx - self.num_hidden, False, temperature)
                else:
                    self.update_node(self.hidden, idx, True, temperature)

                cur_energy = self.get_system_energy()
                if cur_energy > best_energy:
                    best_energy = cur_energy

                if idx >= self.num_hidden:
                    counter += 1
                    temperature = max(0, temperature - tcr)
                    if info: # Don't show hidden nodes
                        if counter % 100 == 0:
                            print(f"system energy: {cur_energy}  counter: {counter}  best_se: {best_energy}  "
                                  f"max_se: {self.num_values * self.num_hidden}")
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

    def train(self, filepath, epochs, info=False):
        temp_wei = torch.zeros(self.num_values, self.num_hidden, device=device)
        for i in range(epochs):
            self.random_noise(10)
            self.embed_img(use_all=True, info=info)
            temp_wei += self.W
        self.W = temp_wei
        self.save_model(filepath)

    def random_noise(self, repetitions):
        temp_wei = torch.zeros(self.num_values, self.num_hidden, device=device)
        for i in range(repetitions):
            self.random_values()
            self.W = self.random_weights()
            self.generate(max_iters=2000, info=False)
            self.update_weights()
            temp_wei -= self.W # subtracting random noise
        self.W = temp_wei


if __name__ == '__main__':
    nn = Network(784)
    # nn.random_noise(10)
    nn.embed_img(num_images=10, num_labels=1)
    nn.embed_test('test_2.png')
    nn.decode_img()
    nn.generate(find_minimum=True)
    nn.decode_img()
    # model_path = 'boltzman-model.pth'
    # train_again = False
    # if train_again or not os.path.exists(model_path):
    #     print(f"Model not found at {model_path}\nTraining new model...")
    #     nn = Network(784)
    #     nn.train(model_path, 10, info=False)
    # loaded_nn = Network.load_model(model_path)
    # # from specified
    # loaded_nn.embed_test('test_2.png')
    # loaded_nn.decode_img()
    # loaded_nn.generate(max_iters=10000, find_minimum=True)
    # loaded_nn.decode_img()
    # # from random
    # loaded_nn.random_values()
    # loaded_nn.generate(find_minimum=True)
    # loaded_nn.decode_img()
    # print(torch.count_nonzero(loaded_nn.X == -1))
# '''
