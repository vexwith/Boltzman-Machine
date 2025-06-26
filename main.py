import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(mnist_pytorch, batch_size=1, shuffle=False)

class Network:

    def __init__(self, num_values):
        self.num_values = num_values
        self.num_minimas = 0
        self.X = torch.ones(num_values, 1)
        # self.X[:num_values//2] = -1
        self.W = torch.zeros(num_values, num_values)

    def random_values(self):
        self.X = torch.ones(self.num_values, 1)
        self.X[torch.rand(self.num_values, 1) > 0.5] = -1

    def random_weights(self):
        self.W = torch.ones(self.num_values, self.num_values)
        self.W[torch.rand(self.num_values, self.num_values) > 0.5] = -1
        self.W.fill_diagonal_(0)
        self.W = self.W.tril() + self.W.T.triu()

    def embed_img(self, num_images=1):
        for i, (image, label) in enumerate(train_loader_pytorch):
            if i >= num_images: break
            self.X = image.squeeze(0).flatten()
            self.X = torch.where(self.X > 0, torch.tensor(1), torch.tensor(-1))
            print(f"label: {label.item()}")
        self.update_weights()

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
        img_array = np.array(img)  # Shape: (H, W), dtype: uint8

        # Convert to PyTorch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        binary_tensor = torch.where(img_tensor == 1, -1.0, 1.0)

        # Ensure shapes match (safety check)
        assert binary_tensor.shape == self.X.shape, \
            f"Shape mismatch: expected {self.X.shape}, got {binary_tensor.shape}"

        # Copy values into self.X (in-place to avoid reallocation)
        self.X.copy_(binary_tensor)
        self.update_weights()

    def update_weights(self):
        for i, row in enumerate(self.W):
            for j, col in enumerate(row):
                if i == j: continue
                self.W[i][j] += self.X[i].item() * self.X[j].item()
        self.num_minimas += 1

    def get_neuron_neighbour_energy(self, idx):
        return self.W[idx] @ self.X

    def get_system_energy(self, wei): # we want it to be the highest
        return ((wei @ self.X).T @ self.X).item() / self.num_minimas # sum of every (weight_ij * x_i * x_j)

    def generate(self, max_iters=1000):
        fig, img_display = self.init_plot()

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
                self.update_plot(img_display, 0.005)

                if counter >= max_iters: break
            if prev_energy - cur_energy == 0: break
        plt.close(fig)
        plt.ioff()


nn = Network(784)
nn.embed_img()
nn.random_values()
# print(nn.X)
# print(torch.count_nonzero(nn.X == -1))
# print(nn.W)
# print(torch.count_nonzero(nn.W < 0) / torch.count_nonzero(nn.W > 0))
# print(nn.get_system_energy(nn.W.triu()))
nn.generate(max_iters=1000)
nn.decode_img()
print(torch.count_nonzero(nn.X == -1))

