base_dir = "/content/traffic_imputation"
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "outputs")


os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Verify the base path
base_path = "/content/drive/My Drive/deeploycv"
print(f"Base path exists: {os.path.exists(base_path)}")

import os
import numpy as np

# Paths
output_folder = "/content/drive/My Drive/deeploycv/output"

# Verify if the output folder exists
if os.path.exists(output_folder):
    print(f"Output folder found: {output_folder}")
    # List all files in the folder
    files = os.listdir(output_folder)
    print(f"Files in output folder: {files}")
else:
    print("Output folder does not exist. Please check the path.")

# Filter GASF matrix files
gasf_files = [f for f in os.listdir(output_folder) if f.endswith('_gasf_matrix.npy')]

# Load a sample GASF matrix
if gasf_files:
    sample_gasf_path = os.path.join(output_folder, gasf_files[0])
    gasf_matrix = np.load(sample_gasf_path)

    print(f"Loaded GASF matrix from: {sample_gasf_path}")
    print(f"GASF Matrix shape: {gasf_matrix.shape}")

    # Display the first few values of the matrix for verification
    print("Sample of GASF matrix values:")
    print(gasf_matrix[:5, :5])
else:
    print("No GASF matrices found in the output folder.")

import matplotlib.pyplot as plt

# Visualize the GASF matrix
plt.figure(figsize=(6, 6))
plt.imshow(gasf_matrix, cmap="hot", interpolation="nearest")
plt.colorbar(label="GASF Value")
plt.title("GASF Matrix Visualization")
plt.axis("off")  # Hide axes for better visual clarity
plt.show()

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to Apply Mask
def apply_mask(matrix, mask_ratio=0.2):

    matrix_size = matrix.shape[0]
    mask = torch.ones_like(matrix)
    num_masked = int(mask_ratio * matrix_size * matrix_size)

    # Randomly select indices to mask
    masked_indices = torch.randperm(matrix_size * matrix_size)[:num_masked]
    mask.view(-1)[masked_indices] = 0

    # Apply the mask to the matrix
    masked_matrix = matrix * mask

    return masked_matrix, mask

# Enhanced Dataset Class
class GASFDataset(Dataset):
    def __init__(self, gasf_folder, target_size=12, mask_ratio=0.2, normalize_range=(-1, 1)):
        self.gasf_files = [
            os.path.join(gasf_folder, f) for f in os.listdir(gasf_folder) if f.endswith("_gasf_matrix.npy")
        ]
        self.target_size = target_size
        self.mask_ratio = mask_ratio
        self.normalize_range = normalize_range  # Tuple (min_val, max_val)

    def __len__(self):
        return len(self.gasf_files)

    def __getitem__(self, idx):
        # Load the GASF matrix
        gasf_matrix = np.load(self.gasf_files[idx])
        gasf_matrix = torch.tensor(gasf_matrix, dtype=torch.float32)

        # Normalize the matrix to the specified range
        min_val, max_val = self.normalize_range
        gasf_matrix = (max_val - min_val) * (gasf_matrix - gasf_matrix.min()) / (
            gasf_matrix.max() - gasf_matrix.min() + 1e-6
        ) + min_val

        # Resize or pad the matrix to the target size
        gasf_matrix = F.pad(
            gasf_matrix,
            pad=(0, self.target_size - gasf_matrix.shape[1], 0, self.target_size - gasf_matrix.shape[0]),
            mode="constant",
            value=0,
        )

        # Apply masking
        masked_matrix, mask = apply_mask(gasf_matrix, mask_ratio=self.mask_ratio)

        return masked_matrix.unsqueeze(0), mask.unsqueeze(0), gasf_matrix.unsqueeze(0)


# Define Output Folder
output_folder = "/content/drive/My Drive/deeploycv/output"  # Update your path as needed
if not os.path.exists(output_folder):
    raise ValueError("Output folder does not exist. Please provide the correct path.")

# Initialize Dataset and DataLoader
gasf_dataset = GASFDataset(output_folder, target_size=12, mask_ratio=0.2, normalize_range=(-1, 1))
gasf_loader = DataLoader(gasf_dataset, batch_size=8, shuffle=True, drop_last=True)


# Verify the DataLoader and Visualize
for batch_idx, (masked_matrices, masks, original_matrices) in enumerate(gasf_loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Masked Matrices Shape: {masked_matrices.shape}")
    print(f"Masks Shape: {masks.shape}")
    print(f"Original Matrices Shape: {original_matrices.shape}")

    # Visualize one sample
    sample_idx = 0
    masked_matrix = masked_matrices[sample_idx].squeeze(0).numpy()
    mask = masks[sample_idx].squeeze(0).numpy()
    original_matrix = original_matrices[sample_idx].squeeze(0).numpy()

    # Plot original, mask, and masked matrix
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_matrix, cmap="hot")
    plt.title("Original GASF Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask (Randomly Generated)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(masked_matrix, cmap="hot")
    plt.title("Masked GASF Matrix")
    plt.axis("off")

    plt.show()
    break  # Only visualize one batch

import torch.nn as nn

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_channels=2, matrix_size=12):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # Input: [batch, 2, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output: [batch, 1, 12, 12]
            nn.Tanh()  # Output values in range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, matrix_size=12):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (matrix_size // 4) * (matrix_size // 4), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Check if GPU is available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models, optimizers, and loss functions
generator = Generator(matrix_size=12).to(device)
discriminator = Discriminator(matrix_size=12).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# GAN Training Loop
num_epochs = 100
latent_dim = 100
log_interval = 10  # Logging frequency
save_interval = 10  # Save/visualize frequency

for epoch in range(num_epochs):
    for i, (masked_matrices, masks, original_matrices) in enumerate(gasf_loader):
        # Move data to the appropriate device
        masked_matrices = masked_matrices.to(device)
        masks = masks.to(device)
        original_matrices = original_matrices.to(device)
        batch_size = masked_matrices.size(0)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real labels for original matrices
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # Real loss
        real_loss = adversarial_loss(discriminator(original_matrices), real_labels)

        # Fake loss
        input_for_generator = torch.cat((masked_matrices, masks), dim=1)  # Concatenate masked input and mask
        reconstructed_matrices = generator(input_for_generator)
        fake_loss = adversarial_loss(discriminator(reconstructed_matrices.detach()), fake_labels)

        # Total Discriminator Loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generator adversarial loss
        g_adv_loss = adversarial_loss(discriminator(reconstructed_matrices), real_labels)

        # Generator reconstruction loss (focuses on masked regions)
        g_recon_loss = reconstruction_loss(
            reconstructed_matrices * (1 - masks), original_matrices * (1 - masks)
        )

        # Total Generator Loss
        g_loss = g_adv_loss + g_recon_loss
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Logging
        if i % log_interval == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(gasf_loader)}], "
                f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Recon Loss: {g_recon_loss.item():.4f}"
            )

    # ---------------------
    # Save and Visualize Reconstructed Samples
    # ---------------------
    if (epoch + 1) % save_interval == 0:
        with torch.no_grad():
            sample_reconstructed = reconstructed_matrices[0].squeeze().cpu().numpy()
            sample_original = original_matrices[0].squeeze().cpu().numpy()
            sample_masked = masked_matrices[0].squeeze().cpu().numpy()

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(sample_original, cmap="hot")
            plt.title("Original GASF Matrix")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(sample_masked, cmap="hot")
            plt.title("Masked GASF Matrix")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(sample_reconstructed, cmap="hot")
            plt.title(f"Reconstructed (Epoch {epoch+1})")
            plt.axis("off")

            plt.show()

# Load a batch for evaluation
generator.eval()
masked_matrices, masks, original_matrices = next(iter(gasf_loader))
masked_matrices = masked_matrices.to(device)
masks = masks.to(device)
original_matrices = original_matrices.to(device)

# Generate imputed matrices
with torch.no_grad():
    input_for_generator = torch.cat((masked_matrices, masks), dim=1)  # Concatenate masked input and masks
    imputed_matrices = generator(input_for_generator)

# Visualize Results
for i in range(3):  # Visualize first 3 samples
    original = original_matrices[i].squeeze(0).cpu().numpy()
    masked = masked_matrices[i].squeeze(0).cpu().numpy()
    imputed = imputed_matrices[i].squeeze(0).cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="hot")
    plt.title("Original Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(masked, cmap="hot")
    plt.title("Masked Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(imputed, cmap="hot")
    plt.title("Imputed Matrix")
    plt.axis("off")

    plt.show()

# Initialize a counter before training starts
total_synthetic = 0

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (masked_matrices, masks, original_matrices) in enumerate(gasf_loader):
        # Training logic here (omitted for brevity)

        # Count synthetic matrices
        total_synthetic += masked_matrices.size(0)

    # Logging or visualization per epoch (optional)

# After Training
print(f"Total synthetic matrices generated during training: {total_synthetic}")

# Set the generator to evaluation mode
generator.eval()

# Initialize accumulators for RMSE and MAE
rmse_accumulator = 0
mae_accumulator = 0
num_batches = 0

# Evaluate over the DataLoader
for batch_idx, (masked_matrices, masks, original_matrices) in enumerate(gasf_loader):
    masked_matrices = masked_matrices.to(device)
    masks = masks.to(device)
    original_matrices = original_matrices.to(device)

    # Generate imputed matrices
    with torch.no_grad():
        input_for_generator = torch.cat((masked_matrices, masks), dim=1)
        imputed_matrices = generator(input_for_generator)

    # Calculate RMSE and MAE for masked regions only
    diff = imputed_matrices - original_matrices
    mse = torch.mean((diff * (1 - masks)) ** 2)  # Mean Squared Error
    mae = torch.mean(torch.abs(diff * (1 - masks)))  # Mean Absolute Error

    rmse_accumulator += torch.sqrt(mse).item()
    mae_accumulator += mae.item()
    num_batches += 1

# Compute average RMSE and MAE
average_rmse = rmse_accumulator / num_batches
average_mae = mae_accumulator / num_batches

print(f"Average RMSE: {average_rmse:.4f}")
print(f"Average MAE: {average_mae:.4f}")