import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

##############################################################################
#                                DATA LOADING                                #
##############################################################################

# Example placeholders for your data paths
train_data_path = "train_pbmc.h5ad"
val_data_path   = "valid_pbmc.h5ad"

# Read Anndata
train_adata = sc.read_h5ad(train_data_path)
val_adata   = sc.read_h5ad(val_data_path)

# 1) Highly Variable Gene Selection on the train set
#    to ensure we select the same HVGs for train and val.
#    We'll pick e.g. top 2000 HVGs from train_adata.
sc.pp.highly_variable_genes(train_adata, flavor="seurat", n_top_genes=2000)

# Subset train_adata to only HVGs
train_adata = train_adata[:, train_adata.var['highly_variable']]

# Subset val_adata to the same HVGs (intersection on gene names)
val_adata = val_adata[:, train_adata.var_names]

# Optional preprocess steps (log1p, scale, etc.)
# Uncomment if needed and ensure consistency between train and val
# sc.pp.log1p(train_adata)
# sc.pp.log1p(val_adata)
# sc.pp.scale(train_adata)
# sc.pp.scale(val_adata)

# Convert .X to torch tensors
X_train = torch.tensor(train_adata.X.toarray(), dtype=torch.float32)
X_val   = torch.tensor(val_adata.X.toarray(), dtype=torch.float32)

# Identify which cells are "stimulated" or "unstimulated".
# Adjust the 'condition' column as per your dataset.
train_pert_mask = (train_adata.obs['condition'] == 'stimulated').values
train_ctrl_mask = (train_adata.obs['condition'] == 'control').values

val_pert_mask = (val_adata.obs['condition'] == 'stimulated').values
val_ctrl_mask = (val_adata.obs['condition'] == 'control').values

# Extract cell types for plotting
# Replace 'cell_type' with the actual column name in your data
if 'cell_type' in val_adata.obs.columns:
    val_cell_types = val_adata.obs['cell_type'].values
else:
    raise ValueError("The 'cell_type' column is not found in val_adata.obs. Please adjust accordingly.")

##############################################################################
#                                DATASET CLASS                               #
##############################################################################

class TrainingCellData(Dataset):
    def __init__(self, X):
        """
        X: torch.Tensor of shape [n_cells, n_genes]
        """
        super().__init__()
        self.X = X

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

train_dataset = TrainingCellData(X_train)
val_dataset   = TrainingCellData(X_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

##############################################################################
#                               MODEL DEFINITION                             #
##############################################################################

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor):
    """
    Reparameterization trick:
    z = mu + sigma * eps, where sigma = exp(0.5 * logvar).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # same shape as std
    return mu + std * eps

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim):
        super().__init__()
        """
        Example:
          input_dim = 2000
          hidden_layers = [512, 256, 128]
          latent_dim = 64
        """

        # Build encoder
        encoder_modules = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            encoder_modules.append(nn.Linear(prev_dim, h_dim))
            encoder_modules.append(nn.ReLU())
            prev_dim = h_dim
        # Final layer to go down to some 'intermediate' dimension
        encoder_modules.append(nn.Linear(prev_dim, latent_dim))
        encoder_modules.append(nn.ReLU())
        self.encoder_block = nn.Sequential(*encoder_modules)

        # Separate heads for mu and logvar
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        # Build decoder (reverse)
        decoder_modules = []
        reversed_layers = list(reversed(hidden_layers))
        # First layer: latent_dim -> last hidden
        decoder_modules.append(nn.Linear(latent_dim, reversed_layers[0]))
        decoder_modules.append(nn.ReLU())
        prev_dim = reversed_layers[0]
        # Add intermediate hidden layers
        for h_dim in reversed_layers[1:]:
            decoder_modules.append(nn.Linear(prev_dim, h_dim))
            decoder_modules.append(nn.ReLU())
            prev_dim = h_dim
        # Final layer: to go back to input_dim
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        # We typically might not have a ReLU on the output, or
        # might have e.g. nn.Sigmoid() if data is normalized [0,1].
        # Adjust to your needs. For now, do no final activation:
        self.decoder_block = nn.Sequential(*decoder_modules)

    def encode(self, x):
        """
        Pass input through encoder; produce mu, logvar
        """
        encoded = self.encoder_block(x)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        return mu, logvar

    def decode(self, z):
        return self.decoder_block(z)

    def forward(self, x):
        """
        Standard VAE forward:
          1) encode
          2) reparam
          3) decode
          4) return reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

##############################################################################
#                           TRAINING & EVALUATION                            #
##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparams
input_dim     = X_train.shape[1]  # should be 2000 if HVG=2000
hidden_layers = [512, 256, 128]
latent_dim    = 64
lr            = 1e-3
num_epochs    = 10

model = VAE(input_dim, hidden_layers, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def kl_divergence(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.sum(kld)

def predict_perturbation_and_mse(model, unpert_X, pert_X):
    """
    Returns an MSE measure between the average predicted expression
    and the average actual expression for the stimulated set.
    """
    model.eval()
    with torch.no_grad():
        # Move data to device
        unpert_X = unpert_X.to(device)
        pert_X   = pert_X.to(device)

        # Encode both sets
        mu_unpert, logvar_unpert = model.encode(unpert_X)
        mu_pert,   logvar_pert   = model.encode(pert_X)

        mean_unpert = mu_unpert.mean(dim=0)  # shape [latent_dim]
        mean_pert   = mu_pert.mean(dim=0)    # shape [latent_dim]
        diff        = mean_pert - mean_unpert

        # Predict the "stimulated" version for each unpert cell
        preds = model.decode(mu_unpert + diff.unsqueeze(0))  # [N_unpert, G]
        avg_pred = preds.mean(dim=0)             # [G]

        # Compare to actual mean of pert_X
        avg_pert = pert_X.mean(dim=0)            # [G]
        mse = F.mse_loss(avg_pred, avg_pert, reduction='mean')
        return mse.item()

# Initialize LabelEncoder for cell types
le = LabelEncoder()
val_labels_encoded = le.fit_transform(val_cell_types)  # Convert string labels to integers

# Lists to store metrics for plotting
train_recon_losses = []
train_kl_losses = []
val_recon_losses = []
val_kl_losses = []
val_mse_per_epoch = []
sample_mse_per_epoch = []

# Lists to store latent representations for visualization
latent_representations = []
latent_labels = []

# Define number of samples to evaluate for prediction accuracy
num_sample_cells = 10  # You can adjust this number

# Pre-select random sample indices from the validation set for consistency across epochs
sample_indices = np.random.choice(
    np.where(val_ctrl_mask)[0], size=num_sample_cells, replace=False
)

for epoch in range(1, num_epochs+1):
    model.train()
    epoch_train_recon_loss = 0.0
    epoch_train_kl_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, batch, reduction='mean')
        # KL divergence
        kld = kl_divergence(mu, logvar)
        # Total VAE loss
        loss = recon_loss + kld
        loss.backward()
        optimizer.step()

        epoch_train_recon_loss += recon_loss.item() * batch.size(0)
        epoch_train_kl_loss    += kld.item() * batch.size(0)

    # Average epoch losses
    epoch_train_recon_loss /= len(train_loader.dataset)
    epoch_train_kl_loss    /= len(train_loader.dataset)
    train_recon_losses.append(epoch_train_recon_loss)
    train_kl_losses.append(epoch_train_kl_loss)

    # Validation
    model.eval()
    epoch_val_recon_loss = 0.0
    epoch_val_kl_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            recon_loss = F.mse_loss(recon, batch, reduction='mean')
            kld = kl_divergence(mu, logvar)

            epoch_val_recon_loss += recon_loss.item() * batch.size(0)
            epoch_val_kl_loss    += kld.item() * batch.size(0)

    epoch_val_recon_loss /= len(val_loader.dataset)
    epoch_val_kl_loss    /= len(val_loader.dataset)
    val_recon_losses.append(epoch_val_recon_loss)
    val_kl_losses.append(epoch_val_kl_loss)

    # Compute Validation MSE for perturbation prediction
    mse_val = predict_perturbation_and_mse(model, X_val[val_ctrl_mask], X_val[val_pert_mask])
    val_mse_per_epoch.append(mse_val)

    # Compute Sample MSEs: Predict perturbation for selected samples and compare to actual perturbed cells
    # Note: This requires paired data. If your validation set doesn't have paired cells,
    # you might need to adjust this part accordingly.

    # For demonstration, we'll assume that for each selected unperturbed cell,
    # there's a corresponding perturbed cell in the validation set.
    # This requires that val_adata has a pairing between control and stimulated cells.

    # If no pairing exists, you might compute the MSE between predicted perturbation and the average perturbed expression
    # Alternatively, you can skip this part or implement a different strategy.

    # Here, we'll proceed with the assumption of paired data.

    # Fetch the selected sample unperturbed cells
    sample_unpert_X = X_val[sample_indices].to(device)

    # For demonstration, assume that the first `num_sample_cells` stimulated cells are the true counterparts
    # WARNING: Replace this with your actual pairing logic
    true_pert_indices = np.random.choice(
        np.where(val_pert_mask)[0], size=num_sample_cells, replace=False
    )
    sample_true_pert_X = X_val[true_pert_indices].to(device)

    # Predict perturbations for the sample unperturbed cells
    model.eval()
    with torch.no_grad():
        mu_unpert, logvar_unpert = model.encode(sample_unpert_X)
        mean_unpert = mu_unpert.mean(dim=0)  # [latent_dim]
        # Compute mean perturbation vector
        mu_pert, logvar_pert = model.encode(X_val[val_pert_mask].to(device))
        mean_pert = mu_pert.mean(dim=0)      # [latent_dim]
        diff = mean_pert - mean_unpert      # [latent_dim]

        # Predict the perturbed version
        z_pred = mu_unpert + diff.unsqueeze(0)  # Broadcasting diff to match batch size
        preds = model.decode(z_pred)            # [num_sample_cells, G]
        # Compute MSE between predictions and true perturbed cells
        mse_samples = F.mse_loss(preds, sample_true_pert_X, reduction='mean').item()
        sample_mse_per_epoch.append(mse_samples)

    print(f"[Epoch {epoch}] "
          f"Train Recon: {epoch_train_recon_loss:.4f} | Train KL: {epoch_train_kl_loss:.4f} || "
          f"Val Recon: {epoch_val_recon_loss:.4f} | Val KL: {epoch_val_kl_loss:.4f} | "
          f"Val MSE: {mse_val:.4f} | Sample MSE: {mse_samples:.4f}")

##############################################################################
#                  PREDICTING PERTURBATION FROM UNstimulated                 #
##############################################################################
# The function `predict_perturbation_and_mse` remains the same as defined earlier.

##############################################################################
#                           LATENT SPACE EXTRACTION                          #
##############################################################################

def get_latent_representations(model, X, device):
    """
    Encodes the entire dataset and returns latent vectors.
    """
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        mu, logvar = model.encode(X)
        z = reparametrize(mu, logvar)
        latent_vectors = z.cpu().numpy()
    return latent_vectors

# Extract latent representations for the validation set
val_latent = get_latent_representations(model, X_val, device)

##############################################################################
#                                  PLOTTING                                   #
##############################################################################

# Plot training and validation reconstruction and KL losses
epochs = range(1, num_epochs+1)

plt.figure(figsize=(12, 5))

# Plot Reconstruction Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_recon_losses, label='Train Recon Loss')
plt.plot(epochs, val_recon_losses, label='Val Recon Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Reconstruction Loss')
plt.legend()

# Plot KL Divergence Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_kl_losses, label='Train KL Loss')
plt.plot(epochs, val_kl_losses, label='Val KL Loss')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.title('KL Divergence Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot Validation MSE and Sample MSE over epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_mse_per_epoch, label='Validation MSE')
plt.plot(epochs, sample_mse_per_epoch, label='Sample MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Perturbation Prediction MSE Over Epochs')
plt.legend()
plt.show()

##############################################################################
#                           LATENT SPACE VISUALIZATION                       #
##############################################################################

def plot_latent_space(latent_vectors, labels, label_names, method='tsne', title='Latent Space'):
    """
    Plots the latent space using PCA or t-SNE.

    Parameters:
    - latent_vectors: numpy array of shape [n_samples, latent_dim]
    - labels: numpy array of shape [n_samples], numerical labels
    - label_names: list or array of label names corresponding to numerical labels
    - method: 'tsne' or 'pca'
    - title: Title of the plot
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    latent_2d = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:,0], latent_2d[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Latent Space Visualization using {method.upper()}')

    # Create a legend with unique labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles, label_names, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot using t-SNE
plot_latent_space(val_latent, val_labels_encoded, le.classes_, method='tsne', title='Latent Space (t-SNE)')

# Plot using PCA
plot_latent_space(val_latent, val_labels_encoded, le.classes_, method='pca', title='Latent Space (PCA)')
