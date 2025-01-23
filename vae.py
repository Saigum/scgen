import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scanpy as sc
from tqdm import tqdm

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

# A few optional preprocess steps (log1p, scale, etc.),
# depending on how your data was preprocessed already:
# sc.pp.log1p(train_adata)
# sc.pp.log1p(val_adata)
# sc.pp.scale(train_adata)
# sc.pp.scale(val_adata)
# (Be consistent in your train/val transformations!)

# Convert .X to torch tensors
X_train = torch.tensor(train_adata.X.toarray(), dtype=torch.float32)
X_val   = torch.tensor(val_adata.X.toarray(), dtype=torch.float32)


# Identify which cells are "stimulated" or "unstimulated".
# For simplicity, we'll assume:
#   - train_adata.obs['condition'] in {'stimulated','control'}
#   - val_adata.obs['condition']   in {'stimulated','control'}
# Adjust to your actual metadata columns accordingly.
train_pert_mask = (train_adata.obs['condition'] == 'stimulated').values
train_ctrl_mask = (train_adata.obs['condition'] == 'control').values

val_pert_mask = (val_adata.obs['condition'] == 'stimulated').values
val_ctrl_mask = (val_adata.obs['condition'] == 'control').values

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
    # Standard formula for each sample:
    # KL = -0.5 * \sum(1 + logvar - mu^2 - exp(logvar))
    # Return the mean over batch, or sum, depending on how you want to scale
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.sum(kld)

for epoch in range(1, num_epochs+1):
    model.train()
    train_recon_loss = 0.0
    train_kl_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        # Reconstruction loss (MSE, or could use e.g. F.binary_cross_entropy())
        recon_loss = F.mse_loss(recon, batch, reduction='mean')
        # KL
        kld = kl_divergence(mu, logvar)
        # Total VAE loss
        loss = recon_loss + kld
        loss.backward()
        optimizer.step()

        train_recon_loss += recon_loss.item() * batch.size(0)
        train_kl_loss    += kld.item() * batch.size(0)

    # Average epoch losses
    train_recon_loss /= len(train_loader.dataset)
    train_kl_loss    /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_recon_loss = 0.0
    val_kl_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            recon_loss = F.mse_loss(recon, batch, reduction='mean')
            kld = kl_divergence(mu, logvar)

            val_recon_loss += recon_loss.item() * batch.size(0)
            val_kl_loss    += kld.item() * batch.size(0)

    val_recon_loss /= len(val_loader.dataset)
    val_kl_loss    /= len(val_loader.dataset)

    print(f"[Epoch {epoch}] "
          f"Train Recon: {train_recon_loss:.4f} | Train KL: {train_kl_loss:.4f} || "
          f"Val Recon: {val_recon_loss:.4f} | Val KL: {val_kl_loss:.4f}")

##############################################################################
#                  PREDICTING PERTURBATION FROM UNstimulated                 #
##############################################################################
# We'll define a function that takes:
#   model      - the trained VAE
#   unpert_X   - tensor of unstimulated cells [N, G]
#   pert_X     - tensor of stimulated cells   [M, G]
#
# Strategy:
#  1) Encode unpert_X => mu_unpert
#  2) Encode pert_X   => mu_pert
#  3) mean_unpert = average of mu_unpert
#     mean_pert   = average of mu_pert
#  4) For each unpert cell, create z_pred[i] = mu_unpert[i] + (mean_pert - mean_unpert)
#  5) Decode z_pred[i] => predicted expression of the "stimulated" version
#  6) Compare to actual stimulated expression (in some aggregated sense).
#     Without a 1-to-1 cell mapping, we often compare average expression:
#       MSE( mean of predicted_pert , mean of actual_pert_X )
#     or we can do other strategies if your data is matched.

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

        # Compute mean latent vectors
        mean_unpert = mu_unpert.mean(dim=0)  # shape [latent_dim]
        mean_pert   = mu_pert.mean(dim=0)    # shape [latent_dim]
        diff        = mean_pert - mean_unpert

        # Predict the "stimulated" version for each unpert cell
        preds_list = []
        for i in range(unpert_X.shape[0]):
            z_pred = mu_unpert[i] + diff
            x_pred = model.decode(z_pred.unsqueeze(0))  # shape [1, G]
            preds_list.append(x_pred.squeeze(0))

        preds = torch.stack(preds_list, dim=0)    # [N_unpert, G]
        avg_pred = preds.mean(dim=0)             # [G]

        # Compare to actual mean of pert_X
        avg_pert = pert_X.mean(dim=0)            # [G]
        mse = F.mse_loss(avg_pred, avg_pert, reduction='mean')
        return mse.item()

# Example usage on validation set:
# In the validation set, separate out the unstimulated cells and stimulated cells
X_val_unpert = X_val[val_ctrl_mask]
X_val_pert   = X_val[val_pert_mask]

mse_val = predict_perturbation_and_mse(model, X_val_unpert, X_val_pert)
print(f"Validation MSE for perturbation prediction = {mse_val:.4f}")
