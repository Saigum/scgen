from torch.nn import functional as f
from torch import nn, optim
import torch as th
import numpy as np
from torch.utils.data import DataLoader,Dataset
from  preprocessing import load_data
import argparse
from tqdm import tqdm


data_path = "../dataset/pbmc_perturb"
### Sample B is the perturbed set,
### Sample A is the non perturbed set.


device = th.device("cuda" if th.cuda.is_available() else "cpu")

def reparametrize(mu:th.Tensor,logvar:th.Tensor):
    return(mu + (th.exp(2*logvar)*th.randn(size=mu.shape[1])))

class VAE(nn.Module):
    def __init__(self,input_dim,hidden_layers,latent_dim):
        layers = [input_dim] + hidden_layers
        layers.append(latent_dim)
        self.encoder_block = nn.Sequential([nn.Sequential
                                            (nn.Linear(layers[i],layers[i+1]),nn.ReLU()) 
                                            for i in range(len(layers)-1)])
        self.decoder_block = nn.Sequential([nn.Sequential
                                            (nn.Linear(layers[i],layers[i-1]),nn.ReLU()) 
                                            for i in range(len(layers)-1,-1,-1)])
        self.mu = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.ReLU())
        self.logvar = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.ReLU())
    def forward(self,x):
        encoded = self.encoder_block(x)
        mu,logvar = self.mu(encoded),self.logvar(encoded)
        decoded = self.decoder_block(reparametrize(mu,logvar))
        return decoded,mu,logvar


## for this dataset, I'm going to assume you provide 2 adatas.
## one being adata's for the perturbed set
## one being adata for the non-perturbed set, they need not be equal in size.
class TrainingCellData(Dataset):
    def __init__(self,adata,is_perturbed=0):
        self.adata = adata
        self.is_perturbed = is_perturbed
    def __getitem__(self, index):
        ## Equivalent to you getting it for a cell
        # n_cells x n_genes
        gene_vector = th.zeros(self.adata.shape[1])
        gene_vector = self.adata.X[index]
        return gene_vector
    def __len__(self):
        return(self.adata.shape[0])
    


adata_peturbed,adata_unperturbed = load_data(data_path)





def accuracy(mu,logvar)




hvg_size = 2000

## have a cell_specific loader too, where you can choose excluded cell types.


train_set = TrainingCellData(adata)
val_set = TrainingCellData(val_adata)
trainloader = DataLoader(train_set,batch_size=4)
valloader =DataLoader(val_set,batch_size=4)

hidden_layers = [512,256,128]
latent_dim=64
num_epochs = 100
lr = 3e3
model = VAE(hvg_size,hidden_layers,latent_dim).to(device)
optimizer = optim.adagrad(VAE.parameters(),lr)

with tqdm(total=len(trainloader)*num_epochs) as pbar:
    ### training phase
    for epoch in range(1,num_epochs+1):
        epoch_wise_recon_loss =0
        epoch_wise_kl_loss = 0
        for batch in trainloader:
            optimizer.zero_grad()
            reconstruction,mu,logvar = model(batch.to(device))
            recon_loss = f.mse_loss(reconstruction,batch)
            kl_loss = 0.5(-logvar.sum() -1 + mu.T@mu  + logvar.exp() )
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            pbar.write(f'Loss: {loss.item()}')
            pbar.update()
            epoch_wise_recon_loss += recon_loss
            epoch_wise_kl_loss += kl_loss
        epoch_wise_kl_loss = epoch_wise_kl_loss/len(trainloader)
        epoch_wise_recon_loss = epoch_wise_recon_loss/len(trainloader)
        print(f"Recon Loss for epoch {epoch} is {epoch_wise_recon_loss}")
        print(f"KL Loss for epoch {epoch} is {epoch_wise_kl_loss}")

        ## validation phase.
        with tqdm(total=len(valloader),leave=False) as pbar:
            epoch_wise_recon_loss =0
            epoch_wise_kl_loss = 0
            model.eval()
            with th.no_grad():
                for batch in valloader:
                    reconstruction,mu,logvar = model(batch.to(device))
                    recon_loss = f.mse_loss(reconstruction,batch)
                    kl_loss = 0.5(-logvar.sum() -1 + mu.T@mu  + logvar.exp() )
                    epoch_wise_recon_loss += recon_loss.item()
                    epoch_wise_kl_loss += kl_loss
                epoch_wise_kl_loss = epoch_wise_kl_loss/len(valloader)
                epoch_wise_recon_loss = epoch_wise_recon_loss/len(valloader)
                print(f"Val Recon Loss for epoch {epoch} is {epoch_wise_recon_loss}")
                print(f"Val KL Loss for epoch {epoch} is {epoch_wise_kl_loss}")
            ## now let's calculate  how well it predicts the 100 degs
            ## and how well it predicts the actual 

                

            

            


        
    





        