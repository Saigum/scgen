import scanpy as sc
import anndata as ad
import numpy as np
from scipy.stats import median_abs_deviation

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

def preprocess_data(adata:ad.AnnData):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    ## outlier removal ????
    ##  log-normalization.
    sc.pp.normalize_total(adata,target_sum=1e4)
    ## remove mt genes ??
    


def load_data(dataset_str):
    if(dataset_str=="pbmc"):
        adata_unperturbed = sc.read_10x_mtx("../datasets/pbmc_perturb/sampleA_unperturbed")
        adata_peturbed = sc.read_10x_mtx("../datasets/pbmc_perturb/sampleB_perturbed")
    else:
        adata_peturbed = 1
        adata_unperturbed =0
        pass
    adata_peturbed.var_names_make_unique()
    adata_unperturbed.var_names_make_unique()
    return adata_peturbed,adata_unperturbed