import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Discrete fixed-time survival
def survival_fixed_time(fix_time, time, event):
    """
    It returns 0 if the individual survives the fixed time point.
    Else, it returns 1 if the event occurs before the fixed time point.
    None is returned if the individual is censored before the fixed time point.
    """
    if (time > fix_time): return 0
    else:
        if (event == 1.0): return 1
        else: return None

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1   
    weights = 1 / torch.Tensor(count)
    weights = weights.double()    
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weights[val[1]]
    return weight
def get_weights_focal_loss(images, nclasses, mode="proportion"):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1  
    if mode == "proportion":
        weights = torch.Tensor(count)/sum(count)
    elif mode == "count":
        weights = 1/torch.Tensor(count)
    weights = weights.double()    
    return weights

class TranscriptomicImagesDatasetNonLung(Dataset):
    """
    Dataset for transcriptomic data seen as images using the KEGG-BRITE functional hierarchy for the
    TCGA Pan-Cancer gene-expression dataset and the TCGA Pan-Cancer curated clinical dataset (Non lung).
    Code widely inspired from https://github.com/guilopgar/GeneExpImgTL
    """

    def __init__(self, path_to_pan_cancer_hdf5_files, path_to_treemap_images):
        """
        :param path_to_pan_cancer_hdf5_files: path to hdf5 files (without final /) containing transcriptomic data (both lung and non lung)
        :param path_to_treemap_images: path to tree map images for transcriptomic data based on the KEGG-BRITE functional hierarchy
        """
        
        # step 1 : get the transcriptomic images
        # Define survival variable of interest
        surv_variable = "PFI"
        surv_variable_time = "PFI.time"
        # Load samples-info dataset
        file_name_to_use = "/non_Lung_pancan.h5" 
        
        Y_info = pd.read_hdf(path_to_pan_cancer_hdf5_files+file_name_to_use, 
                     key="sample_type")
        # Load survival clinical outcome dataset
        Y_surv = pd.read_hdf(path_to_pan_cancer_hdf5_files+file_name_to_use, 
                     key="sample_clinical")
        # Filter tumor samples from survival clinical outcome dataset
        Y_surv = Y_surv.loc[Y_info.tumor_normal=="Tumor"]
        # Drop rows where surv_variable or surv_variable_time is NA
        Y_surv.dropna(subset=[surv_variable, surv_variable_time], inplace=True)
        # create a discrete time class variable using the fixed-time point.
        time = 230
        Y_surv_disc = Y_surv[['PFI', 'PFI.time']].apply(lambda row: survival_fixed_time(time, row['PFI.time'], row['PFI']), axis=1)
        Y_surv_disc.dropna(inplace=True)
        # Load gene-exp images
        n_width = 175
        n_height = 175
        dir_name = path_to_treemap_images+"/gene_exp_treemap_" + str(n_width) + "_" + str(n_height) + "_npy/"
        # images matrix is :
        X_gene_exp = np.array([np.load(dir_name + s.replace("-", ".") + ".npy") for s in Y_surv_disc.index])
        self.images = torch.tensor(X_gene_exp)
        # class labels are :
        Y_surv_disc_class = LabelEncoder().fit_transform(Y_surv_disc)
        self.labels = torch.tensor(Y_surv_disc_class)
        self.mean, self.std = self.get_mean_std()
        self.std = torch.where(self.std == 0,  torch.tensor([1.]).double(),  self.std)
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, ix):
        return ((self.images[ix]-self.mean)/self.std).unsqueeze(0), self.labels[ix]
    def get_mean_std(self):
        return self.images.mean(dim = 0), self.images.std(dim = 0)

class TranscriptomicImagesDatasetLung(Dataset):
    """
    Dataset for transcriptomic data seen as images using the KEGG-BRITE functional hierarchy for the
    TCGA Pan-Cancer gene-expression dataset and the TCGA Pan-Cancer curated clinical dataset (lung).
    Code widely inspired from https://github.com/guilopgar/GeneExpImgTL
    """

    def __init__(self, path_to_pan_cancer_hdf5_files, path_to_treemap_images):
        """
        :param path_to_pan_cancer_hdf5_files: path to hdf5 files (without final /) containing transcriptomic data (both lung and non lung)
        :param path_to_treemap_images: path to tree map images for transcriptomic data based on the KEGG-BRITE functional hierarchy
        """
        
        # step 1 : get the transcriptomic images
        # Define survival variable of interest
        surv_variable = "PFI"
        surv_variable_time = "PFI.time"
        # Load samples-info dataset
        Y_info_ft = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/Lung_pancan.h5", key="sample")
        # Load survival clinical outcome dataset
        Y_surv_ft = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/Lung_pancan.h5", key="survival_outcome")
        # Filter tumor samples from survival clinical outcome dataset
        Y_surv_ft = Y_surv_ft.loc[Y_info_ft.tumor_normal=="Tumor"]
        # Drop rows where surv_variable or surv_variable_time is NA
        Y_surv_ft.dropna(subset=[surv_variable, surv_variable_time], inplace=True)
        # create a discrete time class variable using the fixed-time point.
        time = 230
        Y_surv_disc_ft = Y_surv_ft[['PFI', 'PFI.time']].apply(lambda row: survival_fixed_time(time, row['PFI.time'], row['PFI']), axis=1)
        Y_surv_disc_ft.dropna(inplace=True)
        # get the images
        # Load gene-exp images
        n_width = 175
        n_height = 175
        dir_name = path_to_treemap_images+"/gene_exp_treemap_" + str(n_width) + "_" + str(n_height) + "_npy/"
        X_gene_exp_ft = np.array([np.load(dir_name + s.replace("-", ".") + ".npy") for s in Y_surv_disc_ft.index])
        # Convert discrete time survival numerical variables into binary variables
        Y_surv_disc_class_ft = LabelEncoder().fit_transform(Y_surv_disc_ft)
        
        self.images = torch.tensor(X_gene_exp_ft)
        self.labels = torch.tensor(Y_surv_disc_class_ft)
        self.mean, self.std = self.get_mean_std()
        self.std = torch.where(self.std == 0,  torch.tensor([1.]).double(),  self.std)
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, ix):
        return ((self.images[ix]-self.mean)/self.std).unsqueeze(0), self.labels[ix]
    def get_mean_std(self):
        return self.images.mean(dim = 0), self.images.std(dim = 0)
    
    
    
class TranscriptomicVectorsDatasetNonLung(Dataset):
    """
    Dataset for transcriptomic data seen as vectors (raw) for the
    TCGA Pan-Cancer gene-expression dataset and the TCGA Pan-Cancer curated clinical dataset (Non lung).
    Code widely inspired from https://github.com/guilopgar/GeneExpImgTL
    """

    def __init__(self, path_to_pan_cancer_hdf5_files):
        """
        :param path_to_pan_cancer_hdf5_files: path to hdf5 files (without final /) containing transcriptomic data (both lung and non lung)
        """
        
        # Define survival variable of interest
        surv_variable = "PFI"
        surv_variable_time = "PFI.time"
        # Load samples-info dataset
        Y_info = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/non_Lung_pancan.h5", 
                     key="sample_type")
        # Load survival clinical outcome dataset
        Y_surv = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/non_Lung_pancan.h5", 
                     key="sample_clinical")
        # Filter tumor samples from survival clinical outcome dataset
        Y_surv = Y_surv.loc[Y_info.tumor_normal=="Tumor"]
        # Drop rows where surv_variable or surv_variable_time is NA
        Y_surv.dropna(subset=[surv_variable, surv_variable_time], inplace=True)
        # create a discrete time class variable using the fixed-time point.
        time = 230
        Y_surv_disc = Y_surv[['PFI', 'PFI.time']].apply(
            lambda row: survival_fixed_time(time, row['PFI.time'], row['PFI']), axis=1)
        Y_surv_disc.dropna(inplace=True)
        # Load gene-exp vectors
        """
        # how to get this file
        df_gene_tree_map = pd.read_csv(path_to_pan_cancer_hdf5_files+"/KEGG_exp_to_tree_map.csv")
        # removing duplicate rows
        df_gene_tree_map.drop_duplicates(subset=['geneId'], inplace=True)
        # only keep pan can columns
        columns_to_drop = ["geneId","geneName","keggId","keggBriteId","Functional Annotation Group","Functional Annotation Subgroup","Functional Annotation","tamPixel","order"]
        df_gene_tree_map.drop(columns_to_drop, axis=1, inplace=True)
        # save 
        df_gene_tree_map.to_csv(path_to_pan_cancer_hdf5_files+"/KEGG_gene_exp.csv", index=False)
        """
        df_gene_exp = pd.read_csv(path_to_pan_cancer_hdf5_files+"/KEGG_gene_exp.csv")
        df_gene_exp.set_index('Unnamed: 0', inplace=True)
        # Remove NON_BRCA and BRCA suffix from the samples names
        df_gene_exp.index = [s.split('_')[0] for s in df_gene_exp.index]
        # Select samples with discrete time survival information associated
        df_gene_exp_disc = df_gene_exp.loc[[s.replace(".", ".") for s in Y_surv_disc.index]]
        # Convert discrete time survival numerical variables into binary variables
        Y_surv_disc_class = LabelEncoder().fit_transform(Y_surv_disc)
        
        self.vectors = torch.tensor(df_gene_exp_disc.values)
        self.labels = torch.tensor(Y_surv_disc_class)
        self.mean, self.std = self.get_mean_std()
        self.std = torch.where(self.std == 0,  torch.tensor([1.]).double(),  self.std)

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, ix):
        return ((self.vectors[ix]-self.mean)/self.std), self.labels[ix]
    def get_mean_std(self):
        return self.vectors.mean(dim=0), self.vectors.std(dim=0)

class TranscriptomicVectorsDatasetLung(Dataset):
    """
    Dataset for transcriptomic data seen as vectors (raw) for the
    TCGA Pan-Cancer gene-expression dataset and the TCGA Pan-Cancer curated clinical dataset (lung).
    Code widely inspired from https://github.com/guilopgar/GeneExpImgTL
    """

    def __init__(self, path_to_pan_cancer_hdf5_files):
        """
        :param path_to_pan_cancer_hdf5_files: path to hdf5 files (without final /) containing transcriptomic data (both lung and non lung)
        """
        # Define survival variable of interest
        surv_variable = "PFI"
        surv_variable_time = "PFI.time"
        # Load samples-info dataset
        Y_info_ft = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/Lung_pancan.h5", key="sample")
        # Load survival clinical outcome dataset
        Y_surv_ft = pd.read_hdf(path_to_pan_cancer_hdf5_files+"/Lung_pancan.h5", key="survival_outcome")
        # Filter tumor samples from survival clinical outcome dataset
        Y_surv_ft = Y_surv_ft.loc[Y_info_ft.tumor_normal=="Tumor"]
        # Drop rows where surv_variable or surv_variable_time is NA
        Y_surv_ft.dropna(subset=[surv_variable, surv_variable_time], inplace=True)
        time = 230
        Y_surv_disc_ft = Y_surv_ft[['PFI', 'PFI.time']].apply(
            lambda row: survival_fixed_time(time, row['PFI.time'], row['PFI']), axis=1)
        Y_surv_disc_ft.dropna(inplace=True)
        # Load gene-exp vectors
        """
        # how to get this file
        df_gene_tree_map = pd.read_csv(path_to_pan_cancer_hdf5_files+"/KEGG_exp_to_tree_map.csv")
        # removing duplicate rows
        df_gene_tree_map.drop_duplicates(subset=['geneId'], inplace=True)
        # only keep pan can columns
        columns_to_drop = ["geneId","geneName","keggId","keggBriteId","Functional Annotation Group","Functional Annotation Subgroup","Functional Annotation","tamPixel","order"]
        df_gene_tree_map.drop(columns_to_drop, axis=1, inplace=True)
        # save 
        df_gene_tree_map.to_csv(path_to_pan_cancer_hdf5_files+"/KEGG_gene_exp.csv", index=False)
        """
        df_gene_exp = pd.read_csv(path_to_pan_cancer_hdf5_files+"/KEGG_gene_exp.csv")
        df_gene_exp.set_index('Unnamed: 0', inplace=True)
        # Remove NON_BRCA and BRCA suffix from the samples names
        df_gene_exp.index = [s.split('_')[0] for s in df_gene_exp.index]
        # Select samples with discrete time survival information associated
        df_gene_exp_disc_ft = df_gene_exp.loc[[s.replace(".", ".") for s in Y_surv_disc_ft.index]]
        # Convert discrete time survival numerical variables into binary variables
        Y_surv_disc_class_ft = LabelEncoder().fit_transform(Y_surv_disc_ft)
        
        self.vectors = torch.tensor(df_gene_exp_disc_ft.values)
        self.labels = torch.tensor(Y_surv_disc_class_ft)
        self.mean, self.std = self.get_mean_std()
        self.std = torch.where(self.std == 0,  torch.tensor([1.]).double(),  self.std)

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, ix):
        return ((self.vectors[ix]-self.mean)/self.std), self.labels[ix]
    def get_mean_std(self):
        return self.vectors.mean(dim=0), self.vectors.std(dim=0)

def get_data_loaders(dataset, batch_size_train = 128, batch_size_validation = 128, no_batching_for_validation = False ,test_proportion = 0.33):
    """
    Function that returns train and validation dataloaders (balanced train).
    :param dataset: torch.utils.data.Dataset object.
    :param batch_size_train: batch size for the train loader.
    :param batch_size_validation: batch size for the test loader.
    :param no_batching_for_validation: if true then batch size for validation loader equals the length of the validation set (only one batch),
    otherwise use batch_size_validation.
    :param test_proportion: proportion of examples to use in the validation set.
    """
    
        
    # test train split
    len_validation = np.round(test_proportion*len(dataset))
    len_train = len(dataset)-len_validation
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, [int(len_train), int(len_validation)])
    if no_batching_for_validation :
        batch_size_validation = int(len_validation)
    # dataloaders
    # step 1: dealing with class imbalance
    weights_train = make_weights_for_balanced_classes(dataset_train, 2)
    sampler_train = torch.utils.data.WeightedRandomSampler(torch.tensor(weights_train).type('torch.DoubleTensor'), len(weights_train))
    # step 2: define the dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train,shuffle=False, sampler= sampler_train)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation,shuffle=False)
    return dataloader_train, dataloader_validation

def get_unblanced_data_loaders(dataset, batch_size_train = 128, batch_size_validation = 128, no_batching_for_validation = False ,test_proportion = 0.33):
    """
    Function that returns train and validation dataloaders (unbalanced train).
    :param dataset: torch.utils.data.Dataset object.
    :param batch_size_train: batch size for the train loader.
    :param batch_size_validation: batch size for the test loader.
    :param no_batching_for_validation: if true then batch size for validation loader equals the length of the validation set (only one batch),
    otherwise use batch_size_validation.
    :param test_proportion: proportion of examples to use in the validation set.
    """
    
        
    # test train split
    len_validation = np.round(test_proportion*len(dataset))
    len_train = len(dataset)-len_validation
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, [int(len_train), int(len_validation)])
    if no_batching_for_validation :
        batch_size_validation = int(len_validation)
    # dataloaders
    # step 2: define the dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train,shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation,shuffle=False)
    return dataloader_train, dataloader_validation
