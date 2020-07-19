import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from sklearn.model_selection import train_test_split,GroupKFold,GroupShuffleSplit
from sklearn.metrics import accuracy_score,r2_score,auc
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix
from torch_sparse import SparseTensor
from fastprogress import progress_bar,master_bar