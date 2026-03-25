################################################################
#
# MODULES
#
################################################################

import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale
import csv

import os
import glob
import random
#import time
from timeit import default_timer as timer
import argparse
import dill as pickle 

import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F

from reduction_leaky import reduction
from utils import debuginfoStr,loadGAS ,build_data, build_graph, build_graph_pruned
from sub_sample import sub_sample
from pyHGT.model import GNN, GNN_from_raw

from warnings import filterwarnings
import itertools
filterwarnings("ignore")

import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix



################################################################
#
# DEEPMAPS PREPARATION FOR HGT PERFORMANCE
#
################################################################

# Protection stocastic procedures
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('cuda version: ',torch.version.cuda) # Checking for cuda version

# Arguments
parser = argparse.ArgumentParser(description='Training GNN on gene cell graph')
parser.add_argument('--data_path', type=str)
parser.add_argument('--metadata_path', type=str)
parser.add_argument('--omic1_path', type=str)
parser.add_argument('--omic2_path', type=str)
parser.add_argument('--epoch', type=int, default=100)
# sampling times
parser.add_argument('--n_batch', type=int, default=25,
                    help='Number of batch (sampled graphs) for each epoch')

parser.add_argument('--omic1_rate', type=float, default=1)
parser.add_argument('--omic2_rate', type=float, default=1)

# Result
parser.add_argument('--data_name', type=str,
                    help='The name for dataset')
parser.add_argument('--result_dir', type=str,
                    help='The address for storing the models and optimization results.')
parser.add_argument('--reduction', type=str, default='AE',
                    help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256,
                    help='Number of hidden dimension (AE)')
parser.add_argument('--omic1_corr_cutoff', type=int, default=0.7,
                    help='Number of hidden dimension (AE)')
parser.add_argument('--omic2_corr_cutoff', type=int, default=0.7,
                    help='Number of hidden dimension (AE)')
                    
# GAE
parser.add_argument('--n_hid', type=int,
                    help='Number of hidden dimension (HGT)', default=128)
parser.add_argument('--ae_n_hid', type=int,
                    help='Number of hidden dimension (AE)', default=256)
parser.add_argument('--n_heads', type=int,
                    help='Number of attention head', default=8)
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout ratio')
parser.add_argument('--lr', type=float,
                    help='learning rate', default=0.0001)

parser.add_argument('--batch_size', type=int,
                    help='Number of output nodes for training', default=32)
parser.add_argument('--layer_type', type=str, default='hgt',
                    help='the layer type for GAE')
parser.add_argument('--loss', type=str, default='sup',
                    help='the loss for GAE')
parser.add_argument('--factor', type=float, default='0.5',
                    help='the attenuation factor')
parser.add_argument('--patience', type=int, default=5,
                    help='patience')
parser.add_argument('--rf', type=float, default='0.0',
                    help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=1,
                    help='cuda 0 use GPU0 else cpu ')
parser.add_argument('--rep', type=str, default='T',
                    help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1,
                    help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='optimizer')

args = parser.parse_args()
#args.metadata_path = "/home/inigo/Desktop/BiomiX_HGT/EGA/Metadata/EGAS00001001746_metadata_CLL.tsv"
#args.omic1_path = "/home/inigo/Desktop/BiomiX_HGT/EGA_omic1.csv"
#args.omic2_path = "/home/inigo/Desktop/BiomiX_HGT/EGA_omic2.csv"
#args.result_dir = "/home/inigo/Desktop/BiomiX_HGT/EGA_biomix_function"

#gene_cell = minmax_scale(pd.read_csv('AUCs/aHD1.csv').iloc[: , 1:].to_numpy(), axis=0) # Initial adjacency matrix is the one generated in KNN

#print(f'GAS loaded!',gene_cell.shape)
#args.dropout = 0.2
#args.epoch = 100 # Hyperparameter specification
#args.n_hid = 104 # Hyperparameter specification
#args.n_heads = 13 # Hyperparameter specification
#args.lr = 0.001 # Hyperparameter specification
#args.n_batch = 32 # Hyperparameter specification
file0=f'epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_{args.lr}n_batch{args.n_batch}' # Text saving hypermarametrization
print(f'\n{file0}') #Print hyperparametrization
args.metadata_path

#args.result_dir = 'Output_MTB' # Sets parent directory for output
gene_dir = args.result_dir+'/gene/' # Sets directory for gene output
sample_dir = args.result_dir+'/sample/' # Sets directory for cell output
metabo_dir = args.result_dir+'/metabo/' # Sets directory for cell output
model_dir = args.result_dir+'/model/' # Sets directory for model output
att_dir = args.result_dir+'/att/' # Sets directory for attention output
loss_dir = args.result_dir+'/loss/' # Sets directory for loss output
plots_dir = args.result_dir+'/plots/' # Sets directory for loss output
embbs_dir = args.result_dir+'/embeddings/' # Sets directory for loss output
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(gene_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(metabo_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(att_dir , exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(embbs_dir, exist_ok=True)

#start_time = time.time() # Initialize time counting
#print('---0:00:00---scRNA starts.') # Prints the initialization 

# Setting working space GPU or CPU
if args.cuda == 0:
    device = torch.device("cuda:" + "0") # Sets space to GPU
    print("cuda>>>")
else:
    device = torch.device("cpu") # Sets space to CPU

print(device) # Print device. IMPORTANT for incompatibilities.



################################################################
#
# METADATA PROCESSING BEFORE HGT
#
################################################################

#metadata = pd.read_csv("/home/inigo/Desktop/BiomiX_HGT/MTB/Metadata/MTBLS7623_Metadata.tsv", sep="	")
metadata = pd.read_csv(args.metadata_path, sep="	")
#metadata = metadata[metadata["CONDITION"] != "PTB_DM"].reset_index(drop=True)
sample_order = metadata.ID.tolist()
#trans = trans[['ID'] + sample_order]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample input (your real metadata)
labels = metadata['CONDITION']

# Step 1: Label encode to integers
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
# This will map: HC → 0, PTB → 1, PTB_DM → 2

# Step 2: One-hot encode
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))



################################################################
#
# HGT INFERENCE
#
################################################################

omic_1_df = pd.read_csv(args.omic1_path)
omic_1_df = omic_1_df.loc[:, ~omic_1_df.columns.str.contains("^Unnamed")]
#omic_1_df = omic_1_df[['ID'] + sample_order]
#omic_1 = np.array(omic_1_df.iloc[:,1:])  # your raw count matrix
omic_2_df = pd.read_csv(args.omic2_path)
omic_2_df = omic_2_df.loc[:, ~omic_2_df.columns.str.contains("^Unnamed")]
#omic_2_df = omic_2_df[['ID'] + sample_order]
#omic_2 = np.array(omic_2_df.iloc[:,1:])  # your raw count matrix

meta_samples = set(metadata["ID"])
omic_1_samples = set(omic_1_df.columns[1:])  # exclude "ID" column
omic_2_samples = set(omic_2_df.columns[1:])
common_samples = list(meta_samples & omic_1_samples & omic_2_samples)
metadata = metadata[metadata["ID"].isin(common_samples)].reset_index(drop=True)
omic_1_df = omic_1_df[["ID"] + common_samples]
omic_1 = np.array(omic_1_df.iloc[:,1:])  # your raw count matrix
omic_2_df = omic_2_df[["ID"] + common_samples]
omic_2 = np.array(omic_2_df.iloc[:,1:])  # your raw count matrix
metadata = metadata.set_index("ID").loc[common_samples].reset_index()

labels = metadata['CONDITION']
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))
#trans = trans[["ID"] + common_samples]
#metabo = metabo[["ID"] + common_samples]
#h_n = 256
#encoded_trans,encoded2_trans, losses, losses2 = reduction(args.reduction,trans_log_normalized,device,h_n) # Autoencoder for transcriptomics
encoded_trans, encoded2_trans, losses, losses2 = reduction(args.reduction, omic_1, device, args.ae_n_hid)
encoded_metabo,encoded2_metabo, losses, losses2 = reduction(args.reduction, omic_2, device,args.ae_n_hid) # Autoencoder for metabolomics
samples = np.concatenate((omic_1,omic_2),axis=0)
encoded_samples,encoded2_samples, losses, losses2 = reduction(args.reduction,samples,device,args.ae_n_hid) # Autoencoder for samples

def normalize_embeddings(X):
    # Normalize each row (embedding vector) to zero mean, unit variance
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


# Step 1: Concatenate (N x 256 → N x 768)
combined = torch.cat([encoded_trans, encoded_metabo, encoded2_samples], dim=0)  # Shape: (N, 768)
normalized = normalize_embeddings(combined)

encoded_trans = normalized[0:encoded_trans.shape[0],:]
encoded_metabo = normalized[encoded_trans.shape[0]:(encoded_trans.shape[0]+encoded_metabo.shape[0]),:]
encoded2_sample = normalized[(encoded_trans.shape[0]+encoded_metabo.shape[0]):normalized.shape[0],:]


pd.DataFrame(encoded_trans.detach().numpy()).to_csv(os.path.join(embbs_dir,"ae_omic1_embeddings.csv"))
pd.DataFrame(encoded_metabo.detach().numpy()).to_csv(os.path.join(embbs_dir,"ae_omic2_embeddings.csv"))
pd.DataFrame(encoded2_samples.detach().numpy()).to_csv(os.path.join(embbs_dir,"ae_samples_embeddings.csv"))
debuginfoStr('Feature extraction finished') # Print verbose


#def normalize_embeddings(X):
#    # Normalize each row (embedding vector) to zero mean, unit variance
#    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def compute_binary_correlation_mask(A, B, ae_n_hid, threshold=0.7):
    # Normalize embeddings for Pearson correlation
    #A_norm = normalize_embeddings(A)
    #B_norm = normalize_embeddings(B)
    A_norm = A
    B_norm = B
    # Pearson correlation = cosine similarity of standardized vectors
    corr = np.dot(A_norm, B_norm.T) / ae_n_hid  # 256 is the dimension
    return (corr >= threshold).astype(np.uint8)  # Binary mask: 1 if corr >= 0.7


# A, B, C are numpy arrays of shape (N1, 256), (N2, 256), and (N3, 256)
# binary masks for each pair
mask_trans = compute_binary_correlation_mask(encoded_trans.detach().numpy(), encoded_trans.detach().numpy(), args.ae_n_hid, threshold=args.omic1_corr_cutoff)
np.fill_diagonal(mask_trans, 0)
mask_metabo = compute_binary_correlation_mask(encoded_metabo.detach().numpy(), encoded_metabo.detach().numpy(),args.ae_n_hid, threshold=args.omic2_corr_cutoff)
np.fill_diagonal(mask_metabo, 0)
mask_sample = np.zeros([encoded2_samples.shape[0],encoded2_samples.shape[0]])

# NO MULTIOMIC CORRELATIONS - SEPARATE AEs ARTIFACTS
from sklearn.neighbors import NearestNeighbors
import numpy as np

def compute_knn_mask(A, B, k=100, metric='cosine'):
    """
    For each row in A, find k nearest neighbors in B.
    Returns a binary matrix of shape (A.shape[0], B.shape[0]).
    """
    neigh = NearestNeighbors(n_neighbors=k, metric=metric)
    neigh.fit(B)
    dists, indices = neigh.kneighbors(A)
    mask = np.zeros((A.shape[0], B.shape[0]), dtype=np.uint8)
    for i, neighbors in enumerate(indices):
        mask[i, neighbors] = 1
    return mask

# k = 100 nearest neighbors
mask_trans_metabo = compute_knn_mask(encoded_trans.detach().numpy(), encoded_metabo.detach().numpy(), k=round(encoded_metabo.shape[0]/3))
mask_trans_sample = compute_knn_mask(encoded_trans.detach().numpy(), encoded2_samples.detach().numpy(), k=round(encoded2_samples.shape[0]/3))
mask_metabo_samples = compute_knn_mask(encoded_metabo.detach().numpy(), encoded2_samples.detach().numpy(), k=round(encoded2_samples.shape[0]/3))
np.fill_diagonal(mask_trans_metabo, 0)
np.fill_diagonal(mask_trans_sample, 0)
np.fill_diagonal(mask_metabo_samples, 0)

print(f'genes connect: {np.sum(mask_trans)}, metabo connect: {np.sum(mask_metabo)}, samples connect: {np.sum(mask_sample)}, genes-metabo connect: {np.sum(mask_trans_metabo)}, genes-samples connect: {np.sum(mask_trans_sample)}, mask_metabo_samples: {np.sum(mask_metabo_samples)}')



################################################################
#
# GENERATE HETEROGENOUS GRAPH
#
################################################################

graph = build_graph(mask_trans_metabo,mask_trans_sample,mask_metabo_samples,mask_trans,mask_metabo,mask_sample, encoded_trans,encoded_metabo,encoded2_samples) # Generate graph based in autoencoder and KNN matrix
debuginfoStr('Build Graph finished') # Print verbose



################################################################
#   
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################

#start = timer()
print("Start sampling!")
np.random.seed(seed)
jobs = []
#args.gene_rate= 1
#args.metabo_rate = 1
#args.sample_rate=1
#sample_num=int((encoded2_samples.shape[0]*args.sample_rate)/args.n_batch) # Sets the number of cells in subgraph
sample_num=round(encoded2_samples.shape[0]/3)
gene_num=int((encoded_trans.shape[0]*args.omic1_rate)/args.n_batch) # Sets the number of TFs in subrgraph
metabo_num=int((encoded_metabo.shape[0]*args.omic2_rate)/args.n_batch) # Sets the number of TFs in subrgraph
print(f'sample_num: {sample_num}, gene_num: {gene_num}, metabo_num: {metabo_num}')
for _ in range(args.n_batch): # Iterates over subsampling
    p = sub_sample(graph,
                    mask_trans_sample,
                    mask_metabo_samples,
                    sample_num,
                    gene_num,
                    metabo_num,
                    encoded_trans.shape[0],
                    encoded_metabo.shape[0],
                    encoded2_samples.shape[0]) # Sub-graphs sampling as jobs to learn from in HGT
    jobs.append(p)
    

print("Sampling end!")
debuginfoStr('Cell Graph constructed and pruned')

# Split jobs into train, validation, and test sets
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
n_jobs = len(jobs)
n_train = int(n_jobs * train_ratio)
n_val = int(n_jobs * val_ratio)
n_test = n_jobs - n_train - n_val
random.shuffle(jobs)
train_jobs = jobs[:n_train]
val_jobs = jobs[n_train:n_train + n_val]
test_jobs = jobs[n_train + n_val:]
print(f"Train jobs: {len(train_jobs)}, Val jobs: {len(val_jobs)}, Test jobs: {len(test_jobs)}")



################################################################
#   
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################
#args.n_heads = 8
#args.n_hid = 128  # Ensure divisibility
gnn = GNN(conv_name=args.layer_type, in_dim=256,
                n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
                num_types=3, num_relations=13, use_RTE=False, n_labels=one_hot_labels.shape[1] 
                ).to(device) # Loads HGT function
#gnn = GNN_class(conv_name=args.layer_type, in_dim=256,
#                n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#                num_types=2, num_relations=6, use_RTE=False, 
#                h_out=patient_loss.shape[1]
#                ).to(device) # Loads HGT function
# Options to use different optimization algorithms. Default: adamw
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(gnn.parameters(), lr=args.lr)
    

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=args.factor, patience=args.patience, verbose=True)
    
    

################################################################
#
# TRAIN HGT IN SUB-SAMPLING
#
################################################################


start = timer()
gnn.train() # The GNN arquitecture. It shows the layers and dimensions.
training_loss = []
kl_gene_losses = []
kl_metabo_losses = []
cross_entropy_losses = []
cosine_losses = []
total_losses = []
val_losses = []

cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.5, reduction='sum').to(device)  # Initialize CosineEmbeddingLoss

patience = 50
patience_counter = 0
best_val_loss = float('inf')
best_model_state = None

for epoch in np.arange(args.epoch): # Iterates over the epochs. Number set to 100.
    L = 0 # Initialization
    #check = 0
    gnn.train()
    for job in train_jobs: # Iterates over the graph-subsampling generated in the preprocessing
        #print(check)
        feature,time,edge_list,indxs = job # Initialize the job. Sets feature (cells and genes), time (time dealing in transformers), edges (connections) and indeces (no idea).
        node_dict = {} # Initialization
        node_feature = [] # Initialization
        node_type = [] # Initialization
        node_time = [] # Initialization
        edge_index = [] # Initialization
        edge_type = [] # Initialization
        edge_time = [] # Initialization
        
        node_num = 0 # Initialization
        types = graph.get_types()   # ['gene','cell']
        for t in types: # Iterates over genes and cells
            node_dict[t] = [node_num, len(node_dict)] # Creates a dictionary of cells and genes
            node_num += len(feature[t])
                # node_dict: {'gene':[0,0],'cell':[134,1]}
                 
        for t in types: # Iterates over genes and cellsc
            t_i = node_dict[t][1]
            node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(device)) # Saves cells or genes matrices  
            node_time += list(time[t]) # Saves node time
            node_type += [node_dict[t][1] for _ in range(len(feature[t]))] # Saves node type
             
        edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())} # Reference dictionary of connections
        edge_dict['self'] = len(edge_dict) # Incoporates self lenght parameter to edge dictionary
        # {'g_c': 0, 'rev_g_c': 1 ,'self': 2}
        for target_type in edge_list:
            for source_type in edge_list[target_type]:
                for relation_type in edge_list[target_type][source_type]:
                    for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                        tid, sid = ti + \
                            node_dict[target_type][0], si + \
                            node_dict[source_type][0]
                        edge_index += [[sid, tid]]
                        edge_type += [edge_dict[relation_type]]
                        edge_time += [120]
                            
                        
                    
                
        node_feature_tensor = torch.cat((node_feature[0],node_feature[1]),0) # Nodes features matrix in tensor format
        node_feature = torch.cat((node_feature_tensor, node_feature[2]),0)
        node_type = torch.LongTensor(node_type) # Nodes type matrix in tensor format
        edge_time = torch.LongTensor(edge_time) # Edges time matrix in tensor format (all the same)
        edge_index = torch.LongTensor(edge_index).t() # Edges indexes (nodes-nodes connections) in tensor format
        #edge_index[1][np.array(edge_type) == edge_dict['g_g']] = edge_index[1][np.array(edge_type) == edge_dict['g_g']]-node_dict['sample'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_g_g']] = edge_index[0][np.array(edge_type) == edge_dict['rev_g_g']]-node_dict['sample'][0]
        #edge_index[1][np.array(edge_type) == edge_dict['s_s']] = edge_index[1][np.array(edge_type) == edge_dict['s_s']]+node_dict['sample'][0]+node_dict['metabolite'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_s_s']] = edge_index[0][np.array(edge_type) == edge_dict['rev_s_s']]+node_dict['sample'][0]+node_dict['metabolite'][0]
        #edge_index[1][np.array(edge_type) == edge_dict['m_m']] = edge_index[1][np.array(edge_type) == edge_dict['m_m']]+node_dict['metabolite'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_m_m']] = edge_index[0][np.array(edge_type) == edge_dict['rev_m_m']]+node_dict['metabolite'][0]
        edge_type = torch.LongTensor(edge_type) # Edges type in tensor format
        #node_rep, cell_class = gnn.forward(node_feature, 
        node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(node_feature,node_type.to(device),edge_time.to(device),edge_index.to(device),edge_type.to(device)) # Applies HGT to obtain nodes representations
            #if args.rep == 'T':
                #node_rep = torch.trunc(node_rep*10000000000)/10000000000 # NO IDEA.
                #if args.reduction == 'raw':
                #    for t in types:
                #        t_i = node_dict[t][1]
                #       print("t_i="+str(t_i))
                #        node_decoded_embedding[t_i] = torch.trunc(
                #            node_decoded_embedding[t_i]*10000000000)/10000000000
        gene_matrix = node_rep[node_type == 0, ] # Extracts gene nodes representations
        metabo_matrix = node_rep[node_type == 1, ] # Extracts cells nodes representations
        sample_matrix = node_rep[node_type == 2, ]
        regularization_loss = 0 # Initialization of regularization loss
        for param in gnn.parameters():
            regularization_loss += torch.sum(torch.pow(param, 2)) # Saves the regularization loss over parameters
                
        if (args.loss == "sup"): # No other option
            #decoder = torch.mm(gene_matrix, cell_matrix.t())
            #decoder_2 = torch.mm(cell_matrix, gene_matrix.t())
            decoder_g = torch.mm(sample_matrix, gene_matrix.t())
            decoder_m = torch.mm(sample_matrix, metabo_matrix.t())
            adj = samples.T[indxs['sample'], ] # Extracts genes indeces 
            indexes = np.concatenate([indxs['gene'],indxs['metabolite']+encoded_trans.shape[0]])
            adj_g = adj[:, indxs['gene']]
            adj_m = adj[:, indxs['metabolite']]
            adj = adj[:, indexes] # Extracts cells indeces
            adj = torch.tensor(adj, dtype=torch.float32).to(device) # Sets an adjacency matrix
            adj_g = torch.tensor(adj_g, dtype=torch.float32).to(device)
            adj_m = torch.tensor(adj_m, dtype=torch.float32).to(device)
            samples_labels = one_hot_labels[indxs['sample'], ]
            #samples_labels = one_hot_labels[node_type == 2, ]
            #cells_labels = patient_loss[indxs['cell'],:]
            if args.reduction == 'raw':
                if epoch % 2 == 0:
                    loss = F.kl_div(decoder.softmax(
                        dim=-1).log(), adj.softmax(dim=-1), reduction='sum')+args.rf*regularization_loss
                else:
                    loss = nn.MSELoss()(
                        node_feature[0], node_decoded_embedding[0])+args.rf*regularization_loss
                    for t_i in range(1, len(types)):
                        loss += nn.MSELoss()(node_feature[t_i],
                                                    node_decoded_embedding[t_i])
                            
                        
            else:
                #loss = F.kl_div(decoder_g.softmax(dim=-1).log(), adj_g.softmax(dim=-1), reduction='sum')
                #loss += F.kl_div(decoder_m.softmax(dim=-1).log(),adj_m.softmax(dim=-1), reduction='sum') # Calculates loss function based in KL divergence
                #loss += F.cross_entropy(class_rep.softmax(dim=-1),torch.FloatTensor(samples_labels).to(device))
                kl_gene = F.kl_div(decoder_g.softmax(dim=-1).log(), adj_g.softmax(dim=-1), reduction='sum')
                kl_metabo = F.kl_div(decoder_m.softmax(dim=-1).log(), adj_m.softmax(dim=-1), reduction='sum')
                ce_loss = F.cross_entropy(class_rep,torch.FloatTensor(samples_labels).to(device))
                #loss = 0.2*kl_gene + 0.2*kl_metabo + 0.6*ce_loss
                #loss = kl_gene + kl_metabo + ce_loss
                #loss += F.kl_div(cell_class.softmax(dim=-1).log(),torch.FloatTensor(cells_labels).to(device).softmax(dim=-1), reduction='sum')
                # Compute CosineEmbeddingLoss for sample-sample pairs
                num_samples = sample_matrix.shape[0]
                max_pairs = 1000  # Limit pairs to avoid O(n^2) computation
                sample_idx_i = []
                sample_idx_j = []
                targets = []
                
                # Assume samples_labels is one-hot; convert to class indices
                sample_classes = torch.argmax(torch.LongTensor(samples_labels), dim=1)
                
                # Sample positive pairs (same class)
                for cls in torch.unique(sample_classes):
                    cls_idx = torch.where(sample_classes == cls)[0]
                    if len(cls_idx) >= 2:  # Need at least 2 samples for pairs
                        pairs = torch.combinations(cls_idx, r=2)
                        sample_idx_i.extend(pairs[:, 0].tolist())
                        sample_idx_j.extend(pairs[:, 1].tolist())
                        targets.extend([1] * len(pairs))
                    
                # Sample negative pairs (different classes)
                for i in range(num_samples):
                    for j in range(i + 1, num_samples):
                        if sample_classes[i] != sample_classes[j]:
                            sample_idx_i.append(i)
                            sample_idx_j.append(j)
                            targets.append(-1)
                
                # Limit total pairs
                if len(sample_idx_i) > max_pairs:
                    idx = torch.randperm(len(sample_idx_i))[:max_pairs]
                    sample_idx_i = [sample_idx_i[i] for i in idx]
                    sample_idx_j = [sample_idx_j[i] for i in idx]
                    targets = [targets[i] for i in idx]
                    
                if len(sample_idx_i) > 0:
                    sample_emb_i = sample_matrix[sample_idx_i]
                    sample_emb_j = sample_matrix[sample_idx_j]
                    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                    cosine_loss = cosine_loss_fn(sample_emb_i, sample_emb_j, target_tensor)
                else:
                    cosine_loss = torch.tensor(0.0, device=device)
                    
                #loss = 0.1 * kl_gene + 0.1 * kl_metabo + 0.4 * ce_loss + 0.4 * cosine_loss
                loss = 0.3 * ce_loss + 0.7 * cosine_loss
                
        L += loss.item() # Saves loss 
        optimizer.zero_grad() # Initializies optimizer
        loss.backward() # Sets backpropagation to loss function
        optimizer.step() # Sets a step in optimizer
        #check += 1
        
    scheduler.step(L/(int(len(train_jobs))))
        # Validation phase
    gnn.eval()
    val_L = 0
    with torch.no_grad():
        for job in val_jobs:
            feature, time, edge_list, indxs = job
            node_dict = {}
            node_feature = []
            node_type = []
            node_time = []
            edge_index = []
            edge_type = []
            edge_time = []
            
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
                
            for t in types:
                t_i = node_dict[t][1]
                node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(device))
                node_time += list(time[t])
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
                
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            for target_type in edge_list:
                for source_type in edge_list[target_type]:
                    for relation_type in edge_list[target_type][source_type]:
                        for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                            tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                            edge_index += [[sid, tid]]
                            edge_type += [edge_dict[relation_type]]
                            edge_time += [120]
                            
            node_feature_tensor = torch.cat((node_feature[0], node_feature[1]), 0)
            node_feature = torch.cat((node_feature_tensor, node_feature[2]), 0)
            node_type = torch.LongTensor(node_type).to(device)
            edge_time = torch.LongTensor(edge_time).to(device)
            edge_index = torch.LongTensor(edge_index).t().to(device)
            edge_type = torch.LongTensor(edge_type).to(device)
            node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(
                node_feature, node_type, edge_time, edge_index, edge_type
            )
            gene_matrix = node_rep[node_type == 0, :]
            metabo_matrix = node_rep[node_type == 1, :]
            sample_matrix = node_rep[node_type == 2, :]
            regularization_loss = 0
            for param in gnn.parameters():
                regularization_loss += torch.sum(torch.pow(param, 2))
                
            if args.loss == "kl":
                decoder_g = torch.mm(sample_matrix, gene_matrix.t())
                decoder_m = torch.mm(sample_matrix, metabo_matrix.t())
                adj = samples.T[indxs['sample'], :]
                indexes = np.concatenate([indxs['gene'], indxs['metabolite'] + encoded_trans.shape[0]])
                adj_g = adj[:, indxs['gene']]
                adj_m = adj[:, indxs['metabolite']]
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
                adj_g = torch.tensor(adj_g, dtype=torch.float32).to(device)
                adj_m = torch.tensor(adj_m, dtype=torch.float32).to(device)
                samples_labels = one_hot_labels[indxs['sample'], :]
                if args.reduction == 'raw':
                    if epoch % 2 == 0:
                        val_loss = F.kl_div(decoder.softmax(dim=-1).log(), adj.softmax(dim=-1), reduction='sum') + args.rf * regularization_loss
                    else:
                        val_loss = nn.MSELoss()(node_feature[0], node_decoded_embedding[0]) + args.rf * regularization_loss
                        for t_i in range(1, len(types)):
                            val_loss += nn.MSELoss()(node_feature[t_i], node_decoded_embedding[t_i])
                else:
                    kl_gene = F.kl_div(decoder_g.softmax(dim=-1).log(), adj_g.softmax(dim=-1), reduction='sum')
                    kl_metabo = F.kl_div(decoder_m.softmax(dim=-1).log(), adj_m.softmax(dim=-1), reduction='sum')
                    ce_loss = F.cross_entropy(class_rep, torch.FloatTensor(samples_labels).to(device))
                    num_samples = sample_matrix.shape[0]
                    max_pairs = 1000
                    sample_idx_i = []
                    sample_idx_j = []
                    targets = []
                    sample_classes = torch.argmax(torch.FloatTensor(samples_labels).to(device), dim=1)
                    for cls in torch.unique(sample_classes):
                        cls_idx = torch.where(sample_classes == cls)[0]
                        if len(cls_idx) >= 2:
                            pairs = torch.combinations(cls_idx, r=2)
                            sample_idx_i.extend(pairs[:, 0].tolist())
                            sample_idx_j.extend(pairs[:, 1].tolist())
                            targets.extend([1] * len(pairs))
                            
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            if sample_classes[i] != sample_classes[j]:
                                sample_idx_i.append(i)
                                sample_idx_j.append(j)
                                targets.append(-1)
                                
                    if len(sample_idx_i) > max_pairs:
                        idx = torch.randperm(len(sample_idx_i))[:max_pairs]
                        sample_idx_i = [sample_idx_i[i] for i in idx]
                        sample_idx_j = [sample_idx_j[i] for i in idx]
                        targets = [targets[i] for i in idx]
                        
                    if len(sample_idx_i) > 0:
                        sample_emb_i = sample_matrix[sample_idx_i]
                        sample_emb_j = sample_matrix[sample_idx_j]
                        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                        cosine_loss = cosine_loss_fn(sample_emb_i, sample_emb_j, target_tensor)
                    else:
                        cosine_loss = torch.tensor(0.0, device=device)
                        
                    #val_loss = 0.2 * kl_gene + 0.2 * kl_metabo + 0.3 * ce_loss + 0.3 * cosine_loss
                    val_loss = 0.3 * ce_loss + 0.7 * cosine_loss
                    
                val_L += val_loss.item()
    #print('Epoch :', epoch+1, '|', 'train_loss:%.12f' % (L/(int(len(jobs)))/args.n_batch)) # Print loss in epoch
    #print('Epoch :', epoch+1, '|', 'train_loss:%.12f' % (L/(int(len(jobs))))) # Print loss in epoch
    #val_loss=1
    printed_val_loss = val_L/(int(len(val_jobs)))
    printed_train_loss = L/(int(len(train_jobs)))
    print(f'Epoch: {epoch+1} | Train Loss: {printed_train_loss:.12f}\nEpoch: {epoch+1} | Val Loss: {printed_val_loss:.12f}')
    #training_loss.append((L/(int(samples.shape[0]))/args.n_batch))
    kl_gene_losses.append(kl_gene.item())
    kl_metabo_losses.append(kl_metabo.item())
    cross_entropy_losses.append(ce_loss.item())
    cosine_losses.append(cosine_loss.item())  # Track cosine loss
    total_losses.append(loss.item())
    val_losses.append(printed_val_loss)
    # Early stopping checks
    #if cosine_loss.item() < 1e-6 and ce_loss.item() < 1e-6:
    #    patience_counter += 1
    #    print(f'Early stopping condition: cosine_loss ({cosine_loss.item():.6f}) and ce_loss ({ce_loss.item():.6f}) near zero')
    #elif val_loss > loss:
    #    patience_counter += 1
    #    print(f'Early stopping condition: val_loss ({val_loss:.6f}) > train_loss ({loss:.6f})')
    #else:
    #    patience_counter = 0
    #    best_val_loss = min(best_val_loss, val_loss)
    #    best_model_state = gnn.state_dict()
    #    state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
    #    'epoch': epoch} # Saves a dictionary of model
    #    model0=f'BiomiX_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
    #    torch.save(state, model_dir+model0) # Saves model in folder
    #    
    #if patience_counter >= patience:
    #    print(f'Early stopping triggered after {patience} epochs of no improvement')
    #    gnn.load_state_dict(best_model_state)
    #    break
    

    
    



end = timer()
print(f"BiomiX-HGT took {round((end - start)/60,2)} mins to train")
unique_vals, counts = torch.unique(edge_type, return_counts=True) # Count value repetitions
count_dict = dict(zip(unique_vals.tolist(), counts.tolist())) # Combine into a dictionary (optional)
print(f"Edge dict is:\n{edge_dict}\nAnd count dict is:\n{count_dict}")

#import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(total_losses, label='Total Loss', linewidth=2)
#plt.plot(kl_gene_losses, label='KL Gene')
#plt.plot(kl_metabo_losses, label='KL Metabolite')
plt.plot(cosine_losses, label='Cosine Loss')
plt.plot(cross_entropy_losses, label='Cross Entropy')
plt.plot(val_losses, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Components Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig(plots_dir+"training_loss_components.pdf", format='pdf')
plt.close()


state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
        'epoch': epoch} # Saves a dictionary of model
model0=f'BiomiX_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}' 
torch.save(state, model_dir+model0) # Saves model in folder
pd.DataFrame(total_losses).to_csv(loss_dir+model0+"_total_losses.csv")
#pd.DataFrame(kl_gene_losses).to_csv(loss_dir+model0+"_kl_gene_losses.csv")
#pd.DataFrame(kl_metabo_losses).to_csv(loss_dir+model0+"_kl_metabo_losses.csv")
pd.DataFrame(cosine_losses).to_csv(loss_dir+model0+"_cosine_losses.csv")
pd.DataFrame(cross_entropy_losses).to_csv(loss_dir+model0+"_cross_entropy_losses.csv")
pd.DataFrame(val_losses).to_csv(loss_dir+model0+"_val_losses.csv")
debuginfoStr('Graph Autoencoder training finished')

print(class_rep)
print(torch.FloatTensor(samples_labels))



################################################################
#
# APPLY HGT TO WHOLE GRAPH
#   
################################################################

# Load trained model
gnn.load_state_dict(state['model'])
gnn.eval()

# Determine batch size
sample_len = samples.T.shape[0]
if sample_len > 10000:
    ba = 500
else:
    ba = sample_len
    

#if (gene_cell.shape[1]>10000):
#        
#        if (gene_cell.shape[0]>10000):
#            ba = 500
#        else:
#            ba = gene_cell.shape[0]
#    else:
#        if (gene_cell.shape[0]>10000):
#            ba = 5000
#        else:
#            ba = gene_cell.shape[0] # Adapts batch to data dimensions

sample_embedding = []
gene_embedding = []
metabo_embedding = []
attention = []
saliency = []
g32_embeddings = []
g64_embeddings = []
g128_embeddings = []
class_embeddings = []

#with torch.no_grad():
#with gnn.eval():
for i in range(0, sample_len, ba):
        # Sub-batch slice
    sample_batch = samples.T[i:i+ba]
        # Plotting
        #encoded_sample_batch = encoded2_samples[i:i+ba]
        
        # Recompute masks for the batch
        #mask_trans_sample_batch = compute_knn_mask(encoded_trans.cpu().numpy(), encoded_sample_batch.cpu().numpy(), k=100)
        #mask_metabo_samples_batch = compute_knn_mask(encoded_metabo.cpu().numpy(), encoded_sample_batch.cpu().numpy(), k=100)
        
        # Build full graph from fixed gene/metabo, and batch of samples
        #graph = build_graph(
        #    mask_trans_metabo=mask_trans_metabo,
        #    mask_trans_sample=mask_trans_sample[:,i:i+ba],
        #    mask_metabo_samples=mask_metabo_samples[:,i:i+ba],
        #    mask_trans=mask_trans,
        #    mask_metabo=mask_metabo,
        #    mask_sample=mask_sample[i:i+ba, i:i+ba],  # sub-mask for current batch
        #    encoded_trans=encoded_trans,
        #    encoded_metabo=encoded_metabo,
        #    encoded2_samples=encoded2_samples[i:i+ba,:]
        #)
        
    x,node_type, edge_time, edge_index,edge_type = build_data(
        mask_trans_metabo=mask_trans_metabo,
        mask_trans_sample=mask_trans_sample[:,i:i+ba],
        mask_metabo_samples=mask_metabo_samples[:,i:i+ba],
        mask_trans=mask_trans,
        mask_metabo=mask_metabo,
        mask_sample=mask_sample[i:i+ba, i:i+ba],  # sub-mask for current batch
        encoded_trans=encoded_trans,
        encoded_metabo=encoded_metabo,
        encoded2_samples=encoded2_samples[i:i+ba,:],
        edge_dict= edge_dict
    )
    
    gnn.eval() 
    x_tensor = torch.cat((x['gene'],x['metabolite']),0) # Nodes features matrix in tensor format
    x = torch.cat((x_tensor, x['sample']),0)
    x.requires_grad = True
    node_type = torch.LongTensor(node_type) # Nodes type matrix in tensor format
    node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(x, 
                                    node_type.to(device),edge_time.to(device), 
                                    edge_index.to(device), edge_type.to(device)) # Applies HGT
    #gene_name = gene_name + list(np.array(edge_index[0]+i)) # Saves gene results
    #cell_name = cell_name + list(np.array(edge_index[1]-adj.shape[0])) # Saves cells results
    attention.append(gnn.att.detach().numpy()) # Saves attention
    gene_matrix = node_rep[node_type == 0, ] # Extracts gene nodes representations
    metabo_matrix = node_rep[node_type == 1, ] # Extracts cells nodes representations
    sample_matrix = node_rep[node_type == 2, ]
    gene_embedding.append(gene_matrix.detach().numpy())
    metabo_embedding.append(metabo_matrix.detach().numpy())
    sample_embedding.append(sample_matrix.detach().numpy())
    # Choose a scalar objective for gradients
    #score = class_rep.norm()  # or class_rep.mean() or a specific class prediction
    #score.backward()
    target = node_rep.mean() + class_rep.mean()
    target.backward()
    # Grab gradients
    #saliency.append(x.grad.data.abs())  # [N, F] gradient magnitude
    saliency.append(x.grad.abs().cpu().numpy())
    #s_embedding.append(gene_matrix)
    g32_embeddings.append(g_32.detach().numpy())
    g64_embeddings.append(g_64.detach().numpy())
    g128_embeddings.append(g_128.detach().numpy())
    class_embeddings.append(class_rep.detach().numpy())
                    



# SAVING RESULTS
if samples.shape[1] % ba == 0:
    gene_matrix = np.vstack(gene_embedding[0:int(samples.shape[1]/ba)])
    metabo_matrix = np.vstack(metabo_embedding[0:int(samples.shape[1]/ba)])
    sample_matrix = np.vstack(sample_embedding[0:int(samples.shape[1]/ba)])
    attention = np.vstack(attention[0:int(samples.shape[1]/ba)])
    saliency = np.vstack(saliency[0:int(samples.shape[1]/ba)])
    g32_embeddings = np.vstack(g32_embeddings[0:int(samples.shape[1]/ba)])
    g64_embeddings = np.vstack(g64_embeddings[0:int(samples.shape[1]/ba)])
    g128_embeddings = np.vstack(g128_embeddings[0:int(samples.shape[1]/ba)])
    class_embeddings = np.vstack(class_embeddings[0:int(samples.shape[1]/ba)])
    #gene_matrix = gene_matrix
    #metabo_matrix = metabo_matrix
    #sample_matrix = sample_matrix
    #attention = gnn.att
else:
    gene_tensor = np.vstack(gene_embedding[0:int(samples.shape[1]/ba)])
    gene_matrix = np.concatenate((gene_tensor, gene_matrix), 0)
    final_attention = np.vstack(attention[0:int(samples.shape[1]/ba)])
    attention = np.concatenate((final_attention, gnn.att), 0)
    metabo_tensor = np.vstack(metabo_embedding[0:int(samples.shape[1]/ba)])
    metabo_matrix = np.concatenate((metabo_tensor, gene_matrix), 0)
    sample_tensor = np.vstack(sample_embedding[0:int(samples.shape[1]/ba)])
    sample_matrix = np.concatenate((sample_tensor, gene_matrix), 0)
    final_saliency = np.vstack(saliency[0:int(samples.shape[1]/ba)])
    saliency = np.concatenate((final_saliency, saliency), 0)
    final_g32_embeddings = np.vstack(g32_embeddings[0:int(samples.shape[1]/ba)])
    g32_embeddings = np.concatenate((final_g32_embeddings, g32_embeddings), 0)
    final_g64_embeddings = np.vstack(g64_embeddings[0:int(samples.shape[1]/ba)])
    g64_embeddings = np.concatenate((final_g64_embeddings, g64_embeddings), 0)
    final_g128_embeddings = np.vstack(g128_embeddings[0:int(samples.shape[1]/ba)])
    g128_embeddings = np.concatenate((final_g128_embeddings, g128_embeddings), 0)
    final_class_embeddings = np.vstack(class_embeddings[0:int(samples.shape[1]/ba)])
    class_embeddings = np.concatenate((final_class_embeddings, class_embeddings), 0)
    

#cell_matrix = cell_matrix.detach().numpy()
np.savetxt(gene_dir+file0, gene_matrix, delimiter=' ')
np.savetxt(metabo_dir+file0, metabo_matrix, delimiter=' ')
np.savetxt(sample_dir+file0, sample_matrix, delimiter=' ')
#np.savetxt(embeddings_dir+file0, sample_matrix, delimiter=' ')
positions = pd.DataFrame(edge_index.T)
df = pd.DataFrame(attention)
df2 = pd.concat([positions, df], axis=1)
attention = df2
#attention_save = np.concatenate([edge_index.T,attention],axis=1)
#np.savetxt(att_dir+file0, attention_save, delimiter=' ')
df2.to_csv(att_dir+file0+".csv",sep=",", index=True)
np.savetxt(embbs_dir+"g128_embeddings", g128_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g64_embeddings", g64_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g32_embeddings", g32_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"class_embeddings", class_embeddings, delimiter=' ')
#np.savetxt(embbs_dir+"node_embeddings", class_rep, delimiter=' ')



################################################################
#
# EVALUATION
#   
################################################################

# Compute ROC-AUC and F1-Score
#from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# Convert class_embeddings to probabilities
class_probs = F.softmax(torch.tensor(class_embeddings, dtype=torch.float32), dim=1).numpy()
# Convert one_hot_labels to class indices
true_labels = np.argmax(one_hot_labels, axis=1)

# ROC-AUC
try:
    roc_auc = roc_auc_score(
        y_true=one_hot_labels,
        y_score=class_probs,
        multi_class='ovr',
        average='macro'
    )
    # Per-class ROC-AUC
    roc_auc_per_class = {}
    for i in range(one_hot_labels.shape[1]):
        if np.sum(one_hot_labels[:, i]) > 0:  # Only compute for classes with samples
            roc_auc_per_class[i] = roc_auc_score(
                one_hot_labels[:, i],
                class_probs[:, i]
            )
    print(f"ROC-AUC (macro): {roc_auc:.4f}")
    print(f"Per-class ROC-AUC: {roc_auc_per_class}")
except ValueError as e:
    print(f"Error computing ROC-AUC: {e}")
    roc_auc = 0.0
    roc_auc_per_class = {}

# F1-Score
pred_labels = np.argmax(class_probs, axis=1)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
f1_micro = f1_score(true_labels, pred_labels, average='micro')
print(f"F1-Score (macro): {f1_macro:.4f}, (weighted): {f1_weighted:.4f}, (micro): {f1_micro:.4f}")


# Compute and plot confusion matrix
#import seaborn as sns
cm = confusion_matrix(true_labels, pred_labels)
# Ensure metadata.CONDITION matches number of classes

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=metadata.CONDITION.unique(), yticklabels=metadata.CONDITION.unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(plots_dir+'Confusion_matrix_figure.pdf')
plt.close()



#-----------------------------------------------------------------------
#
#----------------- Interpretability Plots ------------------------------
#
#-----------------------------------------------------------------------

edge_attn = torch.FloatTensor(attention.iloc[:,2:].mean(axis=1))
src_nodes = edge_index[0]
tgt_nodes = edge_index[1]
# Aggregate per-node attention (e.g., mean of outgoing edge attentions)
node_attention_sum = torch.zeros((x.shape[0], 1), device=edge_attn.device)
node_attention_count = torch.zeros((x.shape[0], 1), device=edge_attn.device)
node_attention_sum.index_add_(0, src_nodes, edge_attn)
node_attention_count.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float32).unsqueeze(1))

# Filter nodes with non-zero counts
connected_mask = (node_attention_count > 0).squeeze()
connected_nodes = torch.arange(x.shape[0], device=device)[connected_mask].cpu().numpy()
print(f"Filtering out {x.shape[0] - len(connected_nodes)} nodes with zero attention count")
# Filter node-related data
node_attention_sum = node_attention_sum[connected_mask].detach().cpu().numpy()
node_attention_count = node_attention_count[connected_mask].detach().cpu().numpy()
node_type_filtered = node_type[connected_mask].cpu().numpy()
saliency_filtered = saliency[connected_mask]
# Create node mapping for edge indices
#node_mapping = {old: new for new, old in enumerate(connected_nodes)}
# Filter attention DataFrame
#attention_filtered = attention[
#    attention['source'].isin(connected_nodes) & 
#    attention['target'].isin(connected_nodes)
#].copy()
#attention_filtered['source'] = attention_filtered['source'].map(node_mapping)
#attention_filtered['target'] = attention_filtered['target'].map(node_mapping)

node_attention_mean = abs(node_attention_sum) / node_attention_count
node_attention_mean = (node_attention_mean.T*(abs(saliency_filtered).mean(axis=1))).T
node_attention_mean = (node_attention_mean/node_attention_mean.sum())*100
node_attention_mean = abs(saliency_filtered).mean(axis=1)


#sample_node_attention = pd.concat([pd.DataFrame(trans.columns[1:]),pd.DataFrame(node_attention_mean[node_type == 2])], axis=1)
#sample_node_attention.columns = ['Sample_ID', 'Attention']
#node_attention_mean = (node_attention_mean/node_attention_mean[node_type != 2].sum())*100
#sample_node_attention = node_attention_mean[node_type == 2]
#gene_node_attention = node_attention_mean[node_type == 0]
#metabo_node_attention = pd.concat([metabo.ID.reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type == 1].detach().numpy())], axis=1)
metabo_node_attention = pd.concat([omic_2_df.ID.reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type_filtered == 1])], axis=1)
metabo_node_attention.columns = ['Omic_2_ID', 'Attention']
#gene_node_attention = pd.concat([trans.ID[top_gene_indices].reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type == 0].detach().numpy())], axis=1)
##gene_node_attention = pd.concat([trans.Symbol[top_gene_indices].reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type_filtered == 0])], axis=1)
gene_node_attention = pd.concat([omic_1_df.ID.reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type_filtered == 0])], axis=1)
#gene_node_attention = pd.concat([trans.ID.reset_index(drop=True),pd.DataFrame(node_attention_mean[node_type_filtered == 0])], axis=1)
gene_node_attention.columns = ['Omic_1_ID', 'Attention']
#sample_node_attention = pd.concat([pd.DataFrame(trans.columns[1:]),pd.DataFrame(node_attention_mean[node_type == 2].detach().numpy())], axis=1)
#import mygene
#mg = mygene.MyGeneInfo()
#gene_info = mg.querymany(gene_node_attention['Gene_ID'].tolist(), scopes='ensembl.gene', fields='symbol', species='human')
#gene_symbols = pd.DataFrame(gene_info)[['query', 'symbol']].drop_duplicates().rename(columns={'query': 'Gene_ID', 'symbol': 'Gene_Symbol'})
#gene_node_attention = pd.merge(gene_node_attention, gene_symbols, on='Gene_ID', how='left')
#gene_node_attention['Gene_Symbol'].fillna(gene_node_attention['Gene_ID'], inplace=True)  # fallback to ID if no match
#from matplotlib import gridspec
# --------------------------
# Prepare gene + metabolite input
# --------------------------
gene_df = gene_node_attention[['Omic_1_ID', 'Attention']].copy()
gene_df['Type'] = 'Omic1_feature'
gene_df = gene_df.rename(columns={'Omic_1_ID': 'Feature'})
metabo_df = metabo_node_attention[['Omic_2_ID', 'Attention']].copy()
metabo_df['Type'] = 'Omic2_feature'
metabo_df = metabo_df.rename(columns={'Omic_2_ID': 'Feature'})
combined = pd.concat([gene_df, metabo_df], ignore_index=True)
combined = combined.sort_values('Attention', ascending=False).reset_index(drop=True)
# --------------------------
# Cumulative attention + saturation
# --------------------------
combined['Cumulative_Attention'] = combined['Attention'].cumsum()
combined['Cumulative_Attention_%'] = 100 * combined['Cumulative_Attention'] / combined['Attention'].sum()
saturation_threshold = 95
saturation_idx = (combined['Cumulative_Attention_%'] >= saturation_threshold).idxmax()
saturation_x = saturation_idx + 1
saturation_y = combined.loc[saturation_idx, 'Cumulative_Attention_%']
#top_features = combined.iloc[:saturation_x]
top_features = combined.iloc[:min(combined.shape[0],300)]
# --------------------------
# Prepare top 100 gene & metabolite
# --------------------------
gene_top100 = gene_df.sort_values('Attention', ascending=False).head(min(100,gene_df.shape[0]))
metabo_top100 = metabo_df.sort_values('Attention', ascending=False).head(min(100,metabo_df.shape[0]))
gene_total_attention = gene_df['Attention'].sum()
metabo_total_attention = metabo_df['Attention'].sum()
# --------------------------
# Plot all into one PDF page
# --------------------------
fig = plt.figure(figsize=(18, 24))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.6])
# Top 100 Genes
ax0 = fig.add_subplot(gs[0, 0])
ax0.barh(gene_top100['Feature'], gene_top100['Attention'], color='steelblue')
ax0.invert_yaxis()
ax0.set_xlabel('Attention Score')
ax0.set_title(f'Top 100 Omic_1 Features\nTotal: {gene_total_attention:.2f} | Mean: {gene_total_attention / len(gene_df):.5f}')
ax0.tick_params(axis='y', labelsize=5)
# Top 100 Metabolites
ax1 = fig.add_subplot(gs[0, 1])
ax1.barh(metabo_top100['Feature'], metabo_top100['Attention'], color='darkorange')
ax1.invert_yaxis()
ax1.set_xlabel('Attention Score')
ax1.set_title(f'Top 100 Omic_2 Features\nTotal: {metabo_total_attention:.2f} | Mean: {metabo_total_attention / len(metabo_df):.5f}')
ax1.tick_params(axis='y', labelsize=5)
# Top Features Before Saturation
ax2 = fig.add_subplot(gs[0, 2])
colors = top_features['Type'].map({'Omic1_feature': 'steelblue', 'Omic2_feature': 'darkorange'})
ax2.barh(top_features['Feature'], top_features['Attention'], color=colors)
ax2.invert_yaxis()
ax2.set_xlabel('Attention Score')
ax2.set_title(f'Features Before Saturation\nTop {300}/{saturation_x} Features to reach {saturation_threshold}%')
ax2.tick_params(axis='y', labelsize=3)
# Cumulative Attention Curve
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(combined['Cumulative_Attention_%'], color='black')
ax3.axvline(x=saturation_x, color='red', linestyle='--')
ax3.axhline(y=saturation_y, color='red', linestyle='--')
ax3.text(saturation_x + 2, saturation_y - 5, f'{saturation_x} features\n({saturation_y:.2f}%)',
         color='red', fontsize=10, ha='left', va='top')
ax3.set_xlabel('Number of Top Features')
ax3.set_ylabel('Cumulative Attention (%)')
ax3.set_title('Cumulative Feature Attention with Saturation Threshold')
# Final layout & save
plt.tight_layout()
plt.savefig(plots_dir+'Interpretability_nodes_attention_figure.pdf')
plt.close()





#-----------------------------------------------------------------------
#
#----------------- Embedding Plots ------------------------------
#
#-----------------------------------------------------------------------

#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#import umap
#import torch
#import numpy as np
#import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
# ----------------------------------
# Helper: Project to 2D with UMAP
# ----------------------------------
def reduce_embedding(embedding, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embedding)

# Convert to numpy if necessary
def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

# ----------------------------------
# Prepare Embeddings & Labels
# ----------------------------------
labels = metadata.CONDITION.reset_index(drop=True)
node_names = omic_1_df.columns[1:]

embeddings = {
    'Cell Nodes': to_numpy(node_rep[node_type == 2]),
    'g128 Embedding': to_numpy(g128_embeddings),
    'g64 Embedding': to_numpy(g64_embeddings),
    'g32 Embedding': to_numpy(g32_embeddings),
    'Class Rep': to_numpy(class_rep)
}
# ----------------------------------
# Create Multi-panel Figure
# ----------------------------------
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])  # Slightly more space for bottom
# Plot 1: Cell Nodes UMAP (Top row)
ax1 = fig.add_subplot(gs[0, :])
cell_nodes_umap = reduce_embedding(embeddings['Cell Nodes'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax1.scatter(cell_nodes_umap[idx, 0], cell_nodes_umap[idx, 1], label=cond, s=15)

ax1.set_title('UMAP of Cell Node Embeddings')
ax1.legend(markerscale=2, fontsize=8)
# Plot 2: g128 Embedding UMAP
ax2 = fig.add_subplot(gs[1, 0])
umap128 = reduce_embedding(embeddings['g128 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax2.scatter(umap128[idx, 0], umap128[idx, 1], label=cond, s=10)

ax2.set_title('UMAP of g128 Embedding')
# Plot 3: g64 Embedding UMAP
ax3 = fig.add_subplot(gs[1, 1])
umap64 = reduce_embedding(embeddings['g64 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax3.scatter(umap64[idx, 0], umap64[idx, 1], label=cond, s=10)

ax3.set_title('UMAP of g64 Embedding')
# Plot 4: g32 Embedding UMAP
ax4 = fig.add_subplot(gs[1, 2])
umap32 = reduce_embedding(embeddings['g32 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax4.scatter(umap32[idx, 0], umap32[idx, 1], label=cond, s=10)

ax4.set_title('UMAP of g32 Embedding')
# Plot 5: Class Rep — choose UMAP or raw depending on dimensions
class_rep_np = embeddings['Class Rep']
D = class_rep_np.shape[1]
if D == 2:
    ax5 = fig.add_subplot(gs[2, :])
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(class_rep_np[idx, 0], class_rep_np[idx, 1], label=cond, s=15)
    
    ax5.set_title('2D Class Representation')
    ax5.legend(markerscale=2, fontsize=8)
elif D == 3:
    ax5 = fig.add_subplot(gs[2, :], projection='3d')
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(class_rep_np[idx, 0], class_rep_np[idx, 1], class_rep_np[idx, 2], label=cond, s=15)
    
    ax5.set_title('3D Class Representation')
    ax5.legend(markerscale=2, fontsize=8)
    ax5.view_init(elev=20, azim=45)
else:
    ax5 = fig.add_subplot(gs[2, :])
    reduced = reduce_embedding(class_rep_np, n_components=2)
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(reduced[idx, 0], reduced[idx, 1], label=cond, s=15)
    
    ax5.set_title(f'UMAP of Class Rep ({D}D)')
    ax5.legend(markerscale=2, fontsize=8)

# Final layout and save
plt.tight_layout()
plt.savefig(plots_dir+'embeddings_evaluations_figure.pdf')
plt.close()





#-----------------------------------------------------------------------
#
#----------------- Sample Features Interpretability --------------------
#
#-----------------------------------------------------------------------

#import numpy as np
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import linkage, leaves_list
#from scipy.spatial.distance import pdist
#import os





#-----------------------------------------------------------------------
#
#----------------- Network Visualization -------------------------------
#
#-----------------------------------------------------------------------

#import networkx as nx
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#from matplotlib.lines import Line2D
#from matplotlib.patches import Patch

# Step 1: Select top 100 genes and top 100 metabolites by attention
top_n = 100
top_genes = gene_node_attention.sort_values(by='Attention', ascending=False).head(top_n)
top_metabolites = metabo_node_attention.sort_values(by='Attention', ascending=False).head(top_n)
attention_network = pd.concat([df2.iloc[:, [0, 1]], pd.DataFrame(edge_attn.T)],axis=1)
attention_network.columns = ['source', 'target', 'weight']

# Get all samples
samples = metadata[['ID', 'CONDITION']].reset_index(drop=True)

# Create node lists
node_labels = pd.concat([
    top_genes['Omic_1_ID'].reset_index(drop=True),
    top_metabolites['Omic_2_ID'].reset_index(drop=True),
    samples['ID'].reset_index(drop=True)
])
node_attention = np.concatenate([
    top_genes['Attention'].values,
    top_metabolites['Attention'].values,
    np.zeros(len(samples))  # Samples may not have attention scores
])
node_types = np.array(
    [0] * len(top_genes) +  # Genes
    [1] * len(top_metabolites) +  # Metabolites
    [2] * len(samples)  # Samples
)
node_shapes = (
    ['triangle'] * len(top_genes) +
    ['square'] * len(top_metabolites) +
    ['circle'] * len(samples)
)

# Step 2: Build Graph
G = nx.Graph()

num_samples = len(samples)
gene_indices = np.array(top_genes.index)
metabo_indices = np.array(top_metabolites.index)+gene_node_attention.shape[0]
sample_indices = np.array(range(num_samples))+(gene_node_attention.shape[0]+metabo_node_attention.shape[0])
all_selected_indices = np.concatenate([gene_indices, metabo_indices,sample_indices])
# Add nodes with attributes
#for i, (label, shape, node_type, att) in zip(all_selected_indices, zip(node_labels, node_shapes, node_types, node_attention)):
#    G.add_node(i, label=label, shape=shape, node_type=node_type, importance=att)
for i, (label, shape, node_type) in zip(all_selected_indices, zip(node_labels, node_shapes, node_types)):
    G.add_node(i, label=label, shape=shape, node_type=node_type)

# Step 3: Filter edges to include only those between selected nodes
#num_genes = len(top_genes)
#num_metabolites = len(top_metabolites)
#num_samples = len(samples)
#gene_indices = np.array(top_genes.index)
#metabo_indices = np.array(top_metabolites.index)+gene_node_attention.shape[0]
#sample_indices = np.array(range(num_samples))+(gene_node_attention.shape[0]+metabo_node_attention.shape[0])
valid_nodes = set(gene_indices).union(metabo_indices).union(sample_indices)
attention_filtered = attention_network[
    (attention_network['source'].isin(valid_nodes)) & (attention_network['target'].isin(valid_nodes))
]
weight_threshold = np.percentile(attention_filtered['weight'], 80)
attention_filtered = attention_filtered[attention_filtered['weight'] >= weight_threshold]

# Normalize edge weights for layout
#max_weight = attention_filtered['weight'].max()
#attention_filtered['weight_norm'] = attention_filtered['weight'] / max_weight

# Add edges with weights
for i, j in zip(attention_filtered['source'], attention_filtered['target']):
    G.add_edge(i, j, weight=1)

# Step 4: Remove isolated nodes
isolated_nodes = [node for node in G.nodes if G.degree[node] == 0]
G.remove_nodes_from(isolated_nodes)

# Update node attributes after removing isolated nodes
node_labels = [G.nodes[node]['label'] for node in G.nodes]
node_shapes = [G.nodes[node]['shape'] for node in G.nodes]
node_types = [G.nodes[node]['node_type'] for node in G.nodes]
#node_attention = [G.nodes[node]['importance'] for node in G.nodes]

# Step 5: Determine edge types
edge_types = []
for u, v in G.edges:
    source_type = G.nodes[u]['node_type']
    target_type = G.nodes[v]['node_type']
    if source_type == 0 and target_type == 0:
        edge_types.append('omic1-omic1')
    elif source_type == 0 and target_type == 1 or source_type == 1 and target_type == 0:
        edge_types.append('omic1-omic2')
    elif source_type == 0 and target_type == 2 or source_type == 2 and target_type == 0:
        edge_types.append('omic1-sample')
    elif source_type == 1 and target_type == 1:
        edge_types.append('omic2-omic2')
    elif source_type == 1 and target_type == 2 or source_type == 2 and target_type == 1:
        edge_types.append('omic2-sample')
    else:  # source_type == 2 and target_type == 2
        edge_types.append('sample-sample')

# Step 6: Plotting
plt.figure(figsize=(12, 12))

# Node positions
pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)

#import plotly.graph_objects as go

shape_map = {
    'triangle': 'triangle-up',
    'square': 'square',
    'circle': 'circle'
}

# Color maps
condition_color_map = {cond: f'rgb{tuple(int(c*255) for c in color[:3])}' for cond, color in zip(metadata.CONDITION.unique(), plt.cm.tab10.colors)}
node_type_colors = {
    0: 'rgb(0, 0, 255)',  # Blue for genes
    1: 'rgb(0, 128, 0)',  # Green for metabolites
    2: None  # Samples by condition
}
edge_type_colors = {
    'omic1-omic1': 'purple',
    'omic1-omic2': 'orange',
    'omic1-sample': 'cyan',
    'omic2-omic2': 'red',
    'omic2-sample': 'magenta',
    'sample-sample': 'black'
}

# Prepare node data
node_x, node_y, node_colors, node_symbols, node_hover = [], [], [], [], []
for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_type = node[1]['node_type']
    label = node[1]['label']
    if node_type == 2:
        condition = metadata.loc[metadata.ID == label, 'CONDITION'].values[0]
        node_colors.append(condition_color_map[condition])
        node_hover.append(f"Sample: {label}<br>Condition: {condition}")
    else:
        node_colors.append(node_type_colors[node_type])
        node_type_name = 'Gene' if node_type == 0 else 'Metabolite'
        node_hover.append(f"{node_type_name}: {label}")
    node_symbols.append(shape_map[node[1]['shape']])

# Prepare edge data by type
edge_traces = []
for edge_type in edge_type_colors:
    edge_x, edge_y = [], []
    for (u, v), et in zip(G.edges(), edge_types):
        if et == edge_type:
            try:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            except KeyError:
                print(f"Warning: Node {u} or {v} not in pos")
                continue
    if edge_x:
        edge_traces.append(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color=edge_type_colors[edge_type]),
            hoverinfo='none',
            name=edge_type.capitalize(),
            showlegend=True
        ))

# Create figure
fig = go.Figure()

# Add edge traces
for trace in edge_traces:
    fig.add_trace(trace)

# Add node trace
fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    marker=dict(size=8, color=node_colors, symbol=node_symbols, line=dict(width=0.5, color='black')),
    text=node_hover,
    hoverinfo='text',
    name='Nodes',
    showlegend=False
))

# Add legend for nodes and conditions
legend_items = [
    go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='triangle-up'),
        name='Gene'
    ),
    go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='green', symbol='square'),
        name='Metabolite'
    ),
    go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='gray', symbol='circle'),
        name='Sample'
    )
]
for cond, color in condition_color_map.items():
    legend_items.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=color, symbol='circle'),
        name=f'Sample: {cond}'
    ))

for item in legend_items:
    fig.add_trace(item)

# Update layout
fig.update_layout(
    title="Network Visualization: Top 100 Genes, Top 100 Metabolites, All Samples",
    showlegend=True,
    legend=dict(x=1, y=1, xanchor='left', yanchor='top'),
    hovermode='closest',
    plot_bgcolor='white',
    width=800,
    height=800,
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

# Save interactive plot
fig.write_html(plots_dir + "network_interactive.html")

# Debug: Print graph stats
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")
print(f"Edge type counts:\n{pd.Series(edge_types).value_counts()}")
degrees = pd.Series(dict(G.degree()), name='degree')
print(f"Node degrees:\n{degrees.describe()}")
print(f"High-degree nodes (>50):\n{degrees[degrees > 50]}")


