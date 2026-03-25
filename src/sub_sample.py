import numpy as np
from collections import defaultdict

def norm_rowcol(matrix):
    # 按行求和
    row_norm=np.sum(matrix,axis=1).reshape(-1,1)
    # 行归一化
    matrix=matrix/row_norm
    # 按列求和
    col_norm=np.sum(matrix,axis=0)
    return matrix/col_norm

def sub_sample_try(graph,GAS, sampling_size,gene_size,gene_shape,cell_shape):
    #cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False) # Generates sampling size number of random indexes for cells 
    #sub_matrix=GAS[:,cell_indexs-gene_shape] # Substract a samplinz matrix with sampled cells
    #gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0] # Extracts genes indexes with connections on those cells
    
    #sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape] # Substracts a sampling matrix with genes connectec to those sampling cells
    
    #sub_matrix=norm_rowcol(sub_matrix) # Normalizes the matrix over genes anc ells connections
    
    #_indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1] # Extracts gene indexes based in their connectivity to sampling cells
    #gene_indexs=gene_indexs[_indexs] # Gets the genes out of the list of genes
    #gene_indexs=gene_indexs[:gene_size] # Gets the sampling size number highest connected genes
    trans_indexes = np.random.randint(0,trans_shape, size=trans_size)
    metabo_indexes = np.random.randint(0,metabo_shape, size=metabo_size)
    m_index, s_index = np.nonzero(mask_metabo_samples)
    sample_indexes = s_index
    
    feature={
        #'gene':graph.node_feature['gene'][gene_indexs,:],
        #'cell':graph.node_feature['cell'][cell_indexs-gene_shape,:],
        #'cell':graph.node_feature['cell'][cell_indexs,:],
        'gene':graph.node_feature['gene'][trans_indexes,:],
        'metabo':graph.node_feature['metabo'][metabo_indexes,:],
        'sample':graph.node_feature['sample']
    }
    
    times={
        'gene': np.ones(trans_size),
        'metabo':np.ones(metabo_size),
        'sample':np.ones(sampling_size)
    }
    
    indxs={
        #'gene':gene_indexs,
        #'cell':cell_indexs-gene_shape
        #'cell':cell_indexs
        'gene':trans_indexes,
        'metabo':metabo_indexes,
        'sample':sample_Indexes
    }
    
    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))
    
    for i in range(gene_size):
        edge_list['gene']['gene']['self'].append([i,i])
    
    for i in range(sampling_size):
        edge_list['cell']['cell']['self'].append([i,i])
    
    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['cell']['gene']['g_c'][cell_id]:
                edge_list['cell']['gene']['g_c'].append([i,j])
                #edge_list['cell']['gene']['g_c'].append([i,j+gene_size])
                edge_list['gene']['cell']['rev_g_c'].append([j,i])
                #edge_list['gene']['cell']['rev_g_c'].append([j+gene_size,i])
                  
    for i,gene_id_i in enumerate(gene_indexs):
        for j,gene_id_ii in enumerate(gene_indexs):
            if gene_id_ii in graph.edge_list['gene']['gene']['g_g'][gene_id_i]:
                edge_list['gene']['gene']['g_g'].append([i,j])
                edge_list['gene']['gene']['rev_g_g'].append([j,i])
                
    for i,cell_id_i in enumerate(cell_indexs):
        for j,cell_id_ii in enumerate(cell_indexs):
            if cell_id_ii in graph.edge_list['cell']['cell']['c_c'][cell_id_i]:
                edge_list['cell']['cell']['c_c'].append([i,j])
                #edge_list['cell']['cell']['c_c'].append([i+gene_size,j+gene_size])
                edge_list['cell']['cell']['rev_c_c'].append([j,i])
                #edge_list['cell']['cell']['rev_c_c'].append([j+gene_size,i+gene_size])
                
    return feature, times, edge_list, indxs

from collections import defaultdict

def sub_sample(graph, GAS_gene_sample, GAS_metabo_sample,
               sampling_size, gene_size, metabo_size,
               gene_shape, metabo_shape, sample_shape):
    
    num_genes = gene_shape
    num_metabolites = metabo_shape
    num_samples = sample_shape

    offset_gene = 0
    offset_metabo = num_genes
    offset_sample = num_genes + num_metabolites
    # Sample samples
    sample_indices = np.random.choice(np.arange(sample_shape), sampling_size, replace=False)

    # === GENE selection based on sampled samples ===
    sub_gene_matrix = GAS_gene_sample[:, sample_indices]
    gene_indices = np.nonzero(np.sum(sub_gene_matrix, axis=1))[0]
    sub_gene_matrix = GAS_gene_sample[gene_indices, :][:, sample_indices]
    gene_scores = np.sum(sub_gene_matrix, axis=1)
    top_gene_indices = gene_indices[np.argsort(gene_scores)[::-1][:gene_size]] + offset_gene
    #top_gene_indices = np.random.choice(np.arange(gene_shape), gene_size, replace=False)

    # === METABOLITE selection based on sampled samples ===
    sub_metabo_matrix = GAS_metabo_sample[:, sample_indices]
    metabo_indices = np.nonzero(np.sum(sub_metabo_matrix, axis=1))[0]
    sub_metabo_matrix = GAS_metabo_sample[metabo_indices, :][:, sample_indices]
    metabo_scores = np.sum(sub_metabo_matrix, axis=1)
    top_metabo_indices = metabo_indices[np.argsort(metabo_scores)[::-1][:metabo_size]] + offset_metabo
    #top_metabo_indices = np.random.choice(np.arange(metabo_shape), metabo_size, replace=False)
    
    sample_indices = sample_indices + offset_sample

    # === Node features ===
    feature = {
        'gene': graph.node_feature['gene'][top_gene_indices-offset_gene, :],
        'metabolite': graph.node_feature['metabolite'][top_metabo_indices-offset_metabo, :],
        'sample': graph.node_feature['sample'][sample_indices-offset_sample, :]
    }

    # === Time stamp (placeholder) ===
    times = {
        'gene': np.ones(gene_size),
        'metabolite': np.ones(metabo_size),
        'sample': np.ones(sampling_size)
    }

    # === Index dictionary ===
    indxs = {
        'gene': top_gene_indices-offset_gene,
        'metabolite': top_metabo_indices-offset_metabo,
        'sample': sample_indices-offset_sample
    }

    # === Edge list initialization ===
    edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Self-loops
    #for i in range(gene_size):
    #    edge_list['gene']['gene']['self'].append([i, i])
    #for i in range(metabo_size):
    #    edge_list['metabolite']['metabolite']['self'].append([i, i])
    #for i in range(sampling_size):
    #    edge_list['sample']['sample']['self'].append([i, i])

    # gene ↔ sample
    for i, s_id in enumerate(sample_indices):
        for j, g_id in enumerate(top_gene_indices):
            if g_id in graph.edge_list['sample']['gene']['g_s'][s_id]:
                edge_list['sample']['gene']['g_s'].append([i, j])
                #edge_list['gene']['sample']['rev_g_s'].append([j, i])

    # metabo ↔ sample
    for i, s_id in enumerate(sample_indices):
        for j, m_id in enumerate(top_metabo_indices):
            if m_id in graph.edge_list['sample']['metabolite']['m_s'][s_id]:
                edge_list['sample']['metabolite']['m_s'].append([i, j])
                #edge_list['metabolite']['sample']['rev_m_s'].append([j, i])
                
    # metabo ↔ gene
    for i, s_id in enumerate(top_metabo_indices):
        for j, m_id in enumerate(top_gene_indices):
            if m_id in graph.edge_list['metabolite']['gene']['g_m'][s_id]:
                edge_list['metabolite']['gene']['g_m'].append([i, j])
                #edge_list['gene']['metabolite']['rev_g_m'].append([j, i])

    # gene ↔ gene
    for i, g1_id in enumerate(top_gene_indices):
        for j, g2_id in enumerate(top_gene_indices):
            if g2_id in graph.edge_list['gene']['gene']['g_g'][g1_id]:
                edge_list['gene']['gene']['g_g'].append([i, j])
                #edge_list['gene']['gene']['rev_g_g'].append([j, i])

    # metabo ↔ metabo
    for i, m1_id in enumerate(top_metabo_indices):
        for j, m2_id in enumerate(top_metabo_indices):
            if m2_id in graph.edge_list['metabolite']['metabolite']['m_m'][m1_id]:
                edge_list['metabolite']['metabolite']['m_m'].append([i, j])
                #edge_list['metabolite']['metabolite']['rev_m_m'].append([j, i])

    # sample ↔ sample
    for i, s1_id in enumerate(sample_indices):
        for j, s2_id in enumerate(sample_indices):
            if s2_id in graph.edge_list['sample']['sample']['s_s'][s1_id]:
                edge_list['sample']['sample']['s_s'].append([i, j])
                #edge_list['sample']['sample']['rev_s_s'].append([j, i])

    return feature, times, edge_list, indxs

