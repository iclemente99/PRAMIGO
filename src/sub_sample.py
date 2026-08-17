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


def _edge_pairs(sub_mask, target_is_columns):
    """
    Vectorized replacement for the old "for i in ...: for j in ...: if x in graph.edge_list[...]"
    O(n*m) Python double loops. `sub_mask` is the mask matrix already sliced down to the
    sampled row/column indices for this subgraph job, so a single np.nonzero() reproduces
    exactly the same edges the nested-loop + dict-membership check used to find, in
    vectorized numpy instead of per-pair Python/dict overhead.

    target_is_columns=True  -> pairs are [column_index, row_index]   (target, source)
    target_is_columns=False -> pairs are [row_index, column_index]   (target, source)
    """
    rows, cols = np.nonzero(sub_mask)
    if target_is_columns:
        return np.stack([cols, rows], axis=1).tolist()
    return np.stack([rows, cols], axis=1).tolist()


def sub_sample(graph, GAS_gene_sample, GAS_metabo_sample, GAS_gene_metabo,
               mask_gene_gene, mask_metabo_metabo, mask_sample_sample,
               sampling_size, gene_size, metabo_size,
               gene_shape, metabo_shape, sample_shape):

    # Sample samples
    sample_indices = np.random.choice(np.arange(sample_shape), sampling_size, replace=False)

    # === GENE selection based on sampled samples ===
    sub_gene_matrix = GAS_gene_sample[:, sample_indices]
    gene_indices = np.nonzero(np.sum(sub_gene_matrix, axis=1))[0]
    sub_gene_matrix = GAS_gene_sample[gene_indices, :][:, sample_indices]
    gene_scores = np.sum(sub_gene_matrix, axis=1)
    top_gene_indices = gene_indices[np.argsort(gene_scores)[::-1][:gene_size]]

    # === METABOLITE selection based on sampled samples ===
    sub_metabo_matrix = GAS_metabo_sample[:, sample_indices]
    metabo_indices = np.nonzero(np.sum(sub_metabo_matrix, axis=1))[0]
    sub_metabo_matrix = GAS_metabo_sample[metabo_indices, :][:, sample_indices]
    metabo_scores = np.sum(sub_metabo_matrix, axis=1)
    top_metabo_indices = metabo_indices[np.argsort(metabo_scores)[::-1][:metabo_size]]

    # === Node features ===
    feature = {
        'gene': graph.node_feature['gene'][top_gene_indices, :],
        'metabolite': graph.node_feature['metabolite'][top_metabo_indices, :],
        'sample': graph.node_feature['sample'][sample_indices, :]
    }

    # === Time stamp (placeholder) ===
    times = {
        'gene': np.ones(len(top_gene_indices)),
        'metabolite': np.ones(len(top_metabo_indices)),
        'sample': np.ones(sampling_size)
    }

    # === Index dictionary (local, 0-based indices into each node type) ===
    indxs = {
        'gene': top_gene_indices,
        'metabolite': top_metabo_indices,
        'sample': sample_indices
    }

    # === Edge list, built by slicing the graph masks down to the sampled indices ===
    edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # gene <-> sample (target='sample', source='gene')
    sub = GAS_gene_sample[np.ix_(top_gene_indices, sample_indices)]  # rows=gene, cols=sample
    edge_list['sample']['gene']['g_s'] = _edge_pairs(sub, target_is_columns=True)

    # metabolite <-> sample (target='sample', source='metabolite')
    sub = GAS_metabo_sample[np.ix_(top_metabo_indices, sample_indices)]  # rows=metabolite, cols=sample
    edge_list['sample']['metabolite']['m_s'] = _edge_pairs(sub, target_is_columns=True)

    # gene <-> metabolite (target='metabolite', source='gene')
    sub = GAS_gene_metabo[np.ix_(top_gene_indices, top_metabo_indices)]  # rows=gene, cols=metabolite
    edge_list['metabolite']['gene']['g_m'] = _edge_pairs(sub, target_is_columns=True)

    # gene <-> gene (correlation mask is symmetric, so row/col order is interchangeable)
    sub = mask_gene_gene[np.ix_(top_gene_indices, top_gene_indices)]
    edge_list['gene']['gene']['g_g'] = _edge_pairs(sub, target_is_columns=False)

    # metabolite <-> metabolite (symmetric mask)
    sub = mask_metabo_metabo[np.ix_(top_metabo_indices, top_metabo_indices)]
    edge_list['metabolite']['metabolite']['m_m'] = _edge_pairs(sub, target_is_columns=False)

    # sample <-> sample (symmetric mask)
    sub = mask_sample_sample[np.ix_(sample_indices, sample_indices)]
    edge_list['sample']['sample']['s_s'] = _edge_pairs(sub, target_is_columns=False)

    return feature, times, edge_list, indxs
