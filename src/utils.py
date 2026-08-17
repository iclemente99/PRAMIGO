from pyHGT.data import Graph
import numpy as np
import torch
from collections import defaultdict
#import resource
import pandas as pd
try:
    import resource
except ImportError:
    resource = None

#def debuginfoStr(info):
#    print(info)
#    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)
#    print('Mem consumption (GB): '+str(mem))
    
def debuginfoStr(info):
    print(info)
    try:
        if resource is None:
            return
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print('Mem consumption (GB): ' + str(mem))
    except Exception:
        # Silently skip on Windows or any unexpected platform issues
        pass



def loadGAS(data_path):
    df=pd.read_csv(data_path, sep=" ")
    return df.to_numpy()



def build_graph(mask_trans_metabo, mask_trans_sample, mask_metabo_samples,
                mask_trans, mask_metabo, mask_sample,
                encoded_trans, encoded_metabo, encoded2_samples):

    graph = Graph()

    num_genes = encoded_trans.shape[0]
    num_metabolites = encoded_metabo.shape[0]
    num_samples = encoded2_samples.shape[0]

    offset_gene = 0
    offset_metabo = num_genes
    offset_sample = num_genes + num_metabolites

    # === 1. gene ↔ sample edges ===
    g_idx, s_idx = np.nonzero(mask_trans_sample)
    s_idx += offset_sample
    edges_gs = torch.tensor([g_idx, s_idx], dtype=torch.float)

    s_type, r_type, t_type = ('gene', 'g_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_gs.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === 2. gene ↔ metabolite edges ===
    g_idx, m_idx = np.nonzero(mask_trans_metabo)
    m_idx += offset_metabo
    edges_gm = torch.tensor([g_idx, m_idx], dtype=torch.float)

    s_type, r_type, t_type = ('gene', 'g_m', 'metabolite')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_gm.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === 3. metabolite ↔ sample edges ===
    m_idx, s_idx = np.nonzero(mask_metabo_samples)
    m_idx += offset_metabo
    s_idx += offset_sample
    edges_ms = torch.tensor([m_idx, s_idx], dtype=torch.float)

    s_type, r_type, t_type = ('metabolite', 'm_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_ms.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === 4. gene ↔ gene edges ===
    g1_idx, g2_idx = np.nonzero(mask_trans)
    edges_gg = torch.tensor([g1_idx, g2_idx], dtype=torch.float)

    s_type, r_type, t_type = ('gene', 'g_g', 'gene')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_gg.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === 5. metabolite ↔ metabolite edges ===
    m1_idx, m2_idx = np.nonzero(mask_metabo)
    m1_idx += offset_metabo
    m2_idx += offset_metabo
    edges_mm = torch.tensor([m1_idx, m2_idx], dtype=torch.float)

    s_type, r_type, t_type = ('metabolite', 'm_m', 'metabolite')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_mm.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === 6. sample ↔ sample edges ===
    s1_idx, s2_idx = np.nonzero(mask_sample)
    s1_idx += offset_sample
    s2_idx += offset_sample
    edges_ss = torch.tensor([s1_idx, s2_idx], dtype=torch.float)

    s_type, r_type, t_type = ('sample', 's_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    #rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]

    for s_id, t_id in edges_ss.t().tolist():
        elist[t_id][s_id] = 1
        #rlist[s_id][t_id] = 1

    # === Node features ===
    graph.node_feature['gene'] = torch.tensor(encoded_trans, dtype=torch.float)
    graph.node_feature['metabolite'] = torch.tensor(encoded_metabo, dtype=torch.float)
    graph.node_feature['sample'] = torch.tensor(encoded2_samples, dtype=torch.float)

    graph.years = np.ones(num_genes + num_metabolites + num_samples)
    
    return graph
    
    
import torch
import numpy as np
from collections import defaultdict

def build_graph_pruned(mask_trans_metabo, mask_trans_sample, mask_metabo_samples,
                       mask_trans, mask_metabo, mask_sample,
                       encoded_trans, encoded_metabo, encoded2_samples):
    #class Graph:
    #    def __init__(self):
    #        self.edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    #        self.node_feature = defaultdict(list)
    #        self.years = None

    graph = Graph()

    num_genes = encoded_trans.shape[0]
    num_metabolites = encoded_metabo.shape[0]
    num_samples = encoded2_samples.shape[0]

    offset_gene = 0
    offset_metabo = num_genes
    offset_sample = num_genes + num_metabolites
    total_nodes = num_genes + num_metabolites + num_samples

    # Validate input matrix shapes
    assert mask_trans_sample.shape == (num_genes, num_samples), f"Expected mask_trans_sample shape ({num_genes}, {num_samples}), got {mask_trans_sample.shape}"
    assert mask_trans_metabo.shape == (num_genes, num_metabolites), f"Expected mask_trans_metabo shape ({num_genes}, {num_metabolites}), got {mask_trans_metabo.shape}"
    assert mask_metabo_samples.shape == (num_metabolites, num_samples), f"Expected mask_metabo_samples shape ({num_metabolites}, {num_samples}), got {mask_metabo_samples.shape}"
    assert mask_trans.shape == (num_genes, num_genes), f"Expected mask_trans shape ({num_genes}, {num_genes}), got {mask_trans.shape}"
    assert mask_metabo.shape == (num_metabolites, num_metabolites), f"Expected mask_metabo shape ({num_metabolites}, {num_metabolites}), got {mask_metabo.shape}"
    assert mask_sample.shape == (num_samples, num_samples), f"Expected mask_sample shape ({num_samples}, {num_samples}), got {mask_sample.shape}"

    # === 1. gene ↔ sample edges ===
    g_idx, s_idx = np.nonzero(mask_trans_sample)
    assert np.all(g_idx < num_genes), f"Invalid gene indices in mask_trans_sample: {g_idx.max()} >= {num_genes}"
    assert np.all(s_idx < num_samples), f"Invalid sample indices in mask_trans_sample: {s_idx.max()} >= {num_samples}"
    s_idx += offset_sample
    edges_gs = torch.tensor([g_idx, s_idx], dtype=torch.long)
    s_type, r_type, t_type = ('gene', 'g_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_gs.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === 2. gene ↔ metabolite edges ===
    g_idx, m_idx = np.nonzero(mask_trans_metabo)
    assert np.all(g_idx < num_genes), f"Invalid gene indices in mask_trans_metabo: {g_idx.max()} >= {num_genes}"
    assert np.all(m_idx < num_metabolites), f"Invalid metabolite indices in mask_trans_metabo: {m_idx.max()} >= {num_metabolites}"
    m_idx += offset_metabo
    edges_gm = torch.tensor([g_idx, m_idx], dtype=torch.long)
    s_type, r_type, t_type = ('gene', 'g_m', 'metabolite')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_gm.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === 3. metabolite ↔ sample edges ===
    m_idx, s_idx = np.nonzero(mask_metabo_samples)
    assert np.all(m_idx < num_metabolites), f"Invalid metabolite indices in mask_metabo_samples: {m_idx.max()} >= {num_metabolites}"
    assert np.all(s_idx < num_samples), f"Invalid sample indices in mask_metabo_samples: {s_idx.max()} >= {num_samples}"
    m_idx += offset_metabo
    s_idx += offset_sample
    edges_ms = torch.tensor([m_idx, s_idx], dtype=torch.long)
    s_type, r_type, t_type = ('metabolite', 'm_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_ms.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === 4. gene ↔ gene edges ===
    g1_idx, g2_idx = np.nonzero(mask_trans)
    assert np.all(g1_idx < num_genes), f"Invalid gene indices in mask_trans: {g1_idx.max()} >= {num_genes}"
    assert np.all(g2_idx < num_genes), f"Invalid gene indices in mask_trans: {g2_idx.max()} >= {num_genes}"
    edges_gg = torch.tensor([g1_idx, g2_idx], dtype=torch.long)
    s_type, r_type, t_type = ('gene', 'g_g', 'gene')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_gg.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === 5. metabolite ↔ metabolite edges ===
    m1_idx, m2_idx = np.nonzero(mask_metabo)
    assert np.all(m1_idx < num_metabolites), f"Invalid metabolite indices in mask_metabo: {m1_idx.max()} >= {num_metabolites}"
    assert np.all(m2_idx < num_metabolites), f"Invalid metabolite indices in mask_metabo: {m2_idx.max()} >= {num_metabolites}"
    m1_idx += offset_metabo
    m2_idx += offset_metabo
    edges_mm = torch.tensor([m1_idx, m2_idx], dtype=torch.long)
    s_type, r_type, t_type = ('metabolite', 'm_m', 'metabolite')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_mm.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === 6. sample ↔ sample edges ===
    s1_idx, s2_idx = np.nonzero(mask_sample)
    assert np.all(s1_idx < num_samples), f"Invalid sample indices in mask_sample: {s1_idx.max()} >= {num_samples}"
    assert np.all(s2_idx < num_samples), f"Invalid sample indices in mask_sample: {s2_idx.max()} >= {num_samples}"
    s1_idx += offset_sample
    s2_idx += offset_sample
    edges_ss = torch.tensor([s1_idx, s2_idx], dtype=torch.long)
    s_type, r_type, t_type = ('sample', 's_s', 'sample')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges_ss.t().tolist():
        elist[t_id][s_id] = 1
        rlist[s_id][t_id] = 1

    # === Node features ===
    graph.node_feature['gene'] = torch.tensor(encoded_trans, dtype=torch.float)
    graph.node_feature['metabolite'] = torch.tensor(encoded_metabo, dtype=torch.float)
    graph.node_feature['sample'] = torch.tensor(encoded2_samples, dtype=torch.float)
    graph.years = np.ones(num_genes + num_metabolites + num_samples)

    # === Remove isolated nodes ===
    connected_nodes = set()
    for t_type in graph.edge_list:
        for s_type in graph.edge_list[t_type]:
            for r_type in graph.edge_list[t_type][s_type]:
                for t_id, s_dict in graph.edge_list[t_type][s_type][r_type].items():
                    if t_id >= total_nodes:
                        print(f"Warning: Invalid target node ID {t_id} >= {total_nodes}")
                        continue
                    for s_id in s_dict:
                        if s_id >= total_nodes:
                            print(f"Warning: Invalid source node ID {s_id} >= {total_nodes}")
                            continue
                        connected_nodes.add(t_id)
                        connected_nodes.add(s_id)

    # Debug: Print connected nodes
    print(f"Connected nodes: {len(connected_nodes)} out of {total_nodes}")
    print(f"Sample connected nodes: {sorted(list(connected_nodes))[:10]}...")

    # Create mapping for new node indices
    old_to_new = {old: new for new, old in enumerate(sorted(connected_nodes))}
    num_nodes = len(connected_nodes)

    # Filter node features and years
    new_node_feature = {
        'gene': [],
        'metabolite': [],
        'sample': []
    }
    new_years = np.zeros(num_nodes)

    for old_id in sorted(connected_nodes):
        if old_id < num_genes:
            node_type = 'gene'
            feature_idx = old_id
        elif old_id < num_genes + num_metabolites:
            node_type = 'metabolite'
            feature_idx = old_id - num_genes
        else:
            node_type = 'sample'
            feature_idx = old_id - num_genes - num_metabolites
        if feature_idx >= len(graph.node_feature[node_type]):
            print(f"Warning: Invalid feature_idx {feature_idx} for {node_type} (max {len(graph.node_feature[node_type])})")
            continue
        new_node_feature[node_type].append(graph.node_feature[node_type][feature_idx])
        new_years[old_to_new[old_id]] = graph.years[old_id]

    # Convert node features to tensors
    for node_type in new_node_feature:
        if new_node_feature[node_type]:
            new_node_feature[node_type] = torch.stack(new_node_feature[node_type])
        else:
            new_node_feature[node_type] = torch.tensor([], dtype=torch.float)

    # Filter edges
    new_edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for t_type in graph.edge_list:
        for s_type in graph.edge_list[t_type]:
            for r_type in graph.edge_list[t_type][s_type]:
                for t_id, s_dict in graph.edge_list[t_type][s_type][r_type].items():
                    if t_id not in old_to_new:
                        continue
                    new_t_id = old_to_new[t_id]
                    for s_id in s_dict:
                        if s_id not in old_to_new:
                            continue
                        new_s_id = old_to_new[s_id]
                        new_edge_list[t_type][s_type][r_type][new_t_id][new_s_id] = 1

    # Update graph
    graph.edge_list = new_edge_list
    graph.node_feature = new_node_feature
    graph.years = new_years

    return graph



def build_data(mask_trans_metabo, mask_trans_sample, mask_metabo_samples,
               mask_trans, mask_metabo, mask_sample,
               encoded_trans, encoded_metabo, encoded2_samples,
               edge_dict):
    """
    Builds node features, node types, and edge information from full graph data.

    Returns:
        x: dict of node features
        node_type: tensor of node types
        edge_time: tensor of edge timestamps
        edge_index: tensor of [2, num_edges]
        edge_type: tensor of edge types
    """

    # Get node counts
    num_genes = encoded_trans.shape[0]
    num_metabolites = encoded_metabo.shape[0]
    num_samples = encoded2_samples.shape[0]

    # ----- Node type vector -----
    node_type = (
        [0] * num_genes +  # gene
        [1] * num_metabolites +  # metabolite
        [2] * num_samples  # sample
    )
    node_type = torch.LongTensor(node_type)

    # ----- Gene-Metabolite edges -----
    g_idx, m_idx = np.nonzero(mask_trans_metabo)
    m_idx += num_genes  # shift for metabolite node indices
    edge_index_gm = torch.tensor([g_idx, m_idx], dtype=torch.long)
    edge_type_gm = torch.LongTensor([edge_dict['g_m']] * edge_index_gm.shape[1])
    edge_time_gm = torch.LongTensor([0] * edge_index_gm.shape[1])

    # ----- Gene-Sample edges -----
    g_idx, s_idx = np.nonzero(mask_trans_sample)
    s_idx += (num_genes + num_metabolites)
    edge_index_gs = torch.tensor([g_idx, s_idx], dtype=torch.long)
    edge_type_gs = torch.LongTensor([edge_dict['g_s']] * edge_index_gs.shape[1])
    edge_time_gs = torch.LongTensor([0] * edge_index_gs.shape[1])

    # ----- Metabolite-Sample edges -----
    m_idx, s_idx = np.nonzero(mask_metabo_samples)
    m_idx += num_genes
    s_idx += (num_genes + num_metabolites)
    edge_index_ms = torch.tensor([m_idx, s_idx], dtype=torch.long)
    edge_type_ms = torch.LongTensor([edge_dict['m_s']] * edge_index_ms.shape[1])
    edge_time_ms = torch.LongTensor([0] * edge_index_ms.shape[1])

    # ----- Gene-Gene edges -----
    g1_idx, g2_idx = np.nonzero(mask_trans)
    edge_index_gg = torch.tensor([g1_idx, g2_idx], dtype=torch.long)
    edge_type_gg = torch.LongTensor([edge_dict['g_g']] * edge_index_gg.shape[1])
    edge_time_gg = torch.LongTensor([0] * edge_index_gg.shape[1])

    # ----- Metabolite-Metabolite edges -----
    m1_idx, m2_idx = np.nonzero(mask_metabo)
    m1_idx += num_genes
    m2_idx += num_genes
    edge_index_mm = torch.tensor([m1_idx, m2_idx], dtype=torch.long)
    edge_type_mm = torch.LongTensor([edge_dict['m_m']] * edge_index_mm.shape[1])
    edge_time_mm = torch.LongTensor([0] * edge_index_mm.shape[1])

    # ----- Sample-Sample edges -----
    s1_idx, s2_idx = np.nonzero(mask_sample)
    s1_idx += (num_genes + num_metabolites)
    s2_idx += (num_genes + num_metabolites)
    edge_index_ss = torch.tensor([s1_idx, s2_idx], dtype=torch.long)
    edge_type_ss = torch.LongTensor([edge_dict['s_s']] * edge_index_ss.shape[1])
    edge_time_ss = torch.LongTensor([0] * edge_index_ss.shape[1])

    # ----- Concatenate all edges -----
    edge_index = torch.cat([
        edge_index_gm, edge_index_gs, edge_index_ms,
        edge_index_gg, edge_index_mm, edge_index_ss
    ], dim=1)

    edge_type = torch.cat([
        edge_type_gm, edge_type_gs, edge_type_ms,
        edge_type_gg, edge_type_mm, edge_type_ss
    ])

    edge_time = torch.cat([
        edge_time_gm, edge_time_gs, edge_time_ms,
        edge_time_gg, edge_time_mm, edge_time_ss
    ])

    # ----- Node features -----
    x = {
        'gene': torch.tensor(encoded_trans, dtype=torch.float),
        'metabolite': torch.tensor(encoded_metabo, dtype=torch.float),
        'sample': torch.tensor(encoded2_samples, dtype=torch.float)
    }

    return x, node_type, edge_time, edge_index, edge_type


def prune_disconnected_nodes(graph):
    # Determine node counts
    num_genes = graph.node_feature['gene'].shape[0]
    num_metabolites = graph.node_feature['metabolite'].shape[0]
    num_samples = graph.node_feature['sample'].shape[0]

    # Total nodes
    total_nodes = num_genes + num_metabolites + num_samples

    # Build degree count array
    node_degrees = np.zeros(total_nodes)

    # Count edges per node in all edge lists
    for t_type in graph.edge_list:
        for s_type in graph.edge_list[t_type]:
            for r_type in graph.edge_list[t_type][s_type]:
                adj = graph.edge_list[t_type][s_type][r_type]
                for tgt_id in adj:
                    for src_id in adj[tgt_id]:
                        node_degrees[tgt_id] += 1
                        node_degrees[src_id] += 1

    # Determine which nodes are connected
    connected_nodes = np.where(node_degrees > 0)[0]
    print(f"Keeping {len(connected_nodes)} out of {total_nodes} nodes")

    # Build mapping from old indices to new compact indices
    old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_nodes)}

    # Create new node_feature dict
    def filter_node_features(node_type, offset_start, count):
        keep_indices = [old_to_new_index[i] - offset_start for i in connected_nodes if offset_start <= i < offset_start + count]
        return graph.node_feature[node_type][keep_indices]

    new_node_feature = {}
    offset_gene = 0
    offset_metabo = num_genes
    offset_sample = num_genes + num_metabolites

    new_node_feature['gene'] = filter_node_features('gene', offset_gene, num_genes)
    new_node_feature['metabolite'] = filter_node_features('metabolite', offset_metabo, num_metabolites)
    new_node_feature['sample'] = filter_node_features('sample', offset_sample, num_samples)

    graph.node_feature = new_node_feature

    # Update edge_list with remapped node indices, removing edges with pruned nodes
    new_edge_list = {}
    for t_type in graph.edge_list:
        new_edge_list[t_type] = {}
        for s_type in graph.edge_list[t_type]:
            new_edge_list[t_type][s_type] = {}
            for r_type in graph.edge_list[t_type][s_type]:
                adj = graph.edge_list[t_type][s_type][r_type]
                new_adj = {}
                for tgt_id in adj:
                    if tgt_id not in old_to_new_index:
                        continue
                    new_tgt = old_to_new_index[tgt_id]
                    new_adj[new_tgt] = {}
                    for src_id in adj[tgt_id]:
                        if src_id in old_to_new_index:
                            new_src = old_to_new_index[src_id]
                            new_adj[new_tgt][new_src] = 1
                    if len(new_adj[new_tgt]) == 0:
                        del new_adj[new_tgt]
                new_edge_list[t_type][s_type][r_type] = new_adj

    graph.edge_list = new_edge_list

    # Update years if used
    graph.years = graph.years[connected_nodes]

    return graph

