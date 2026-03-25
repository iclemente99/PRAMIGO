import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class AE(nn.Module):
    def __init__(self, dim, h_n):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, h_n)
        self.fc3 = nn.Linear(h_n, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return F.leaky_relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return F.leaky_relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z


def reduction(method,gene_cell,device,h_n):
    if (method == 'AE'):
        encoded,encoded2, losses, losses2 = reduction_AE(gene_cell,device,h_n)
    elif (method == 'raw'):
        encoded = torch.tensor(gene_cell, dtype=torch.float32).to(device)
        encoded2 = torch.tensor(np.transpose(gene_cell),
                                dtype=torch.float32).to(device)
    return encoded,encoded2, losses, losses2

def reduction_AE(gene_cell,device,h_n):
    gene = torch.tensor(gene_cell, dtype=torch.float32).to(device)
    if gene_cell.shape[0] < 5000:
        ba = gene_cell.shape[0]
    else:
        ba = 5000
    encoded, losses =train_AE(gene,ba,device,h_n)

    if gene_cell.shape[1] < 5000:
        ba = gene_cell.shape[1]
    else:
        ba = 5000
    cell = torch.tensor(np.transpose(gene_cell),
                        dtype=torch.float32).to(device)
    encoded2, losses2 =train_AE(cell,ba,device,h_n)
    return encoded,encoded2, losses, losses2


def train_AE(feature,ba,device,h_n):
    loader = data.DataLoader(feature, ba)
    model = AE(dim=feature.shape[1], h_n = h_n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    EPOCH_AE = 2000
    losses = np.zeros(EPOCH_AE)
    for epoch in range(EPOCH_AE):
        embedding1 = []
        for _, batch_x in enumerate(loader)	:
            decoded, encoded = model(batch_x)
            loss = loss_func(batch_x, decoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embedding1.append(encoded)
            losses[epoch] = loss
    print('Epoch :', epoch, '|', 'train_loss:%.12f' % loss.data)
    if feature.shape[0] % ba != 0:
        torch.stack(embedding1[0:int(feature.shape[0]/ba)])
        a = torch.stack(embedding1[0:int(feature.shape[0]/ba)])
        a = a.view(ba*int(feature.shape[0]/ba), h_n)
        encoded = torch.cat((a, encoded), 0)
    else:
        encode = torch.stack(embedding1)
        encoded = encode.view(feature.shape[0], h_n)
    #torch.cuda_empty_cache()
    return encoded, losses
