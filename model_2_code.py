import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class MPNNProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.message_function = nn.Sequential(
            nn.Linear(2*hidden_dim + 2*hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.readout_function = nn.Linear(2*hidden_dim, hidden_dim)
        self.gate_function = nn.Sequential(
            nn.Linear(2*hidden_dim + 2*hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.triplet_message_function = nn.Sequential(
            nn.Linear(3*hidden_dim + 3*hidden_dim+hidden_dim, 8),
            nn.ReLU()
        )
        self.triplet_edge_readout_function = nn.Linear(8,hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_embeddings, edge_embeddings, graph_embeddings, previous_hidden):
        z_i = torch.cat((node_embeddings, previous_hidden), dim=-1)

        m_i = []
        for j in range(node_embeddings.shape[-2]):
            z_j = torch.cat((node_embeddings[:, j, :], previous_hidden[:,j, :]), dim = -1)
            m_ij = self.message_function(torch.cat((z_i, z_j, edge_embeddings[:,j,:], graph_embeddings), dim = -1))
            m_i.append(m_ij)
        m_i = torch.max(torch.stack(m_i, dim = -2), dim = -2)[0]

        g_i = self.gate_function(torch.cat((z_i, m_i, edge_embeddings, graph_embeddings), dim = -1))
        h_i = self.readout_function(torch.cat((z_i, m_i), dim = -1))
        
        h_g_i = g_i * h_i + (1-g_i) * previous_hidden

        t_ijk = []
        for k in range(node_embeddings.shape[-2]):
          for j in range(node_embeddings.shape[-2]):
            t_ijk_value = self.triplet_message_function(torch.cat((node_embeddings[:, j, :], node_embeddings[:, k, :], node_embeddings, edge_embeddings, edge_embeddings, edge_embeddings, graph_embeddings), dim=-1))
            t_ijk.append(t_ijk_value)

        t_ijk_stacked = torch.stack(t_ijk, dim = -2)
        h_ij = self.triplet_edge_readout_function(torch.max(t_ijk_stacked, dim=-2)[0])

        h_i_norm = self.layer_norm(h_g_i)
        return h_i_norm, h_ij
    
class TaskEncoder(nn.Module):
  def __init__(self, input_feature_dims, hidden_dim):
      super().__init__()
      self.encoders = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in input_feature_dims])
  
  def forward(self, node_inputs, edge_inputs, graph_inputs):
      node_embeddings = self.encoders[0](node_inputs)
      edge_embeddings = self.encoders[1](edge_inputs)
      graph_embeddings = self.encoders[2](graph_inputs)
      return node_embeddings, edge_embeddings, graph_embeddings

class TaskDecoder(nn.Module):
    def __init__(self, output_feature_dims, hidden_dim):
      super().__init__()
      self.decoders = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in output_feature_dims])
    
    def forward(self, processed_node_embeddings):
        hints = self.decoders[0](processed_node_embeddings)
        outputs = self.decoders[1](processed_node_embeddings)
        return hints, outputs

def sinkhorn_operator(matrix, temperature=0.1, iterations=10, add_noise=False):
    # Placeholder implementation (add details if needed)
    return matrix
