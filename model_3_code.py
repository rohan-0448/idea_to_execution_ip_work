import torch
import torch.nn as nn
import torch.nn.functional as F

class eGLOM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = TaskEncoder(input_dim, hidden_dim)
        self.mpnn = MPNNProcessor(hidden_dim)
        self.decoder = TaskDecoder(hidden_dim, output_dim, include_position_encoding = True)
        self.hidden_dim = hidden_dim
        self.max_iterations = 5

    def forward(self, ellipse_symbols, positions):
        batch_size, num_ellipses, _ = ellipse_symbols.shape
        device = ellipse_symbols.device
        e = torch.zeros(batch_size, num_ellipses, self.hidden_dim).to(device)
        o = torch.zeros(batch_size, num_ellipses, self.hidden_dim).to(device)
        h_e = torch.zeros(batch_size, num_ellipses, self.hidden_dim).to(device)
        h_o = torch.zeros(batch_size, num_ellipses, self.hidden_dim).to(device)
        w_hist, w_topdown, w_bottomup, w_attn = 0.25, 0.25, 0.25, 0.25
        
        for t in range(self.max_iterations):

            node_embeddings = self.encoder(ellipse_symbols, positions)

            e_bottom_up = self.mpnn.mlp_bu_s2e(node_embeddings)
            e_top_down = self.mpnn.mlp_td_o2e(o, positions)
            e_t = w_hist * h_e + w_bottomup * e_bottom_up + w_topdown * e_top_down
            h_e = e_t

            o_bottom_up = self.mpnn.mlp_bu_e2o(e)
            attention = self.mpnn.attention(o)
            o_t = w_hist * h_o + w_bottomup * o_bottom_up + w_attn * attention
            h_o = o_t

            r_t = self.decoder.mlp_td_e2s(e, positions)
            next_hints, output = self.decoder.mlp_bu_o2out(o)
        
        return next_hints, output, r_t

class TaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
       super().__init__()
       self.linear_s2emb = nn.Linear(input_dim, hidden_dim)
       self.linear_p2emb = nn.Linear(2*10, hidden_dim)

    def forward(self, ellipse_symbols, positions):
        emb_s = self.linear_s2emb(ellipse_symbols)
        emb_p = self.linear_p2emb(positions)
        return emb_s + emb_p
    
class MPNNProcessor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_bu_e2o = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mlp_td_o2e = nn.Sequential(
            nn.Linear(hidden_dim + 2*10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mlp_bu_s2e = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention_temperature = 0.1

    def attention(self, object_embeddings):
        #Simplified attention implementation with softmax only
        batch_size, num_ellipses, hidden_dim = object_embeddings.shape
        attention_weights = torch.matmul(object_embeddings, object_embeddings.transpose(1,2)) / self.attention_temperature
        attention_weights = F.softmax(attention_weights, dim=-1)
        weighted_embeddings = torch.matmul(attention_weights, object_embeddings)

        return weighted_embeddings
       
class TaskDecoder(nn.Module):
   def __init__(self, hidden_dim, output_dim, include_position_encoding):
    super().__init__()
    if include_position_encoding:
      self.mlp_td_e2s = nn.Sequential(
        nn.Linear(hidden_dim + 2*10, hidden_dim),
          nn.ReLU(),
        nn.Linear(hidden_dim, 6)
      )
    else:
      self.mlp_td_e2s = nn.Sequential(
          nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
          nn.Linear(hidden_dim, 6)
        )

    self.mlp_bu_o2out = nn.Linear(hidden_dim, output_dim)

   def forward(self, processed_node_embeddings, positions):
       
       
       return self.mlp_bu_o2out(processed_node_embeddings), self.mlp_td_e2s(processed_node_embeddings, positions)
