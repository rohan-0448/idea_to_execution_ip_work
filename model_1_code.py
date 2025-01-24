import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class FastWeightLayer(nn.Module):
    def __init__(self, hidden_dim, fwl_dim):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, fwl_dim)
        self.hidden_bias = nn.Parameter(torch.zeros(fwl_dim))
        self.projection_layer = nn.Linear(fwl_dim, fwl_dim, bias = False)
        self.projection_bias = nn.Parameter(torch.zeros(fwl_dim))
        self.layer_norm = nn.LayerNorm(fwl_dim)
        self.step_size_hidden_layer = nn.Parameter(torch.ones(self.hidden_layer.weight.shape))
        self.step_size_hidden_bias = nn.Parameter(torch.ones(self.hidden_bias.shape))
        self.step_size_projection_layer = nn.Parameter(torch.ones(self.projection_layer.weight.shape))
        self.step_size_projection_bias = nn.Parameter(torch.ones(self.projection_bias.shape))

    def forward(self, h, embedding_matrix, output_bias, labels):
        batch_size, seq_length, _ = h.shape
        o = self.layer_norm(F.relu(self.hidden_layer(h) + self.hidden_bias) @ self.projection_layer.weight.T + self.projection_bias)
        slow_output = o

        logits = o @ embedding_matrix.T + output_bias
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction = 'none').view(batch_size, seq_length)

        grad_hidden_layer = torch.autograd.grad(loss.sum(), self.hidden_layer.weight, create_graph=True)[0]
        grad_hidden_bias = torch.autograd.grad(loss.sum(), self.hidden_bias, create_graph=True)[0]
        grad_projection_layer = torch.autograd.grad(loss.sum(), self.projection_layer.weight, create_graph=True)[0]
        grad_projection_bias = torch.autograd.grad(loss.sum(), self.projection_bias, create_graph=True)[0]


        gradients = [grad_hidden_layer, grad_hidden_bias, grad_projection_layer, grad_projection_bias]
        
        o_fast = torch.zeros_like(o)
        for t in range(seq_length):

            v_t = F.relu(self.hidden_layer(h[:,t,:]) + self.hidden_bias)
            fast_w = self.projection_layer.weight - self.step_size_projection_layer * torch.stack([torch.matmul(v_t, v.T) for v in v_t], dim=0).matmul(grad_projection_layer)
            fast_b = self.projection_bias - self.step_size_projection_bias *  torch.stack([torch.matmul(v_t, v.T) for v in v_t], dim=0).matmul(grad_projection_bias)
            o_fast[:,t,:] = self.layer_norm((v_t @ fast_w.transpose(1,2) + fast_b))


        logits_fast = o_fast @ embedding_matrix.T + output_bias
        fast_loss = F.cross_entropy(logits_fast.view(-1, logits_fast.shape[-1]), labels.view(-1), reduction = 'none').view(batch_size, seq_length)

        return fast_loss, gradients
class TransformerPlusFWL(nn.Module):
    def __init__(self, model_name, fwl_dim):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_dim = self.transformer.config.hidden_size
        self.fwl = FastWeightLayer(hidden_dim, fwl_dim)

    def forward(self, input_ids, labels):
        h = self.transformer(input_ids).last_hidden_state
        embedding_matrix = self.transformer.get_input_embeddings().weight
        output_bias = self.transformer.lm_head.bias
        fast_loss, _ = self.fwl(h, embedding_matrix, output_bias, labels)
        return fast_loss
