import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

class LinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, past_states, current_state, gradients):
      update_matrix = torch.zeros((self.hidden_dim, self.hidden_dim)).to(current_state.device)
      for i,past_h in enumerate(past_states):
        update_matrix += torch.outer(past_h, current_state) * gradients[i]
      return update_matrix

class FastWeightLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.a = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.linear_attention = LinearAttention(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
      batch_size, seq_len, _ = hidden_states.shape
      fast_outputs = []
      fast_weights = {
          'W': self.W.clone(),
          'b': self.b.clone()
      }
      all_gradients = []
      past_hidden_states = []
      for t in range(seq_len):
          current_hidden = hidden_states[:,t,:]
          output_slow = self.layer_norm(self.relu(current_hidden @ self.U + self.a) @ self.W + self.b )
          output_slow_for_grad = output_slow.clone().requires_grad_(True)
          dummy_target = torch.randint(0, self.hidden_dim, (batch_size,)).to(hidden_states.device)
          loss = F.cross_entropy(output_slow_for_grad, dummy_target)
          grad_W = torch.autograd.grad(loss, self.W, retain_graph=True)[0]
          grad_b = torch.autograd.grad(loss, self.b, retain_graph=True)[0]
          
          gradients = [grad_W, grad_b]
          all_gradients.append(gradients)
          past_hidden_states.append(current_hidden)
          if t > 0:
              W_update = self.linear_attention(past_hidden_states[:-1], current_hidden, [g[0] for g in all_gradients[:-1]])
              b_update = self.linear_attention(past_hidden_states[:-1], current_hidden, [g[1] for g in all_gradients[:-1]]).sum(dim=0)
              fast_weights['W'] = fast_weights['W'] - W_update
              fast_weights['b'] = fast_weights['b'] - b_update
          output_fast = self.layer_norm(self.relu(current_hidden @ self.U + self.a) @ fast_weights['W'] + fast_weights['b'] )
          fast_outputs.append(output_fast)
      fast_outputs = torch.stack(fast_outputs, dim=1)
      return fast_outputs

class SimplifiedTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.fwl = FastWeightLayer(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
      embedded = self.embedding(x)
      hidden_states = self.transformer_encoder(embedded.permute(1,0,2)).permute(1,0,2)
      fwl_output = self.fwl(hidden_states)
      output = self.output_layer(fwl_output)
      return output

def prepare_data(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_and_save_model(X, y, vocab_size, hidden_dim, num_epochs, batch_size, learning_rate, save_path='model_1.pth'):
    model_1 = SimplifiedTransformer(vocab_size, hidden_dim)
    dataloader = prepare_data(X, y, batch_size)
    optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1.to(device)
    for epoch in range(num_epochs):
        model_1.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)
          optimizer.zero_grad()
          output = model_1(batch_x)
          output_reshape = output.view(-1, output.size(-1))
          batch_y_reshape = batch_y.view(-1)
          loss = criterion(output_reshape, batch_y_reshape)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model_1.state_dict(), save_path)

if __name__ == '__main__':
    vocab_size = 100
    hidden_dim = 28
    num_samples = 872
    num_features = hidden_dim
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    X = torch.randint(0, vocab_size, (num_samples, num_features)).numpy()
    y = torch.randint(0, vocab_size, (num_samples, )).numpy()
    train_and_save_model(X, y, vocab_size, hidden_dim, num_epochs, batch_size, learning_rate)
    print("Training Completed and Model Saved!")
