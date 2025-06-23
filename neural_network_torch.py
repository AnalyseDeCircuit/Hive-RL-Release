import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=None):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.to(self.device)

    def forward(self, state_vector):
        # 支持单样本或批量输入
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float()
        state_vector = state_vector.to(self.device)
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)  # [input_dim] -> [1, input_dim]
        x = torch.tanh(self.fc1(state_vector))
        output = self.fc2(x)
        return output.squeeze(-1)  # [batch] or scalar

    def train_step(self, state_batch, target_batch, optimizer, loss_fn=nn.MSELoss()):
        self.train()
        optimizer.zero_grad()
        # 支持批量训练
        if isinstance(state_batch, np.ndarray):
            state_batch = torch.from_numpy(state_batch).float().to(self.device)
        if isinstance(target_batch, np.ndarray):
            target_batch = torch.from_numpy(target_batch).float().to(self.device)
        else:
            target_batch = torch.tensor(target_batch, dtype=torch.float32, device=self.device)
        pred = self.forward(state_batch)
        loss = loss_fn(pred, target_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
