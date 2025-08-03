"""NAM model implemented with PyTorch."""
import torch.nn as nn

class NAM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super(NAM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def predict(self, X):
        import torch
        self.eval()
        with torch.no_grad():
            return self.forward(X).squeeze()

    def fit(self, X, y, epochs=100, lr=0.01):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        import torch.nn.functional as F

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = F.mse_loss(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()