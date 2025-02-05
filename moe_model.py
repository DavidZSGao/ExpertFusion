import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# Define GatingNetwork and MoEModel classes
class GatingNetwork(nn.Module):
    def __init__(self, num_experts=5):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(num_experts)
        self.fc = nn.Linear(num_experts, num_experts)
    def forward(self, expert_preds: torch.Tensor):
        x = self.input_norm(expert_preds)
        logits = self.fc(x)
        weights = F.softmax(logits / 0.5, dim=1)
        return weights

class MoEModel(nn.Module):
    def __init__(self, num_experts=5):
        super().__init__()
        self.gate = GatingNetwork(num_experts)
        self.scale = nn.Parameter(torch.tensor([0.05]))
    def forward(self, expert_preds: torch.Tensor):
        weights = self.gate(expert_preds)
        combined_pred = (expert_preds * weights).sum(dim=1)
        return torch.tanh(combined_pred) * self.scale

def combined_loss(pred: torch.Tensor, target: torch.Tensor, model: nn.Module) -> torch.Tensor:
    huber = F.huber_loss(pred, target, reduction='mean', delta=0.1)
    pred_direction = torch.sign(pred)
    target_direction = torch.sign(target)
    direction_loss = -torch.mean(pred_direction * target_direction)
    scale_reg = 0.1 * model.scale**2
    total_loss = huber + 0.3 * direction_loss + scale_reg
    return total_loss

def train_moe_model(recs: List[Dict], epochs=200):
    X = torch.tensor([r['expert_predictions'] for r in recs], dtype=torch.float32)
    y = torch.tensor([r['target'] for r in recs], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    model = MoEModel(num_experts=X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = combined_loss(outputs, batch_y, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += combined_loss(outputs, batch_y, model).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    return model