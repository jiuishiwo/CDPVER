# train_lstm.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'feature_cope', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 加载训练集和验证集
X_train = np.load('ERIVS-DP/feature_cope/hook_npy_b2_7/train_X_au_dp_adaptive_pca.npy')
y_train = np.load('ERIVS-DP/feature_cope/hook_npy_b2_7/train_y.npy')
X_val = np.load('ERIVS-DP/feature_cope/hook_npy_b2_7/val_X_au_dp_adaptive_pca.npy')
y_val = np.load('ERIVS-DP/feature_cope/hook_npy_b2_7/val_y.npy')

BATCH_SIZE = 16
EPOCHS = 100
NUM_CLASSES = 7
HIDDEN_DIM = 64
NUM_LAYERS = 1

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class VideoLSTMClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, num_classes, dropout=0.37):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, return_attention=False, return_features=False):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        attn_applied = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        features = self.dropout(attn_applied)
        out = self.fc(features)
        if return_features:
            return out, features
        elif return_attention:
            return out, attn_weights
        return out

feature_dim = X_train.shape[2]
model = VideoLSTMClassifier(feature_dim, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, dropout=0.37)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00074)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# 记录最佳验证准确率
best_val_acc = 0.0
best_model_path = os.path.join(MODEL_DIR, 'best_lstm_model.pth')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2%}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    scheduler.step(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> New best model saved! (Val Acc: {val_acc:.2%})")

# 加载最佳模型进行最终推理
if os.path.exists(best_model_path):
    print(f"\nLoading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Best validation accuracy during training: {best_val_acc:.2%}")

# 推理
model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    outputs = model(X_val_tensor)
    preds = torch.argmax(outputs, dim=1)
    acc = (preds.cpu().numpy() == y_val).mean()
    print(f"Final Validation Accuracy: {acc:.2%}")