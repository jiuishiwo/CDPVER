# conditional_diffusion_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据路径
DATA_ROOT = 'ERIVS-DP/feature_cope/hook_npy_b2_7'
AU_ROOT = 'ERIVS-DP/openface_output'
ROUGH_RESULTS_DIR = 'ERIVS-DP/feature_cope/denoise/au_guided_uvit_results'
OUTPUT_DIR = 'ERIVS-DP/feature_cope/denoise/au_guided_uvit_results'

# 扩散模型参数
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AUFeatureEncoder(nn.Module):
    """AU特征编码器"""
    def __init__(self, au_dim=30, hidden_dim=128, output_dim=7):
        super().__init__()
        self.au_dim = au_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # AU特征处理网络
        self.au_encoder = nn.Sequential(
            nn.Linear(au_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # 多尺度特征融合
        self.frame_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.sequence_encoder = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        
    def forward(self, au_features):
        """
        Args:
            au_features: [batch_size, seq_len, au_dim]
        Returns:
            au_probabilities: [batch_size, output_dim]
        """
        batch_size, seq_len, _ = au_features.shape
        
        # 帧级特征编码
        frame_features = self.au_encoder(au_features)  # [batch_size, seq_len, output_dim]
        
        # 序列级特征编码
        sequence_features, _ = self.sequence_encoder(au_features)  # [batch_size, seq_len, hidden_dim]
        
        # 注意力机制
        attended_features, _ = self.frame_attention(
            sequence_features, sequence_features, sequence_features
        )
        
        # 全局池化
        global_features = torch.mean(attended_features, dim=1)  # [batch_size, hidden_dim]
        
        # 生成AU概率分布
        au_probabilities = self.au_encoder(global_features)  # [batch_size, output_dim]
        
        return au_probabilities

class ConditionalDenoiseNetwork(nn.Module):
    """条件去噪网络"""
    def __init__(self, time_dim=128, condition_dim=14, hidden_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 主网络
        self.input_proj = nn.Linear(7 + time_dim + hidden_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 7),
        )
        
    def forward(self, x, timestep, condition):
        """
        Args:
            x: [batch_size, 7] - 噪声化的概率分布
            timestep: [batch_size] - 时间步
            condition: [batch_size, condition_dim] - 条件（AU概率 + 粗略分类概率）
        """
        # 时间编码
        t = self.time_mlp(timestep)
        
        # 条件编码
        c = self.condition_encoder(condition)
        
        # 特征融合
        x_input = torch.cat([x, t, c], dim=-1)
        h = self.input_proj(x_input)
        
        # 主网络
        output = self.net(h)
        
        return output

class ConditionalDiffusionModel(nn.Module):
    """条件扩散模型"""
    def __init__(self, time_dim=128, condition_dim=14, hidden_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # 噪声调度
        self.betas = self._linear_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 网络
        self.denoise_network = ConditionalDenoiseNetwork(time_dim, condition_dim, hidden_dim)
        self.au_encoder = AUFeatureEncoder()
        
    def _linear_beta_schedule(self):
        """线性噪声调度"""
        return torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS)
    
    def _extract(self, a, t, x_shape):
        """提取时间步对应的alpha值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_losses(self, denoise_model, x_start, t, condition, noise=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy, predicted_noise = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t, condition)
        
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        return loss
    
    def p_sample(self, model, x, t, t_index, condition):
        """单步去噪"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(1.0 - self.alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_cumprod_t = self._extract(1.0 / self.alphas_cumprod, t, x.shape)
        
        model_output = model(x, t, condition)
        
        pred_original = (x - sqrt_one_minus_alphas_cumprod_t * model_output) * sqrt_recip_alphas_cumprod_t
        
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t, x.shape)
        
        variance = 0
        if t_index > 0:
            noise = torch.randn_like(x)
            variance = ((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * betas_t
            variance = torch.sqrt(variance) * noise
        
        return pred_original + variance
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition, device):
        """去噪循环"""
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, DIFFUSION_STEPS)), desc='Denoising'):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, condition)
        
        return x
    
    def forward(self, x_start, condition, t=None):
        """前向传播"""
        if t is None:
            t = torch.randint(0, DIFFUSION_STEPS, (x_start.shape[0],), device=x_start.device).long()
        
        return self.p_losses(self.denoise_network, x_start, t, condition)

def load_rough_results():
    """加载粗略分类结果"""
    print("Loading rough classification results...")
    
    rough_probabilities = np.load(os.path.join(ROUGH_RESULTS_DIR, 'rough_probabilities.npy'))
    rough_predictions = np.load(os.path.join(ROUGH_RESULTS_DIR, 'rough_predictions.npy'))
    
    print(f"Rough probabilities shape: {rough_probabilities.shape}")
    print(f"Rough predictions shape: {rough_predictions.shape}")
    
    return rough_probabilities, rough_predictions

def load_au_features_for_diffusion(au_root, names, labels, max_seq_len=64):
    """加载AU特征用于扩散模型"""
    print("Loading AU features for diffusion...")
    
    emotion2idx = {
        "Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, 
        "Neutral": 4, "Sad": 5, "Surprise": 6
    }
    idx2emotion = {v: k for k, v in emotion2idx.items()}
    
    all_au = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
        'AU11_r', 'AU12_r', 'AU13_r', 'AU14_r', 'AU15_r', 'AU16_r', 'AU17_r', 'AU18_r',
        'AU20_r', 'AU22_r', 'AU23_r', 'AU24_r', 'AU25_r', 'AU26_r', 'AU27_r', 'AU28_r',
        'AU29_r', 'AU30_r', 'AU31_r', 'AU32_r', 'AU33_r', 'AU34_r'
    ]
    
    au_features = []
    
    for i, (name, label) in enumerate(zip(names, labels)):
        emotion = idx2emotion[label]
        video_id = name.replace('.npy', '')
        
        csv_path = os.path.join(au_root, 'val', emotion, video_id, f"{video_id}.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                au_vecs = []
                for au in all_au:
                    if au in df.columns:
                        au_vecs.append(df[au].values)
                    else:
                        au_vecs.append(np.zeros(df.shape[0]))
                
                au_feat = np.stack(au_vecs, axis=1)
                
                if au_feat.shape[0] > max_seq_len:
                    au_feat = au_feat[:max_seq_len]
                elif au_feat.shape[0] < max_seq_len:
                    pad = np.zeros((max_seq_len - au_feat.shape[0], au_feat.shape[1]), dtype=au_feat.dtype)
                    au_feat = np.concatenate([au_feat, pad], axis=0)
                
                au_features.append(au_feat)
                
            except Exception as e:
                print(f"Error loading AU features for {video_id}: {e}")
                au_feat = np.zeros((max_seq_len, len(all_au)), dtype=np.float32)
                au_features.append(au_feat)
        else:
            au_feat = np.zeros((max_seq_len, len(all_au)), dtype=np.float32)
            au_features.append(au_feat)
    
    au_features = np.array(au_features)
    print(f"AU features shape: {au_features.shape}")
    
    return au_features

def prepare_diffusion_data(rough_probabilities, au_features, labels):
    """准备扩散模型数据"""
    print("Preparing diffusion data...")
    
    # 编码AU特征
    au_encoder = AUFeatureEncoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    au_encoder.to(device)
    
    au_probabilities = []
    with torch.no_grad():
        for i in range(0, len(au_features), 32):
            batch_au = torch.tensor(au_features[i:i+32], dtype=torch.float32).to(device)
            batch_au_probs = au_encoder(batch_au)
            au_probabilities.append(batch_au_probs.cpu().numpy())
    
    au_probabilities = np.concatenate(au_probabilities, axis=0)
    
    # 组合条件（AU概率 + 粗略分类概率）
    conditions = np.concatenate([au_probabilities, rough_probabilities], axis=1)
    
    print(f"Conditions shape: {conditions.shape}")
    
    return conditions, au_probabilities

def train_diffusion_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4):
    """训练扩散模型"""
    print("Training diffusion model...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_x, batch_condition in train_loader:
            batch_x, batch_condition = batch_x.to(device), batch_condition.to(device)
            
            optimizer.zero_grad()
            loss = model(batch_x, batch_condition)
            loss = loss.mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_condition in val_loader:
                batch_x, batch_condition = batch_x.to(device), batch_condition.to(device)
                loss = model(batch_x, batch_condition)
                val_loss += loss.mean().item() * batch_x.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_diffusion_model.pth'))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_diffusion_model(model, val_loader, device, y_val):
    """评估扩散模型"""
    print("Evaluating diffusion model...")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_condition in val_loader:
            batch_x, batch_condition = batch_x.to(device), batch_condition.to(device)
            
            # 生成去噪后的概率分布
            denoised_probs = model.p_sample_loop(
                model.denoise_network, 
                batch_x.shape, 
                batch_condition, 
                device
            )
            
            # 应用softmax确保概率分布
            denoised_probs = F.softmax(denoised_probs, dim=-1)
            predictions = torch.argmax(denoised_probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(denoised_probs.cpu().numpy())
    
    accuracy = accuracy_score(y_val, all_predictions)
    print(f"Diffusion Model Accuracy: {accuracy:.2%}")
    
    return all_predictions, all_probabilities, accuracy

def main():
    """主函数"""
    print("=== Conditional Diffusion Model Training ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    rough_probabilities, rough_predictions = load_rough_results()
    
    # 加载标签和文件名
    y_val = np.load(os.path.join(DATA_ROOT, 'val_y.npy'))
    val_names = np.load(os.path.join(DATA_ROOT, 'val_names_au.npy'))
    
    # 加载AU特征
    au_features = load_au_features_for_diffusion(AU_ROOT, val_names, y_val)
    
    # 准备扩散模型数据
    conditions, au_probabilities = prepare_diffusion_data(rough_probabilities, au_features, y_val)
    
    # 创建数据加载器
    batch_size = 16  # 较小的批次大小用于扩散模型
    dataset = TensorDataset(
        torch.tensor(rough_probabilities, dtype=torch.float32),
        torch.tensor(conditions, dtype=torch.float32)
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = ConditionalDiffusionModel(
        time_dim=128,
        condition_dim=14,  # 7 (AU概率) + 7 (粗略分类概率)
        hidden_dim=256
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 训练模型
    train_losses, val_losses = train_diffusion_model(
        model, train_loader, val_loader, device, epochs=50, lr=1e-4
    )
    
    # 评估模型
    predictions, probabilities, accuracy = evaluate_diffusion_model(model, val_loader, device, y_val)
    
    # 保存结果
    np.save(os.path.join(OUTPUT_DIR, 'diffusion_predictions.npy'), np.array(predictions))
    np.save(os.path.join(OUTPUT_DIR, 'diffusion_probabilities.npy'), np.array(probabilities))
    np.save(os.path.join(OUTPUT_DIR, 'diffusion_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(OUTPUT_DIR, 'diffusion_val_losses.npy'), np.array(val_losses))
    
    # 保存结果摘要
    results_summary = {
        'diffusion_accuracy': accuracy,
        'best_val_loss': min(val_losses),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'diffusion_steps': DIFFUSION_STEPS,
        'condition_dim': 14,
        'hidden_dim': 256
    }
    
    with open(os.path.join(OUTPUT_DIR, 'diffusion_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("=== Conditional Diffusion Model Training Completed ===")
    print(f"Diffusion model accuracy: {accuracy:.2%}")
    print(f"All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
