import numpy as np
import os
import argparse
import glob
from sklearn.decomposition import PCA, IncrementalPCA

# 检查tqdm是否安装，如果未安装则提供友好提示
try:
    from tqdm import tqdm
except ImportError:
    print("警告: tqdm 未安装，将不显示进度条。建议安装: pip install tqdm")
    # 创建一个假的tqdm，使其在未安装时也能运行
    tqdm = lambda iterable, **kwargs: iterable

def clip_features(data, clip_threshold):
    """
    对每帧特征的L2范数进行裁剪。
    这是实现差分隐私的关键步骤，用于限定敏感度(sensitivity)。

    Args:
        data (np.array): 输入特征，形状为 [num_frames, feature_dim]。
        clip_threshold (float): L2范数裁剪阈值。

    Returns:
        np.array: 裁剪后的特征。
    """
    clipped_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        frame = data[i]
        norm = np.linalg.norm(frame)
        if norm > clip_threshold:
            # 范数超过阈值，则缩放到阈值大小
            clipped_data[i] = frame * (clip_threshold / norm)
        else:
            clipped_data[i] = frame
    return clipped_data

def add_dp_gaussian_noise(data, epsilon, delta, sensitivity):
    """
    为数据添加高斯噪声以实现 (epsilon, delta)-差分隐私。

    Args:
        data (np.array): 输入特征。
        epsilon (float): 隐私预算，越小隐私保护越强，噪声越大。
        delta (float): 隐私失效概率，通常是一个非常小的值。
        sensitivity (float): 敏感度，即L2范数裁剪阈值。

    Returns:
        np.array: 添加了噪声的特征。
    """
    # 根据(epsilon, delta)-DP的定义计算高斯噪声的标准差
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def frame_attention_scores(arr):
    # 以每帧与全局均值的欧氏距离为"重要性分数"
    mean_vec = np.mean(arr, axis=0)
    scores = np.linalg.norm(arr - mean_vec, axis=1)
    return scores

def add_dp_noise_per_frame(arr, key_idx, epsilon_k, epsilon_nk, delta, clip_threshold):
    # arr: [num_frames, feature_dim]
    # key_idx: 关键帧索引集合
    noisy_arr = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        if i in key_idx:
            epsilon = epsilon_k
        else:
            epsilon = epsilon_nk
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * clip_threshold / epsilon
        noise = np.random.normal(0, sigma, arr.shape[1])
        noisy_arr[i] = arr[i] + noise
    return noisy_arr

def process_files(input_files, epsilon_k, epsilon_nk, delta, clip_threshold, top_k=8, pca_dim=64, output_suffix=""):
    """
    处理输入的.npy文件列表，先PCA降维，再clip和动态加噪。
    """
    print(f"准备处理 {len(input_files)} 个文件...")
    # 1. 增量PCA fit（内存优化：每个文件采样最多10000帧，累计最多50000帧用于fit）
    max_fit_frames = 50000
    sampled_frames = []
    sampled_count = 0
    feature_dim = None
    for filepath in input_files:
        if not os.path.exists(filepath):
            tqdm.write(f"警告: 文件不存在，已跳过: {filepath}")
            continue
        all_samples_features = np.load(filepath)
        if feature_dim is None:
            feature_dim = all_samples_features.shape[2]
        for i in range(all_samples_features.shape[0]):
            sample_features = all_samples_features[i]
            non_zero = sample_features[np.any(sample_features != 0, axis=1)]
            if len(non_zero) > 0:
                # 每个样本最多采样100帧
                step = max(1, len(non_zero) // 100)
                sampled = non_zero[::step]
                sampled_frames.append(sampled)
                sampled_count += len(sampled)
                del non_zero
            if sampled_count >= max_fit_frames:
                break
        del all_samples_features
        if sampled_count >= max_fit_frames:
            break
    if sampled_count == 0:
        print("未找到有效特征帧，终止。"); return
    # 合并采样帧（最多50000帧，每帧7170维，总约1.3 GiB，可接受）
    fit_frames = np.concatenate(sampled_frames, axis=0)[:max_fit_frames]
    del sampled_frames
    print(f"PCA fit 采样帧数: {fit_frames.shape[0]}")
    pca = IncrementalPCA(n_components=pca_dim, batch_size=50000)
    pca.fit(fit_frames)
    del fit_frames
    print(f"PCA降维完成，输出维度: {pca_dim}")

    # 2. 正式处理每个文件
    for filepath in tqdm(input_files, desc="总进度", unit="file"):
        if not os.path.exists(filepath):
            tqdm.write(f"警告: 文件不存在，已跳过: {filepath}")
            continue
        all_samples_features = np.load(filepath)
        noisy_samples_list = []
        for i in tqdm(range(all_samples_features.shape[0]), desc=f"-> {os.path.basename(filepath)}", leave=False, unit="sample"):
            sample_features = all_samples_features[i]
            # PCA降维（对每个样本的所有帧）
            reduced = pca.transform(sample_features)
            # 1. 裁剪特征L2范数 (在计算分数和加噪前都应先裁剪)
            clipped_features = clip_features(reduced, clip_threshold)
            # 2. 用注意力机制动态识别关键帧索引
            scores = frame_attention_scores(clipped_features)
            non_zero_indices = np.where(np.all(clipped_features != 0, axis=1))[0]
            if len(non_zero_indices) > top_k:
                top_indices_in_non_zero = np.argsort(scores[non_zero_indices])[-top_k:]
                key_idx = non_zero_indices[top_indices_in_non_zero]
            else:
                key_idx = non_zero_indices
            key_idx_set = set(key_idx)
            # 3. 为裁剪后的样本添加动态噪声
            noisy_features = add_dp_noise_per_frame(clipped_features, key_idx_set, epsilon_k, epsilon_nk, delta, clip_threshold)
            noisy_samples_list.append(noisy_features)
        final_noisy_features = np.stack(noisy_samples_list)
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        output_name = f"{name}_dp_adaptive_pca{output_suffix}{ext}"
        output_path = os.path.join(dir_name, output_name)
        np.save(output_path, final_noisy_features)
    print("-" * 30)
    print("所有文件处理完成！（PCA降维+动态隐私预算分配）")
    print(f"参数: epsilon_k={epsilon_k}, epsilon_nk={epsilon_nk}, delta={delta}, clip_threshold={clip_threshold}, top_k={top_k}, pca_dim={pca_dim}, output_suffix={output_suffix}")
    print("-" * 30)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='差分隐私加噪和降维')
    parser.add_argument('--epsilon_k', type=float, default=50.0, help='关键帧epsilon')
    parser.add_argument('--epsilon_nk', type=float, default=10.0, help='非关键帧epsilon')
    parser.add_argument('--clip_threshold', type=float, default=5.0, help='L2范数裁剪阈值')
    parser.add_argument('--delta', type=float, default=1e-5, help='隐私失效概率')
    parser.add_argument('--top_k', type=int, default=8, help='每个样本动态选top_k关键帧')
    parser.add_argument('--pca_dim', type=int, default=64, help='PCA降维目标维度')
    parser.add_argument('--output_suffix', type=str, default="", help='输出文件名后缀')
    parser.add_argument('--input_files', nargs='+', default=[
        'ERIVS-DP/feature_cope/hook_npy_b2_7/train_X_au.npy',
        'ERIVS-DP/feature_cope/hook_npy_b2_7/val_X_au.npy'
    ], help='输入文件列表')
    
    args = parser.parse_args()
    
    # 处理文件
    process_files(
        args.input_files,
        args.epsilon_k,
        args.epsilon_nk,
        args.delta,
        args.clip_threshold,
        args.top_k,
        args.pca_dim,
        args.output_suffix
    ) 