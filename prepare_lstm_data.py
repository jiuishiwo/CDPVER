# prepare_lstm_data.py
import os
import numpy as np
import pandas as pd
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError('请先安装 scikit-learn 库: pip install scikit-learn')

DATASET_ROOT = r'F:\Computer\Works\Postgraduate\EmotiEffLib-main\dataset\AFEW'
TRAIN_FEATURE_SAVE_DIR = os.path.join(DATASET_ROOT, 'Train', 'video_features_original_b2_7')
VAL_FEATURE_SAVE_DIR = os.path.join(DATASET_ROOT, 'Val', 'video_features_original_b2_7')
TRAIN_ROOT = os.path.join(DATASET_ROOT, 'Train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'Val')
MAX_SEQ_LEN = 64  # 统一序列长度，长的截断，短的补零

# AU特征根目录
AU_FEATURE_ROOTS = {
    'train': r'F:\Computer\Works\Postgraduate\EmotiEffLib-main\ERIVS-DP\openface_output\train',
    'val': r'F:\Computer\Works\Postgraduate\EmotiEffLib-main\ERIVS-DP\openface_output\val',
}

# 标签映射
emotion2idx = {
    "Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Neutral": 4, "Sad": 5, "Surprise": 6
}

# 定义每类情感的AU（增强版本）
emotion2au = {
    "Angry": [  # 愤怒
        'AU04_r',  # 眉间皱纹
        'AU05_r',  # 上眼睑提升器
        'AU07_r',  # 眼睑收紧器
        'AU09_r',  # 鼻子皱纹
        'AU10_r',  # 上唇提升器
        'AU11_r',  # 鼻唇沟加深器
        'AU15_r',  # 嘴角下降器
        'AU16_r',  # 下唇下降器
        'AU17_r',  # 下巴提升器
        'AU18_r',  # 嘴唇收紧器
        'AU20_r',  # 嘴唇拉伸器
        'AU22_r',  # 嘴唇漏斗器
        'AU23_r',  # 嘴唇收紧器
        'AU24_r',  # 嘴唇按压器
        'AU25_r',  # 嘴唇分离器
        'AU26_r',  # 下巴下降器
        'AU27_r',  # 嘴巴拉伸器
        'AU28_r',  # 嘴唇吮吸器
        'AU29_r',  # 下巴推力器
        'AU30_r',  # 下巴侧移器
    ],
    
    "Disgust": [  # 厌恶
        'AU09_r',  # 鼻子皱纹
        'AU10_r',  # 上唇提升器
        'AU11_r',  # 鼻唇沟加深器
        'AU15_r',  # 嘴角下降器
        'AU16_r',  # 下唇下降器
        'AU17_r',  # 下巴提升器
        'AU18_r',  # 嘴唇收紧器
        'AU20_r',  # 嘴唇拉伸器
        'AU22_r',  # 嘴唇漏斗器
        'AU23_r',  # 嘴唇收紧器
        'AU24_r',  # 嘴唇按压器
        'AU25_r',  # 嘴唇分离器
        'AU26_r',  # 下巴下降器
        'AU27_r',  # 嘴巴拉伸器
        'AU28_r',  # 嘴唇吮吸器
    ],
    
    "Fear": [  # 恐惧
        'AU01_r',  # 内眉提升器
        'AU02_r',  # 外眉提升器
        'AU04_r',  # 眉间皱纹
        'AU05_r',  # 上眼睑提升器
        'AU07_r',  # 眼睑收紧器
        'AU09_r',  # 鼻子皱纹
        'AU10_r',  # 上唇提升器
        'AU11_r',  # 鼻唇沟加深器
        'AU15_r',  # 嘴角下降器
        'AU16_r',  # 下唇下降器
        'AU17_r',  # 下巴提升器
        'AU20_r',  # 嘴唇拉伸器
        'AU22_r',  # 嘴唇漏斗器
        'AU25_r',  # 嘴唇分离器
        'AU26_r',  # 下巴下降器
        'AU27_r',  # 嘴巴拉伸器
    ],
    
    "Happy": [  # 快乐
        'AU06_r',  # 脸颊提升器
        'AU12_r',  # 嘴角提升器（微笑）
        'AU13_r',  # 脸颊提升器
        'AU14_r',  # 酒窝
        'AU15_r',  # 嘴角下降器（轻微）
        'AU16_r',  # 下唇下降器
        'AU20_r',  # 嘴唇拉伸器
        'AU25_r',  # 嘴唇分离器
        'AU26_r',  # 下巴下降器
    ],
    
    "Neutral": [  # 中性
        # 中性表情通常AU活动较少，但可以包含一些基础AU
        'AU01_r',  # 内眉提升器
        'AU02_r',  # 外眉提升器
        'AU04_r',  # 眉间皱纹
        'AU12_r',  # 嘴角提升器
        'AU15_r',  # 嘴角下降器
        'AU25_r',  # 嘴唇分离器
    ],
    
    "Sad": [  # 悲伤
        'AU01_r',  # 内眉提升器
        'AU04_r',  # 眉间皱纹
        'AU06_r',  # 脸颊提升器
        'AU07_r',  # 眼睑收紧器
        'AU09_r',  # 鼻子皱纹
        'AU10_r',  # 上唇提升器
        'AU11_r',  # 鼻唇沟加深器
        'AU15_r',  # 嘴角下降器
        'AU17_r',  # 下巴提升器
        'AU20_r',  # 嘴唇拉伸器
        'AU25_r',  # 嘴唇分离器
    ],
    
    "Surprise": [  # 惊讶
        'AU01_r',  # 内眉提升器
        'AU02_r',  # 外眉提升器
        'AU05_r',  # 上眼睑提升器
        'AU15_r',  # 嘴角下降器
        'AU16_r',  # 下唇下降器
        'AU20_r',  # 嘴唇拉伸器
        'AU25_r',  # 嘴唇分离器
        'AU26_r',  # 下巴下降器
        'AU27_r',  # 嘴巴拉伸器
    ]
}
# 所有可能出现的AU
all_au = sorted({au for aus in emotion2au.values() for au in aus})

def build_videoid2label_map(root):
    videoid2label = {}
    for label in os.listdir(root):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith('.avi') or fname.endswith('.mp4'):
                video_id = os.path.splitext(fname)[0]
                videoid2label[video_id] = label
    return videoid2label

def load_au_feature(au_root, label, video_id, max_seq_len, emotion=None):
    csv_path = os.path.join(au_root, label, video_id, f"{video_id}.csv")
    au_dim = len(all_au)
    if not os.path.exists(csv_path):
        # 没有AU特征，返回全零
        return np.zeros((max_seq_len, au_dim), dtype=np.float32)
    df = pd.read_csv(csv_path)
    au_vecs = []
    for au in all_au:
        if emotion and au in emotion2au[emotion]:
            if au in df.columns:
                au_vecs.append(df[au].values)
            else:
                au_vecs.append(np.zeros(df.shape[0]))
        else:
            au_vecs.append(np.zeros(df.shape[0]))
    au_feat = np.stack(au_vecs, axis=1)  # shape: (帧数, len(all_au))
    # 截断/补零
    if au_feat.shape[0] > max_seq_len:
        au_feat = au_feat[:max_seq_len]
    elif au_feat.shape[0] < max_seq_len:
        pad = np.zeros((max_seq_len - au_feat.shape[0], au_feat.shape[1]), dtype=au_feat.dtype)
        au_feat = np.concatenate([au_feat, pad], axis=0)
    return au_feat

def binary_soft_matching(arr, sim_threshold=0.98):
    """
    二分软匹配法压缩序列特征
    arr: [num_frames, feature_dim]
    返回: [num_segments, feature_dim]
    """
    def recursive_segment(start, end):
        if start == end:
            return [arr[start]]
        sim = cosine_similarity(arr[start].reshape(1, -1), arr[end].reshape(1, -1))[0, 0]
        if sim >= sim_threshold:
            # 整段合并
            return [np.mean(arr[start:end+1], axis=0)]
        else:
            mid = (start + end) // 2
            left = recursive_segment(start, mid)
            right = recursive_segment(mid+1, end)
            return left + right
    return np.array(recursive_segment(0, len(arr)-1))

def frame_attention_scores(arr):
    # 以每帧与全局均值的欧氏距离为"重要性分数"
    mean_vec = np.mean(arr, axis=0)
    scores = np.linalg.norm(arr - mean_vec, axis=1)
    return scores

def attention_guided_merge(arr, top_k=8, sim_threshold=0.98):
    scores = frame_attention_scores(arr)
    # 选出top_k个分数最高的帧（关键帧）
    top_idx = np.argsort(scores)[-top_k:]
    mask = np.zeros(len(arr), dtype=bool)
    mask[top_idx] = True
    # 保留关键帧，其余帧用二分软匹配法合并
    compressed = []
    i = 0
    while i < len(arr):
        if mask[i]:
            compressed.append(arr[i])
            i += 1
        else:
            # 找到下一个关键帧或结尾
            j = i + 1
            while j < len(arr) and not mask[j]:
                j += 1
            # 合并[i, j-1]区间
            if j > i:
                merged = binary_soft_matching(arr[i:j], sim_threshold)
                compressed.extend(merged)
            i = j
    return np.array(compressed)

def process_split(feature_dir, root, out_prefix, au_root):
    videoid2label = build_videoid2label_map(root)
    X, y, names = [], [], []
    for fname in os.listdir(feature_dir):
        if not fname.endswith('.npy'):
            continue
        video_id = fname.replace('.npy', '')
        label = videoid2label.get(video_id)
        if label not in emotion2idx:
            continue
        arr = np.load(os.path.join(feature_dir, fname))  # [num_frames, feature_dim]
        # 先用统计型注意力引导的帧合并策略
        arr = attention_guided_merge(arr, top_k=8, sim_threshold=0.98)
        # 读取AU特征，按方案一拼接
        au_feat = load_au_feature(au_root, label, video_id, arr.shape[0], emotion=label)
        # 拼接AU特征
        arr = np.concatenate([arr, au_feat], axis=1)  # [压缩后帧数, feature_dim+AU_dim]
        # 补零/截断到MAX_SEQ_LEN
        if arr.shape[0] > MAX_SEQ_LEN:
            arr = arr[:MAX_SEQ_LEN]
        elif arr.shape[0] < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        # 计算均值和方差
        mean_feat = np.mean(arr, axis=0)   # [feature_dim]
        std_feat = np.std(arr, axis=0)     # [feature_dim]
        max_feat = np.max(arr, axis=0)     # [feature_dim]
        min_feat = np.min(arr, axis=0)     # [feature_dim]
        stat_feat = np.concatenate([mean_feat, std_feat, max_feat, min_feat])  # [feature_dim*4]
        stat_feat = np.tile(stat_feat, (arr.shape[0], 1))  # [64, feature_dim*4]
        arr_with_stat = np.concatenate([arr, stat_feat], axis=1)  # [64, feature_dim*5]
        X.append(arr_with_stat)
        y.append(emotion2idx[label])
        names.append(fname)  # 保存特征文件名
    X = np.stack(X)
    y = np.array(y)
    names = np.array(names)
    np.save(f'ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_X_au.npy', X)
    np.save(f'ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_y_au.npy', y)
    np.save(f'ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_names_au.npy', names)  # 新增：保存文件名
    print(f'Saved ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_X_au.npy, ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_y_au.npy, ERIVS-DP/feature_cope/hook_npy_b2_7/{out_prefix}_names_au.npy:', X.shape, y.shape, names.shape)

if __name__ == '__main__':
    process_split(TRAIN_FEATURE_SAVE_DIR, TRAIN_ROOT, 'train', AU_FEATURE_ROOTS['train'])
    process_split(VAL_FEATURE_SAVE_DIR, VAL_ROOT, 'val', AU_FEATURE_ROOTS['val'])
