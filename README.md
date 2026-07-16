# CDPVER: Privacy-preserving Video Stream Recognition via Conditional Diffusion Model in Emotion Service Computing


## Directory Structure

```
CDPVER/
├── README.md                         # This document
├── requirements.txt                  # Dependencies
├── extract_features.py               # Stage 1: EmotiEffLib feature extraction
├── extract_openface_au.py            # Stage 1b: OpenFace AU feature extraction
├── extract_pose_features.py          # Stage 1c: OpenPose pose feature extraction
├── prepare_lstm_data.py              # Stage 2: Binary soft matching compression + AU feature concatenation
├── add_dp_noise.py                   # Stage 3: PCA dimensionality reduction + L2 clipping + Gaussian noise
├── conditional_diffusion_model.py    # Stage 4: Conditional diffusion model refinement
└── train_lstm_flex.py                # Stage 5: AttnLSTM downstream classification training
```

## Installation

```bash
pip install -r requirements.txt
```

EmotiEffLib must be installed from the official source: `pip install emotiefflib`

## Dataset Preparation

This framework uses the **AFEW (Acted Facial Expressions in the Wild)** dataset. Organize the dataset as follows:

```
dataset/AFEW/
├── Train/
│   ├── AlignedFaces_LBPTOP_Points/Faces/
│   │   ├── <video_id_1>/
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── Angry/
│   │   ├── <video_id>.avi
│   │   └── ...
│   └── ...
└── Val/
    └── (same structure)
```

## Run

Execute the following commands in order:

```bash
# Stage 1: EmotiEffLib facial features
python extract_features.py

# Stage 1b: OpenFace AU features
python extract_openface_au.py

# Stage 1c: OpenPose pose features
python extract_pose_features.py

# Stage 2: Binary soft matching compression + AU concatenation
python prepare_lstm_data.py

# Stage 3: Differential privacy perturbation
python add_dp_noise.py \
    --epsilon_k 50.0 --epsilon_nk 10.0 --delta 1e-5 \
    --clip_threshold 5.0 --top_k 8 --pca_dim 64 \
    --input_files feature_cope/hook_npy_b2_7/train_X_au.npy \
                  feature_cope/hook_npy_b2_7/val_X_au.npy

# Stage 4: Conditional diffusion model refinement
python conditional_diffusion_model.py

# Stage 5: AttnLSTM downstream classification training
python train_lstm_flex.py \
    --train_x feature_cope/hook_npy_b2_7/train_X_au_dp_adaptive_pca.npy \
    --val_x   feature_cope/hook_npy_b2_7/val_X_au_dp_adaptive_pca.npy \
    --tag     dp_eps_k50_nk10 \
    --epochs  60 --batch 16 --lr 7.4e-4 --dropout 0.37 --seed 42
```

Path constants in each script (e.g., `DATA_ROOT`, `AU_FEATURE_ROOTS`) need to be modified according to the local dataset location.

## Key Parameters

| Parameter | Description | Default |
|------|------|--------|
| `pca_dim` | Target dimensionality for PCA reduction | 64 |
| `clip_threshold` | L2 norm clipping threshold | 5.0 |
| `epsilon_k` | Privacy budget for keyframes | 50.0 |
| `epsilon_nk` | Privacy budget for non-keyframes | 10.0 |
| `delta` | Privacy failure probability | 1e-5 |
| `top_k` | Number of keyframes per sample | 8 |
| `sim_threshold` | Binary soft matching similarity threshold | 0.98 |
| `hidden_dim` | LSTM hidden layer dimensionality | 64 |
| `dropout` | Dropout rate | 0.37 |
| `lr` | Learning rate | 7.4e-4 |
| `diffusion_steps` | Number of diffusion timesteps | 200 |
| `beta_start` | Start value of linear schedule | 1e-4 |
| `beta_end` | End value of linear schedule | 0.02 |
| `diffusion_hidden_dim` | Denoising network hidden dimensionality | 256 |
| `condition_dim` | Condition vector dimensionality (AU + rough probability) | 14 |


## Citation

If you use this framework, please cite the corresponding paper:

```
@misc{erivsdp2026,
  title={Privacy-preserving Video Stream Recognition via Conditional Diffusion Model in Emotion Service Computing},
  year={2026}
}
```

## License

This project is for academic research only.
