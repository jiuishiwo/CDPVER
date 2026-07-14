import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

# 输入视频根目录
INPUT_DIRS = [
    r"F:\Computer\Works\Postgraduate\EmotiEffLib-main\dataset\AFEW\Train",
    r"F:\Computer\Works\Postgraduate\EmotiEffLib-main\dataset\AFEW\Val"
]
# 输出根目录
OUTPUT_ROOT = r"F:\Computer\Works\Postgraduate\EmotiEffLib-main\ERIVS-DP\aligned"

def extract_and_detect_faces(video_path, save_dir, mtcnn, frame_interval=1):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing video: {video_path}, frame_interval={frame_interval}")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            print(f"End of video or cannot read frame at idx {frame_idx} in {video_path}")
            break

        # 如果不是要处理的帧，则跳过
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if img is None:
            print(f"Frame {frame_idx} is None in {video_path}")
            frame_idx += 1
            continue
        faces = mtcnn.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not faces:
            print(f"No face detected in frame {frame_idx} of {video_path}")
            frame_idx += 1
            continue
        # 只保留最大人脸
        largest = max(faces, key=lambda f: f['box'][2]*f['box'][3])
        x, y, w, h = largest['box']
        print(f"Detected face box: x={x}, y={y}, w={w}, h={h} in frame {frame_idx} of {video_path}")
        x, y = max(0, x), max(0, y)
        face_img = img[y:y+h, x:x+w]
        if face_img.size == 0:
            print(f"face_img.size == 0 for frame {frame_idx} in {video_path}, box=({x},{y},{w},{h})")
            frame_idx += 1
            continue
        try:
            face_img = cv2.resize(face_img, (112, 112))
        except Exception as e:
            print(f"cv2.resize error at frame {frame_idx} in {video_path}: {e}")
            frame_idx += 1
            continue
        face_save_path = os.path.join(save_dir, f"frame_{frame_idx:05d}_face1.jpg")
        success = cv2.imwrite(face_save_path, face_img)
        if success:
            print(f"Saved face to {face_save_path}")
        else:
            print(f"Failed to save face to {face_save_path}")
        frame_idx += 1
    cap.release()

def main():
    # 每隔 FRAME_INTERVAL 帧提取一次人脸，设置成 1 表示每帧都提
    FRAME_INTERVAL = 3

    mtcnn = MTCNN(min_face_size=20)
    all_video_tasks = []
    for input_root in INPUT_DIRS:
        for root, dirs, files in os.walk(input_root):
            for file in files:
                if file.lower().endswith('.avi') or file.lower().endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_root)
                    video_name = os.path.splitext(file)[0]
                    out_face_dir = os.path.join(OUTPUT_ROOT, os.path.basename(input_root), rel_path, video_name)
                    all_video_tasks.append((video_path, out_face_dir, input_root))
    print(f"\n==== 共 {len(all_video_tasks)} 个视频 ====")
    with tqdm(total=len(all_video_tasks), desc="总进度", ncols=100, unit="video") as pbar:
        last_input_root = None
        sub_pbar = None
        sub_tasks = []
        for idx, (video_path, out_face_dir, input_root) in enumerate(all_video_tasks):
            # 切换子进度条
            if last_input_root != input_root:
                if sub_pbar:
                    sub_pbar.close()
                sub_tasks = [t for t in all_video_tasks if t[2] == input_root]
                sub_pbar = tqdm(total=len(sub_tasks), desc=os.path.basename(input_root), ncols=80, leave=False, unit="video")
                last_input_root = input_root
            # 断点续跑：如果输出人脸目录下已有人脸图片则跳过
            if os.path.exists(out_face_dir) and any(f.endswith('.jpg') for f in os.listdir(out_face_dir)):
                tqdm.write(f"[跳过] {video_path} 已处理，{out_face_dir} 已有图片。")
                pbar.update(1)
                sub_pbar.update(1)
                continue
            tqdm.write(f"处理: {video_path}\n人脸输出到: {out_face_dir}")
            extract_and_detect_faces(video_path, out_face_dir, mtcnn, frame_interval=FRAME_INTERVAL)
            pbar.update(1)
            sub_pbar.update(1)
        if sub_pbar:
            sub_pbar.close()

if __name__ == "__main__":
    main() 