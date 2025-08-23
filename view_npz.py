import cv2
import numpy as np
import sys

"""Keys in round_001.npz:
  'images': shape=(1215, 3, 448, 736), dtype=uint8  
  'actions': shape=(1215,), dtype=int64
  'states': shape=(1215, 3), dtype=float64
  'attention_mask': shape=(1215,), dtype=uint8      """
def view_npz_video(arr, fps=30):
    print(f"Displaying array: shape={arr.shape}, dtype={arr.dtype}")
    # If shape is (N, 3, H, W), transpose to (N, H, W, 3)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_npz.py <file.npz> [key] [fps]")
        sys.exit(1)
    npz_path = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else None
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    data = np.load(npz_path)
    if key is None:
        key = data.files[0]
    arr = data[key]
    view_npz_video(arr, fps)
#usage :  python view_npz.py round_006.npz  images 30
#python view_npz.py round_001_with_poses.npz  left_player_pose 30