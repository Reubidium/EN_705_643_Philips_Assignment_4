from utils import get_frames, store_frames
from pathlib import Path

if __name__ == "__main__":
    sample_video_path = Path("C:/Users/Ruben/JHU/Deep_Learning_Developments_with_PyTorch/assignments/EN_705_643_Assignment_4/UCF50/PlayingGuitar/v_PlayingGuitar_g01_c01.avi")
    if not sample_video_path.exists():
        raise FileNotFoundError(f"Sample video file not found: {sample_video_path}")
    frames, total_frames = get_frames(str(sample_video_path), n_frames=16)
    print("Requested 16 frames")
    print(f"Total frames in video: {total_frames}")
    if frames:
        output_dir = Path("C:/Users/Ruben/JHU/Deep_Learning_Developments_with_PyTorch/assignments/EN_705_643_Assignment_4/test/frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        store_frames(frames, str(output_dir))
        print(f"Stored {len(frames)} frames in: {output_dir}")
        total_saved = len(list(output_dir.glob("*.jpg")))
        print(f"Total frames saved in directory: {total_saved}")
    else:
        print("No frames were extracted from the video.")
