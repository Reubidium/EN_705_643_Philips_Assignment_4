import argparse
from contextlib import nullcontext
from utils import get_frames, store_frames
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def process_dataset(dataset_dir: Path, output_dir: Path, n_frames: int = 16, debug: bool = False) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] - Dataset directory not found: {dataset_dir}")

    log_path = output_dir / "preprocess_debug.txt"
    if debug:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_context = log_path.open("w", encoding="utf-8") if debug else nullcontext()
    with log_context as debug_log:
        def report(message: str, warning: bool = False) -> None:
            if debug_log is not None:
                debug_log.write(f"{message}\n")
            elif warning:
                print(message)

        action_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        total_actions = len(action_dirs)
        for action_count, action_dir in enumerate(action_dirs, start=1):
            report("=" * 50)
            report(f"Entering action directory \"{action_dir.name}\"")
            for video_path in sorted(action_dir.iterdir()):
                if video_path.is_file():
                    frames, total_frames = get_frames(str(video_path), n_frames=n_frames)
                    if not frames:
                        report(f"[WARNING] No frames were extracted from: {video_path}", warning=True)
                        continue

                    video_output_dir = output_dir / action_dir.name / video_path.stem
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    store_frames(frames, str(video_output_dir))
                    report(
                        f"Stored {len(frames)} of {total_frames} frames from "
                        f"{video_path} in {video_output_dir}\n"
                    )
                else:
                    report(f"[WARNING] Skipping non-file entry: {video_path}", warning=True)
            if action_count % 10 == 0:
                print(
                    f"Finished processing {action_count} of "
                    f"{total_actions} action folders..."
                )
    print("Done.")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract frames from videos in an action-folder dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_dir / "UCF50",
        help="Directory containing action folders (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "UCF50_Captured_Frames",
        help="Directory where extracted frames will be saved (default: %(default)s).",
    )
    parser.add_argument(
        "--n-frames",
        type=positive_int,
        default=16,
        help="Number of frames to extract per video (default: %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write progress details to preprocess_debug.txt in the output directory.",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        debug=args.debug,
    )
    if args.debug:
        print(f"Debug log written to: {args.output_dir / 'preprocess_debug.txt'}")


if __name__ == "__main__":
    main()
