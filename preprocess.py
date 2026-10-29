"""Extract uniformly sampled video frames into per-action cache folders."""

import argparse
from contextlib import nullcontext
from pathlib import Path

from utils import get_frames, store_frames


def positive_int(value: str) -> int:
    """Parse a strictly positive frame count for command-line arguments."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _process_action_directory(action_dir, output_dir, n_frames, report):
    """Extract each video in one action folder, preserving progress reporting."""
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


def process_dataset(
    dataset_dir: Path, output_dir: Path, n_frames: int = 16, debug: bool = False
) -> None:
    """Extract cached frames for each action and optionally write a debug log."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] - Dataset directory not found: {dataset_dir}")

    log_path = Path(__file__).resolve().parent / f"{output_dir.name}_dataset_debug_log.txt"
    if debug:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_context = log_path.open("w", encoding="utf-8") if debug else nullcontext()
    with log_context as debug_log:

        def report(message: str, warning: bool = False) -> None:
            """Write debug progress or display warnings when logging is disabled."""
            if debug_log is not None:
                debug_log.write(f"{message}\n")
            elif warning:
                print(message)

        action_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        total_actions = len(action_dirs)
        for action_count, action_dir in enumerate(action_dirs, start=1):
            report("=" * 50)
            report(f'Entering action directory "{action_dir.name}"')
            _process_action_directory(action_dir, output_dir, n_frames, report)
            if action_count % 10 == 0:
                print(
                    f"Finished processing {action_count} of " f"{total_actions} action folders..."
                )
    print("Done.")


def main() -> None:
    """Parse preprocessing settings and extract the requested video frames."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract frames from videos in an action-folder dataset."
    )
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
        help="Write progress details to a dataset debug log in the repository root.",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        debug=args.debug,
    )
    if args.debug:
        log_path = project_dir / f"{args.output_dir.name}_dataset_debug_log.txt"
        print(f"Debug log written to: {log_path}")


if __name__ == "__main__":
    main()
