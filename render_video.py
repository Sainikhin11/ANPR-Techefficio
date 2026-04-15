import argparse
import os
import sys
import time
from pathlib import Path

import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from run_anpr import add_recent_plate, load_config
from src.core.anpr_controller import ANPRController
from src.utils.gpu import check_gpu
from src.utils.logger import setup_logger


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


def get_input_videos(input_dir="input"):
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    )
    return videos


def get_next_output_dir(root_dir="output"):
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    index = 1
    while (root_path / f"output_{index}").exists():
        index += 1

    run_dir = root_path / f"output_{index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def draw_recent_plates(frame, recent_plates):
    if not recent_plates:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()
    box_height = 50 + (len(recent_plates) * 35)
    cv2.rectangle(overlay, (w - 350, 10), (w - 20, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        "Extracted Plates",
        (w - 330, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    for index, (plate, conf) in enumerate(recent_plates):
        cv2.putText(
            frame,
            f"{plate} ({conf:.2f})",
            (w - 330, 80 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


def render_video(config_path, source, output, max_frames=0, open_output=True, log_file=None):
    config = load_config(config_path)
    config.setdefault("output", {})
    config["output"]["root_dir"] = os.path.dirname(output) or "output"

    effective_log_file = log_file or os.path.join(config["output"]["root_dir"], "anpr_system.log")
    setup_logger(config_path, log_file_override=effective_log_file)
    check_gpu()

    video_source = str(source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps if input_fps and input_fps > 0 else config["camera"].get("fps", 30)
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (960, 540),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output}")

    controller = ANPRController(config)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    recent_plates = []
    frame_idx = 0
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rendered_frame, events = controller.process_frame(frame)
            for event in events:
                add_recent_plate(recent_plates, event["plate_number"], event["confidence"])
            if len(recent_plates) > 5:
                recent_plates = recent_plates[:5]

            rendered_frame = draw_recent_plates(rendered_frame, recent_plates)
            writer.write(rendered_frame)

            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.time() - start
                fps = frame_idx / elapsed if elapsed > 0 else 0.0
                if total_frames:
                    print(f"Rendered {frame_idx}/{total_frames} frames ({fps:.2f} FPS) for {Path(video_source).name}")
                else:
                    print(f"Rendered {frame_idx} frames ({fps:.2f} FPS) for {Path(video_source).name}")

            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start
    fps = frame_idx / elapsed if elapsed > 0 else 0.0
    summary = {
        "source": video_source,
        "output": output,
        "processed_frames": frame_idx,
        "average_fps": fps,
        "plates": controller.get_final_plates(),
        "stats": controller.get_stats(),
    }

    print(f"Saved output video: {output}")
    print(f"Processed frames: {frame_idx}")
    print(f"Average processing FPS: {fps:.2f}")
    if summary["plates"]:
        print("Detected plates:")
        for plate, conf in summary["plates"]:
            print(f"  {plate} ({conf:.2f})")

    if open_output:
        try:
            os.startfile(os.path.abspath(output))
        except Exception as exc:
            print(f"Could not auto-play output video: {exc}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Render ANPR annotations into an output video.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--source", required=True, help="Video path to process.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick tests.")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-play the output after rendering.")
    args = parser.parse_args()

    render_video(args.config, args.source, args.output, args.max_frames, open_output=not args.no_open)


if __name__ == "__main__":
    main()
