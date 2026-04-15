import csv
import os
from pathlib import Path

from render_video import get_input_videos, get_next_output_dir, render_video
from run_anpr import load_config
from tools.metrics import MetricsTracker, compute_cer
from src.utils.logger import setup_logger, log


def load_ground_truth(input_dir):
    gt_path = Path(input_dir) / "ground_truth.txt"
    mapping = {}
    if not gt_path.exists():
        return mapping
    if not gt_path.is_file():
        print(f"Skipping ground truth load because {gt_path} is not a file.")
        return mapping

    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "," not in line:
                continue
            video_name, plate = line.split(",", 1)
            mapping[video_name.strip()] = plate.strip().upper()
    return mapping


def choose_best_prediction(plates, ground_truth):
    if not plates:
        return ""

    best_plate = ""
    best_score = None
    for plate, conf in plates:
        cer = compute_cer(plate, ground_truth)
        score = (cer, -conf)
        if best_score is None or score < best_score:
            best_score = score
            best_plate = plate
    return best_plate


def write_run_summary(run_dir, results):
    summary_path = Path(run_dir) / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "input_video",
                "output_video",
                "processed_frames",
                "average_fps",
                "plates_detected",
                "plates_read",
                "detected_plates",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["input_video"],
                    result["output_video"],
                    result["processed_frames"],
                    f"{result['average_fps']:.2f}",
                    result["plates_detected"],
                    result["plates_read"],
                    "; ".join(f"{plate} ({conf:.2f})" for plate, conf in result["plates"]),
                ]
            )
    return str(summary_path)


def write_metrics(run_dir, metrics, gt_count):
    metrics_path = Path(run_dir) / "metrics.txt"
    acc, cer = metrics.report()
    with metrics_path.open("w", encoding="utf-8") as handle:
        if gt_count:
            handle.write(f"Accuracy: {acc * 100:.2f}%\n")
            handle.write(f"CER: {cer:.4f}\n")
        else:
            handle.write("Accuracy: N/A (ground truth not provided)\n")
            handle.write("CER: N/A (ground truth not provided)\n")
        handle.write(f"Plates detected: {metrics.plates_detected}\n")
        handle.write(f"Plates successfully read: {metrics.plates_read}\n")
    return str(metrics_path)


def main():
    config_path = "config/config.yaml"
    config = load_config(config_path)
    input_dir = config.get("paths", {}).get("input_dir", "input")
    output_root = config.get("paths", {}).get("output_dir", "output")
    ground_truth = load_ground_truth(input_dir)
    metrics = MetricsTracker()

    videos = get_input_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"No input videos found in {input_dir}/")

    run_dir = get_next_output_dir(output_root)
    run_log_path = os.path.join(run_dir, "anpr_system.log")
    setup_logger(config_path, log_file_override=run_log_path)
    print(f"Created run output directory: {run_dir}")
    log.info(f"Created run output directory: {run_dir}")

    results = []
    for video_path in videos:
        output_path = os.path.join(run_dir, f"{video_path.stem}.mp4")
        print(f"Processing: {video_path}")
        log.info(f"Processing video: {video_path}")
        summary = render_video(
            config_path,
            str(video_path),
            output_path,
            open_output=True,
            log_file=run_log_path,
        )
        results.append(
            {
                "input_video": str(video_path),
                "output_video": summary["output"],
                "processed_frames": summary["processed_frames"],
                "average_fps": summary["average_fps"],
                "plates_detected": summary["stats"]["plates_detected"],
                "plates_read": summary["stats"]["plates_read"],
                "plates": summary["plates"],
            }
        )
        metrics.add_counts(summary["stats"]["plates_detected"], summary["stats"]["plates_read"])

        gt_plate = ground_truth.get(video_path.name)
        if gt_plate:
            predicted_plate = choose_best_prediction(summary["plates"], gt_plate)
            metrics.update(predicted_plate, gt_plate)

    summary_path = write_run_summary(run_dir, results)
    metrics_path = write_metrics(run_dir, metrics, len(ground_truth))
    acc, cer = metrics.report()

    if ground_truth:
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Avg CER: {cer:.4f}")
    else:
        print("Accuracy: N/A (ground truth not provided)")
        print("Avg CER: N/A (ground truth not provided)")
    print(f"Number of plates detected: {metrics.plates_detected}")
    print(f"Number of plates successfully read: {metrics.plates_read}")
    print(f"Batch complete. Summary saved to: {summary_path}")
    print(f"Metrics saved to: {metrics_path}")
    log.info(f"Number of plates detected: {metrics.plates_detected}")
    log.info(f"Number of plates successfully read: {metrics.plates_read}")
    log.info(f"Batch complete. Summary saved to: {summary_path}")
    log.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
