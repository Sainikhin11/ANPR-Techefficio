# Automatic Number Plate Recognition (ANPR) System

## 1. Project Overview

This project is a production-ready Automatic Number Plate Recognition system built for video-based vehicle plate detection and recognition. It processes recorded input videos, detects license plates using a trained YOLO model, tracks plates across frames, performs OCR using PaddleOCR, removes duplicate readings with fuzzy matching, and renders annotated result videos automatically.

The system is designed for final submission and demo usage with a simple batch workflow:

1. Put videos inside `input/`
2. Run `python run_all.py`
3. Collect rendered outputs from a new numbered folder inside `output/`

The current implementation keeps the existing trained detection weights intact, preserves the OCR engine, processes all detected vehicles instead of only the nearest one, and creates clean structured outputs for every run.

## 2. Problem Solved

The goal of this project is to automate number plate recognition from traffic or parking footage. A practical ANPR system must do more than detect one visible plate. It must:

- detect multiple vehicles in the same frame
- track detections across time
- avoid repeated OCR on the same object every frame
- correct noisy OCR text
- suppress duplicate plate outputs
- save usable outputs for evaluation and demo

This project addresses those requirements in a single pipeline.

## 3. Core Features

- Detects license plates from video frames using YOLO
- Tracks plates across frames using ByteTrack through Ultralytics tracking
- Performs OCR using PaddleOCR
- Processes all tracked vehicles in frame
- Throttles OCR calls per track to reduce unnecessary computation
- Skips very small crops to protect speed and reduce bad OCR reads
- Uses fuzzy duplicate removal with Levenshtein distance
- Saves one annotated output video per input video
- Creates numbered run folders such as `output/output_1`, `output/output_2`
- Automatically opens each output video after rendering
- Writes a `summary.csv` file for each batch run
- Supports GPU acceleration with CUDA, PyTorch, and PaddleOCR GPU

## 4. Technologies Used

### Python
Python is the main programming language used for the complete pipeline, orchestration, file handling, output generation, and video processing workflow.

### YOLO (Ultralytics)
The project uses Ultralytics YOLO for license plate detection and tracking. The file `models/license_plate_detector.pt` is the trained detection model used by the system. Detection is handled in [detector.py](c:/Users/VaishuNikhil/anpr_system/src/core/detector.py).

What YOLO is used for:

- detecting plate bounding boxes
- returning confidence scores
- integrating with tracking through `model.track(...)`

Important note:

- the model weights were not changed
- the detection pipeline remains based on the same trained model

### ByteTrack
ByteTrack is used through the Ultralytics tracking interface with the configuration file [bytetrack.yaml](c:/Users/VaishuNikhil/anpr_system/config/bytetrack.yaml).

What tracking is used for:

- assigning stable track IDs across frames
- keeping OCR history per object
- allowing OCR throttling per track instead of per frame
- reducing duplicate output from the same moving vehicle

### PaddleOCR
PaddleOCR is the OCR engine used to extract characters from cropped plate regions. OCR is implemented in [ocr.py](c:/Users/VaishuNikhil/anpr_system/src/core/ocr.py).

What PaddleOCR is used for:

- reading text from detected license plate crops
- returning recognized text and OCR confidence
- working with preprocessed plate crops for better recognition quality

OCR-related logic included in this project:

- grayscale conversion
- histogram equalization
- sharpening filter
- crop resizing for smaller plate regions
- regex-based final plate validation
- positional character correction for Indian number plate format

### OpenCV
OpenCV is used for all image and video-level operations.

What OpenCV is used for:

- opening input videos
- reading frames
- resizing frames
- cropping detected regions
- drawing bounding boxes and text overlays
- writing output videos
- opening results after processing

### PyTorch
PyTorch is used as the backend for running the YOLO model and GPU acceleration. The detector automatically checks CUDA availability and moves the model to GPU when available.

### Loguru
Loguru is used for structured logging in console and file output. Logging setup is implemented in [logger.py](c:/Users/VaishuNikhil/anpr_system/src/utils/logger.py).

### PyYAML
PyYAML is used to load runtime settings from [config.yaml](c:/Users/VaishuNikhil/anpr_system/config/config.yaml).

### GPUtil
GPUtil is used for GPU memory inspection and runtime monitoring inside [gpu.py](c:/Users/VaishuNikhil/anpr_system/src/utils/gpu.py).

### python-Levenshtein
This library is used for fuzzy duplicate suppression. It compares similar OCR strings and helps prevent repeated outputs for near-identical plate readings.

## 5. Project Structure

```text
anpr_system/
├── backup_cleanup/              # safely moved old outputs, logs, caches
├── config/
│   ├── bytetrack.yaml           # ByteTrack configuration
│   └── config.yaml              # runtime configuration
├── data/                        # optional legacy/support data
├── dataset/                     # existing dataset artifacts
├── input/                       # place input videos here
├── models/
│   └── license_plate_detector.pt
├── output/                      # numbered batch outputs created here
├── src/
│   ├── core/
│   │   ├── anpr_controller.py   # main ANPR logic
│   │   ├── detector.py          # YOLO detection and tracking
│   │   ├── ocr.py               # PaddleOCR pipeline
│   │   └── video_loader.py      # threaded video loader utility
│   └── utils/
│       ├── anpr_postprocess.py  # voting, fuzzy matching, helpers
│       ├── gpu.py               # GPU checks and VRAM stats
│       └── logger.py            # logging setup
├── render_video.py              # renders one annotated output video
├── run_all.py                   # main batch entry point
├── run_anpr.py                  # live/shared helper entry
├── requirements.txt
└── README.md
```

## 6. Main Files Explained

### [run_all.py](c:/Users/VaishuNikhil/anpr_system/run_all.py)
This is the main entry point for the final system.

Responsibilities:

- reads configuration
- scans all supported videos inside `input/`
- creates the next numbered output folder
- processes each video one by one
- saves results in the run folder
- writes a batch summary CSV

This is the file the user runs:

```bash
python run_all.py
```

### [render_video.py](c:/Users/VaishuNikhil/anpr_system/render_video.py)
This file handles one complete video render job.

Responsibilities:

- opens one input video
- creates the output writer
- runs the ANPR controller on every frame
- overlays recent detected plates on the frame
- saves the final annotated video
- auto-plays the output after completion

### [anpr_controller.py](c:/Users/VaishuNikhil/anpr_system/src/core/anpr_controller.py)
This is the main decision-making file in the project.

Responsibilities:

- receives tracked detections from the detector
- maintains per-track state
- runs OCR with controlled frequency
- stores OCR history for each tracked object
- applies voting and correction logic
- suppresses duplicate final outputs
- writes dataset artifacts for the current run

Important upgrade in this version:

- the older closest-object-only behavior was removed
- now every tracked plate in the frame is processed

### [detector.py](c:/Users/VaishuNikhil/anpr_system/src/core/detector.py)
This file loads the YOLO model and performs:

- pure detection with `predict`
- detection + tracking with `track`
- CUDA device selection
- model warmup for faster inference

### [ocr.py](c:/Users/VaishuNikhil/anpr_system/src/core/ocr.py)
This file wraps PaddleOCR and image preprocessing.

Responsibilities:

- initializes PaddleOCR
- enhances plate crops before OCR
- performs OCR prediction
- corrects OCR output for Indian plate formatting
- validates final text using regex

### [anpr_postprocess.py](c:/Users/VaishuNikhil/anpr_system/src/utils/anpr_postprocess.py)
This file contains supporting post-processing utilities such as:

- Levenshtein-based similarity checks
- voting across repeated OCR reads
- blur checking helpers
- motion and confidence helpers

## 7. Detailed Workflow

The full system pipeline works as follows:

### Step 1: Input video collection
`run_all.py` reads all supported videos from `input/`.

Supported formats currently include:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.mpeg`
- `.mpg`
- `.wmv`

### Step 2: Output folder creation
Before processing starts, the system creates the next numbered output folder:

```text
output/output_1
output/output_2
output/output_3
```

This prevents overwriting older results.

### Step 3: Frame reading
Each video is opened using OpenCV and processed frame by frame.

### Step 4: Detection and tracking
For each frame:

- YOLO detects license plate bounding boxes
- ByteTrack assigns track IDs
- all returned tracks are processed

This is important because the final system is no longer limited to only the largest or nearest plate.

### Step 5: OCR gating
OCR is not run on every frame for every track. That would be slow and unnecessary.

The project uses controlled OCR logic:

- OCR is attempted only on selected frames
- each track has `last_ocr_frame`
- if fewer than 10 frames have passed since the previous OCR call for that track, OCR is skipped
- crops smaller than `80 x 25` are ignored

This reduces FPS drop while keeping useful OCR updates.

### Step 6: OCR and text correction
When a crop is selected:

- the crop is preprocessed
- PaddleOCR reads the text
- the text is cleaned
- positional corrections are applied
- Indian plate format constraints are used to reject weak readings

### Step 7: Voting and stabilization
Each track stores OCR history over time. Instead of trusting a single read, the controller collects multiple readings and performs voting to decide a stable plate result.

This helps in cases where OCR output fluctuates slightly across frames.

### Step 8: Duplicate removal
Final plate outputs are deduplicated using fuzzy matching.

The logic is based on Levenshtein distance and same-length comparison:

- if two plate strings are very similar
- and the distance is at most 1
- the higher-confidence result is kept

This protects the output from repeated variants such as:

- `MH12AB1234`
- `MH12A81234`
- `MH12AB1239`

when the OCR difference is only a minor character-level error.

### Step 9: Rendering output
Each processed frame is written into an output video with:

- bounding boxes
- recognized plate text
- recent extracted plates panel
- FPS display

### Step 10: Saving run summary
After batch completion, the system writes a `summary.csv` file inside the run directory with:

- input video path
- output video path
- processed frame count
- average FPS
- deduplicated plate results

### Step 11: Auto-play
After each video finishes rendering, the result video is opened automatically using `os.startfile(...)`.

## 8. Duplicate Removal Logic

Duplicate suppression is one of the most important parts of an ANPR system because OCR can produce repeated or slightly noisy readings of the same plate.

This project uses fuzzy duplicate removal with `python-Levenshtein`.

Current behavior:

- maintain a global `final_plates` list
- compare every new candidate plate against existing saved plates
- if two strings have equal length and edit distance less than or equal to 1, they are treated as the same plate
- retain the higher-confidence version

This gives better final outputs than strict exact-string matching.

## 9. OCR Optimization Strategy

OCR is the most expensive stage after detection. To keep the pipeline smooth, the project uses several controls:

- per-track OCR throttling
- small-crop rejection
- image preprocessing before OCR
- track-based history instead of stateless OCR
- confidence filtering and voting

These choices reduce unnecessary OCR calls and improve practical performance.

## 10. GPU Usage

The project is designed to run with GPU acceleration when CUDA is available.

GPU-related behavior:

- YOLO model is loaded onto CUDA when available
- detector warmup is performed
- PaddleOCR GPU package is included in requirements
- GPU availability is logged at startup
- VRAM usage utilities are available in [gpu.py](c:/Users/VaishuNikhil/anpr_system/src/utils/gpu.py)

If GPU is not available, the project can still run on CPU, but this is not recommended for real-time or large-batch usage.

## 11. Configuration

Main configuration is stored in [config.yaml](c:/Users/VaishuNikhil/anpr_system/config/config.yaml).

Key configuration groups:

### `camera`
- default source path
- width and height
- FPS
- buffer size

### `detection`
- model path
- confidence threshold
- IoU threshold
- image size

### `tracking`
- tracker configuration path
- track buffer
- match threshold

### `ocr`
- OCR language
- confidence threshold
- regex pattern
- stability frame count

### `database`
- deduplication timeout
- image save path placeholder

### `paths`
- input directory
- output directory

### `output`
- output root folder

### `dataset`
- enable or disable run-time dataset saving

### `logging`
- log level
- log file path
- rotation policy

## 12. Setup Instructions

### Requirements

- Python environment
- CUDA-capable GPU recommended
- installed dependencies from `requirements.txt`

### Install

```bash
pip install -r requirements.txt
```

### Required model file

Ensure the following file exists:

```text
models/license_plate_detector.pt
```

## 13. How to Run

### Batch mode

Put all videos inside `input/` and run:

```bash
python run_all.py
```

### Single video render mode

You can also process a single video directly with:

```bash
python render_video.py --source input/your_video.mp4 --output output/output_1/your_video.mp4
```

### Live/helper mode

The file `run_anpr.py` is available for live/display-oriented usage and shared helper functions.

## 14. Output Format

For each batch run, the system creates a new folder like:

```text
output/
└── output_3/
    ├── video1.mp4
    ├── video2.mp4
    ├── summary.csv
    └── dataset/
        ├── events.csv
        └── ocr_training/
            ├── labels.csv
            └── images/
```

### Output files explained

#### Rendered videos
Each output video contains:

- detected plate bounding boxes
- recognized plate labels
- recent extracted plates panel
- FPS overlay

#### `summary.csv`
Contains per-video batch results.

#### `dataset/events.csv`
Stores saved event-level plate information for the run.

#### `dataset/ocr_training/labels.csv`
Stores image-to-text mapping for saved plate crops.

#### `dataset/ocr_training/images/`
Stores cropped plate images associated with final plate reads.

## 15. Logging and Monitoring

The project uses file and console logging through Loguru.

What is logged:

- GPU availability
- detector initialization
- OCR initialization
- OCR reads
- saved events
- OCR call metrics
- warnings and runtime issues

Default log file:

```text
logs/anpr_system.log
```

## 16. Safety and Cleanup

To reduce project clutter without destroying anything needed for execution, old generated outputs and cache files were moved into:

```text
backup_cleanup/
```

This keeps the project cleaner while preserving recoverable artifacts.

Items safely moved there include:

- old output videos
- previous log files
- Python cache folders

## 17. What Was Improved in This Final Version

- cleaned the project structure safely
- added `backup_cleanup/`
- created one-command batch execution with `run_all.py`
- upgraded processing from one plate to all tracked plates
- added robust fuzzy duplicate suppression
- added OCR frequency control per track
- added structured numbered output folders
- added automatic result playback
- added batch summary generation
- updated the documentation for final submission

## 18. Limitations

- OCR quality still depends on plate visibility, motion blur, angle, and lighting
- if the detector misses a plate, OCR cannot recover it
- very small or heavily blurred crops are intentionally skipped
- CPU-only execution will be slower

## 19. Future Improvements

- export JSON reports in addition to CSV
- add REST API or web dashboard
- support multiple camera streams in parallel
- improve country-specific plate formatting rules
- add database integration for production deployments
- add evaluation metrics for precision, recall, and OCR accuracy

## 20. Final Run Command

```bash
python run_all.py
```
