"""
Module 1: Setup and Getting a Pretrained Model
Edge Deployment of YOLO Models via ONNX and TensorRT

Loads a standard pretrained YOLO model, confirms it works on a still image,
then establishes an honest baseline FPS against the course test video.

This baseline is the single most important number in the whole course,
every later module's improvement is measured against it. Run this once,
here, before anything is exported or optimised.
"""

import time
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"          # downloads automatically on first run
TEST_VIDEO = "videos/mall_hallway.mp4"
SAMPLE_IMAGE = "sample.jpg"        # swap in your own image to sanity-check detection


def confirm_model_works(model: YOLO) -> None:
    """Run once on a still image so you can see, with your own eyes, that
    detection is genuinely working before moving on to the video benchmark."""
    print(f"Running a sanity check against {SAMPLE_IMAGE} ...")
    results = model(SAMPLE_IMAGE)
    results[0].show()


def establish_baseline(model: YOLO, video_path: str, warmup_frames: int = 10,
                        measure_seconds: float = 5.0) -> float:
    """
    Benchmarks the model against the course test video, following the same
    three fairness rules taught in Module 1:

    1. Let the first few frames run and settle before reading anything,
       the very first frame or two is often slower while everything warms up.
    2. Read an average over several seconds, not a single instantaneous
       number, a single frame's FPS is noisy.
    3. Use the exact same test video every module from here on, so any
       change in the number later is caused by what you did, not by the
       footage changing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Couldn't open {video_path}. Confirm the videos/ folder is in "
            f"your working directory, or update TEST_VIDEO above."
        )

    frame_count = 0
    measured_frame_count = 0
    measuring = False
    start_time = None

    print(f"Warming up ({warmup_frames} frames) ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            # loop the video if it's shorter than measure_seconds
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # rule 1: skip the warm-up frames entirely
        if frame_count <= warmup_frames:
            model.predict(frame, verbose=False)
            continue

        if not measuring:
            print("Warm-up complete, measuring ...")
            measuring = True
            start_time = time.time()

        model.predict(frame, verbose=False)
        measured_frame_count += 1

        # rule 2: average over several seconds, not a single frame
        elapsed = time.time() - start_time
        if elapsed >= measure_seconds:
            break

    cap.release()

    fps = measured_frame_count / elapsed
    return fps


def main():
    print(f"Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    confirm_model_works(model)

    print(f"\nEstablishing baseline against {TEST_VIDEO} ...")
    fps = establish_baseline(model, TEST_VIDEO)

    print(f"\n{'=' * 40}")
    print(f"BASELINE FPS: {fps:.1f}")
    print(f"{'=' * 40}")
    print(
        "\nRecord this number, every later module's improvement is measured "
        "against it. Next up: exporting to ONNX (Module 2)."
    )


if __name__ == "__main__":
    main()
