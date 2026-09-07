"""
Module 1: Setup and Getting a Pretrained Model

Loads a standard, pretrained YOLO model, confirms it works on a still
image, then benchmarks it against the test video to establish the
baseline frames-per-second figure. Every later module compares its
result against the number this script prints.

No training happens here or anywhere in this course. This script
assumes an already-trained model.
"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = Path("videos/mall-hallway.mp4")
SAMPLE_IMAGE = Path("sample.jpg")


def confirm_model_works(model: YOLO) -> None:
    """Run a single still-image inference and display the result."""
    if not SAMPLE_IMAGE.exists():
        print(f"Skipping still-image check, {SAMPLE_IMAGE} not found.")
        return
    results = model(str(SAMPLE_IMAGE))
    results[0].show()


def run_baseline(model: YOLO) -> None:
    """Run the model against the test video and print the stabilised FPS."""
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"{VIDEO_PATH} not found. Place the test video in the "
            f"videos/ folder, or point VIDEO_PATH at your own file."
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video has ended

        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"Baseline FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Module 1: Baseline (PyTorch)", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    print("Loading pretrained model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")  # downloads automatically on first run

    confirm_model_works(model)

    print("Running baseline benchmark. Press 'q' to stop early.")
    run_baseline(model)

    print(
        "\nDone. Note the FPS figure shown on screen, this is your "
        "Module 1 baseline. Every later module's improvement is "
        "measured against this number."
    )


if __name__ == "__main__":
    main()