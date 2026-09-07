"""
Module 4: Building the Runtime and Live Test

Wires the optimised TensorRT (or OpenVINO) engine built in Module 3
into an actual video pipeline, and measures the full loop: read,
infer, display. This is the honest, end-to-end number, not just the
model's raw inference speed in isolation.

Uses videos/mall_hallway.mp4 by default. To use a live camera instead,
change VIDEO_SOURCE to 0 (or another camera index).
"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_SOURCE = "videos/mall-hallway.mp4"  # set to 0 for a live camera
ENGINE_PATH = Path("yolov8n.engine")  # from Module 3 (TensorRT)
# If you built the OpenVINO path instead, point this at that model
# folder and swap the model.predict() call below accordingly.


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(
            f"{ENGINE_PATH} not found. Build it in Module 3 first "
            f"(trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16)."
        )

    model = YOLO(str(ENGINE_PATH))

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {VIDEO_SOURCE}. "
            f"Check the file path, or the camera index if using a live feed."
        )

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video file ended; live cameras never hit this

        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Module 4: Live Runtime", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(
        "\nDone. Compare the FPS shown here against your Module 3 "
        "engine benchmark. A meaningful gap usually points to "
        "read/display overhead rather than the model itself, see the "
        "troubleshooting guide for details."
    )


if __name__ == "__main__":
    main()