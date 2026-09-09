"""
Module 4: Building the Runtime and Live Test
Edge Deployment of YOLO Models via ONNX and TensorRT

Wires the optimised TensorRT engine into an actual video pipeline and
measures the full loop, read, inference, and display together. This is
deliberately the honest number, what a user actually experiences watching
the screen, not just the engine's raw inference speed in isolation.

Requires yolov8n.engine to already exist, built in Module 3:
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16

If you're on Intel hardware instead of NVIDIA, see the Module 3 lesson for
the OpenVINO equivalent, the loading line below is the only change needed.
"""

import time
import cv2
from ultralytics import YOLO

ENGINE_PATH = "yolov8n.engine"
TEST_VIDEO = "videos/mall_hallway.mp4"
# to use a live camera instead of the test video, change this to an integer
# camera index instead, e.g. VIDEO_SOURCE = 0


def run_pipeline(engine_path: str, video_path: str) -> None:
    print(f"Loading compiled engine: {engine_path} ...")
    # loading the .engine file directly as the model is what points
    # everything at the compiled, hardware-specific version, from here on
    # model.predict() behaves exactly like Module 1, just running faster
    model = YOLO(engine_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Couldn't open {video_path}. To use a live camera instead, "
            f"replace this with an integer camera index, e.g. cv2.VideoCapture(0)."
        )

    prev_time = time.time()

    print("Running live, press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached, looping.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # this is the full loop being measured, read, infer, and display
        # together, deliberately the honest number
        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        cv2.imshow("Live Detection", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    run_pipeline(ENGINE_PATH, TEST_VIDEO)
    print(
        "\nThat's the full pipeline, baseline to live detection. Compare "
        "the FPS you just saw against your Module 1 baseline, same model, "
        "a fraction of the latency.\n"
        "\nIf this number reads noticeably lower than Module 3's trtexec "
        "benchmark, that's expected, this measures the full loop including "
        "read and display overhead, not the engine in isolation. See the "
        "Module 4 lesson's Common Pitfalls section if the gap looks larger "
        "than expected.\n"
        "\nSee the Final Hands-On Exercise in the Module 4 lesson to "
        "complete the course."
    )


if __name__ == "__main__":
    main()
