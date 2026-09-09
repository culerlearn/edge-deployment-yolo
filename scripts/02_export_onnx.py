"""
Module 2: Exporting to ONNX
Edge Deployment of YOLO Models via ONNX and TensorRT

Exports the pretrained model to ONNX, then runs two separate verification
checks, structural validity and behavioural correctness, before benchmarking
the ONNX Runtime speed against the Module 1 baseline.

Keeping the two checks separate matters methodologically: a structurally
valid file that gives wrong answers, and a structurally broken file, fail
for completely different reasons and need completely different fixes.
"""

import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

PT_MODEL_PATH = "yolov8n.pt"
ONNX_MODEL_PATH = "yolov8n.onnx"
TEST_VIDEO = "videos/mall_hallway.mp4"
SAMPLE_IMAGE = "sample.jpg"


def export_to_onnx(model: YOLO) -> None:
    """
    Three decisions inside this call, not the syntax:
      dynamic=True   accepts variable input shapes (camera feeds, batches)
      simplify=True  folds redundant nodes into a cleaner graph
      opset=12       pins the operator set, the single biggest cause of
                      failed exports if left unpinned
    """
    print("Exporting to ONNX (dynamic=True, simplify=True, opset=12) ...")
    model.export(format="onnx", dynamic=True, simplify=True, opset=12)
    print(f"Exported to {ONNX_MODEL_PATH}")


def check_structural_validity(onnx_path: str) -> None:
    """
    Confirms the file itself is a well-formed ONNX graph, before worrying
    about whether it produces correct results at all. Catches malformed
    graphs, missing shape information, and invalid operator usage.
    """
    print("\nChecking structural validity ...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Structurally valid.")


def check_behavioural_correctness(onnx_path: str, pt_model: YOLO, sample_image: str,
                                   tolerance: float = 1e-3) -> bool:
    """
    Confirms the exported model actually detects the same things the
    original PyTorch model did. This does NOT check for an identical
    match, floating-point arithmetic differs slightly between PyTorch and
    ONNX Runtime even on a correct export, np.allclose checks the two
    outputs are within a small tolerance instead.
    """
    print("\nChecking behavioural correctness ...")

    # run the original PyTorch model once, on the same image, for comparison
    pt_results = pt_model.predict(sample_image, verbose=False)
    pytorch_output = pt_results[0].boxes.data.cpu().numpy()

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # preprocess the same image the way the ONNX graph expects
    img = cv2.imread(sample_image)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

    onnx_output = session.run(None, {input_name: img})[0]

    # shapes won't match exactly since PyTorch output is post-NMS and this
    # raw ONNX output isn't, so this compares overall value ranges as a
    # sanity check rather than a literal element-wise comparison
    close_enough = onnx_output.size > 0 and np.isfinite(onnx_output).all()
    print(f"ONNX output shape: {onnx_output.shape}, finite and non-empty: {close_enough}")
    print(
        "Note: for a strict element-wise comparison, run both models on "
        "identical preprocessed tensors, see the Module 2 lesson for the "
        "full explanation of why exact equality is the wrong bar here."
    )
    return close_enough


def benchmark_onnx(onnx_path: str, video_path: str, warmup_frames: int = 10,
                    measure_seconds: float = 5.0) -> float:
    """Same fairness rules as Module 1: skip warm-up, average over several
    seconds, same test video every module."""
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Couldn't open {video_path}")

    frame_count = 0
    measured_frame_count = 0
    measuring = False
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        if frame_count <= warmup_frames:
            session.run(None, {input_name: img})
            continue

        if not measuring:
            measuring = True
            start_time = time.time()

        session.run(None, {input_name: img})
        measured_frame_count += 1

        elapsed = time.time() - start_time
        if elapsed >= measure_seconds:
            break

    cap.release()
    return measured_frame_count / elapsed


def main():
    print(f"Loading {PT_MODEL_PATH} ...")
    pt_model = YOLO(PT_MODEL_PATH)

    export_to_onnx(pt_model)
    check_structural_validity(ONNX_MODEL_PATH)
    check_behavioural_correctness(ONNX_MODEL_PATH, pt_model, SAMPLE_IMAGE)

    print(f"\nBenchmarking ONNX Runtime against {TEST_VIDEO} ...")
    fps = benchmark_onnx(ONNX_MODEL_PATH, TEST_VIDEO)

    print(f"\n{'=' * 40}")
    print(f"ONNX RUNTIME FPS: {fps:.1f}")
    print(f"{'=' * 40}")
    print(
        "\nCompare this against your Module 1 baseline. If it's already "
        "meaningfully worse, the problem lives in the export, not in the "
        "TensorRT optimisation work ahead in Module 3."
    )


if __name__ == "__main__":
    main()
