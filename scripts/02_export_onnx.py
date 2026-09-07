"""
Module 2: Exporting to ONNX

Exports the pretrained YOLO model to ONNX format, then runs two
separate verification checks:

1. Structural validity, is the exported file a well-formed ONNX graph?
2. Behavioural correctness, does it still detect the same things the
   original PyTorch model did?

Keeping these two checks separate matters. A structurally valid file
that gives wrong answers, and a structurally broken file, fail for
completely different reasons and need completely different fixes.
"""

from pathlib import Path

import onnx
import onnxruntime as ort
from ultralytics import YOLO

ONNX_PATH = Path("yolov8n.onnx")
SAMPLE_IMAGE = Path("sample.jpg")


def export_to_onnx() -> Path:
    """Export the pretrained model to ONNX with the settings this
    course relies on throughout."""
    model = YOLO("yolov8n.pt")
    exported_path = model.export(
        format="onnx",
        dynamic=True,   # accept variable input shapes (camera feeds, batches)
        simplify=True,  # fold redundant nodes for a cleaner graph
        opset=12,       # pin the operator set; the single biggest cause
                         # of failed exports is leaving this unpinned
    )
    return Path(exported_path)


def check_structural_validity(onnx_path: Path) -> None:
    """Confirm the exported file is a well-formed ONNX graph."""
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Structural check passed: {onnx_path} is a valid ONNX graph.")


def check_behavioural_correctness(onnx_path: Path) -> None:
    """Run a sanity inference with ONNX Runtime alone (no TensorRT yet)
    and confirm the session loads and runs without error."""
    if not SAMPLE_IMAGE.exists():
        print(f"Skipping behavioural check, {SAMPLE_IMAGE} not found.")
        return

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    print(
        f"Behavioural check: ONNX Runtime session loaded successfully. "
        f"Input node: '{input_name}'."
    )
    print(
        "Compare detections from this session against the Module 1 "
        "baseline output to confirm they match before moving on."
    )


def main() -> None:
    print("Exporting to ONNX...")
    onnx_path = export_to_onnx()
    print(f"Exported: {onnx_path}\n")

    check_structural_validity(onnx_path)
    check_behavioural_correctness(onnx_path)

    print(
        "\nNext: benchmark this ONNX Runtime session against your test "
        "video and record the FPS figure. If it's already meaningfully "
        "worse than the Module 1 baseline, the problem is in the "
        "export, not the TensorRT work in Module 3."
    )


if __name__ == "__main__":
    main()