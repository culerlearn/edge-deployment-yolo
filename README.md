# Edge Deployment of YOLO Models via ONNX and TensorRT

Companion code for the [CulerLearn](https://www.culerlearn.com) course of the same name. This repository contains the working scripts referenced throughout all four modules, take a standard, pretrained YOLO model from PyTorch through to a real-time detection pipeline running at 60+ FPS.

This is not a training repository. Every script here assumes you already have a pretrained YOLO model, and focuses entirely on export, optimisation, and deployment speed.

## Course Structure

| Module | Topic | Script |
|---|---|---|
| 1 | Setup and Getting a Pretrained Model | `scripts/01_baseline.py` |
| 2 | Exporting to ONNX | `scripts/02_export_onnx.py` |
| 3 | Optimising with TensorRT | `trtexec` (run directly, see Quick Start) |
| 3 (alternative) | Optimising with OpenVINO | `ovc` (run directly, see Quick Start) |
| 4 | Building the Runtime and Live Test | `scripts/04_runtime.py` |

Full lesson content, explanations, and the troubleshooting guide live on [CulerLearn](https://www.culerlearn.com), this repo is the code, not the course itself.

## Prerequisites

- Python fundamentals, basic familiarity with PyTorch and YOLO
- Visual Studio Code (this course is built around it, integrated terminal, Python extension)
- An NVIDIA GPU (for the TensorRT path) or Intel-based hardware (for the OpenVINO path)
- A CUDA and cuDNN version compatible with your installed TensorRT release, if using the NVIDIA path

## Setup

```bash
git clone https://github.com/culerlearn/edge-deployment-yolo.git
cd edge-deployment-yolo
python -m venv yolo-deploy
```

Activate the environment (this is the one step that differs by operating system):

```bash
# Windows (PowerShell, VS Code's default integrated terminal)
.\yolo-deploy\Scripts\Activate.ps1

# Windows (Command Prompt)
yolo-deploy\Scripts\activate.bat

# macOS / Linux
source yolo-deploy/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

**Windows note:** if PowerShell blocks the activation script with an execution policy error, this is a common first-run issue, not a broken install. Run this once, then try activating again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Repository Structure

```
edge-deployment-yolo/
├── scripts/
│   ├── 01_baseline.py
│   ├── 02_export_onnx.py
│   └── 04_runtime.py
├── videos/
│   └── mall-hallway.mp4
├── requirements.txt
├── LICENSE
└── README.md
```

**Note on Module 3:** building the TensorRT or OpenVINO engine is a single command each, run directly rather than wrapped in a shell script, since `trtexec` and `ovc` are standalone programs that work identically in PowerShell, Command Prompt, or a macOS/Linux terminal once installed. See the Quick Start section below.

## Test Footage

`videos/mall-hallway.mp4` is a generated video of pedestrians walking through a mall hallway, used throughout Module 4 as the live detection test source. Using a live camera instead is a one-line change, covered in the Module 4 lesson.

## Quick Start

```bash
# Module 1: confirm your pretrained model and record a baseline
python scripts/01_baseline.py

# Module 2: export to ONNX
python scripts/02_export_onnx.py

# Module 3: build an optimised engine (pick the path matching your hardware)
# Run directly, same command on Windows, macOS, or Linux
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16   # NVIDIA
ovc yolov8n.onnx --output_model yolov8n_openvino                 # Intel

# Module 4: run the full pipeline against the test video
python scripts/04_runtime.py
```

## Troubleshooting

Common errors and their fixes are covered per module in the course itself, and in the downloadable troubleshooting reference guide available with course access.

## Licence

MIT, see [LICENSE](LICENSE). Use, adapt, and reuse this code freely, including outside the course.