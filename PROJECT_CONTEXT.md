PROJECT CONTEXT: read fully before writing any code.

## What we are building
A university AI course project (AI2002, Path A: Software-Based). It is a
Streamlit app whose core is a trained YOLOv8 object detection model.

Title: Hardhat and Safety-Vest Compliance Detection for Construction Workers Using YOLOv8

Problem: Safety officers cannot watch every worker on a construction site.
Workers often skip hardhats and safety vests. Our system takes a site image,
detects each worker and their gear, decides each worker's compliance status,
and reports a site compliance rate.

## Hard requirements (project rules)
- The ML model (YOLOv8) must be the core. Rule-based logic only sits on top of it.
- Streamlit app is mandatory: input -> model run -> understandable results.
- Code must be well organized, commented, with meaningful names.
- This is a graded group project and I am the main developer. I must be able to explain every line in the viva, so keep the code simple and readable, and explain what you write.
- Do not copy existing projects or tutorials. Write our own implementation.

## Dataset
Construction Site Safety (v28 export), Roboflow Universe, CC BY 4.0.
2,801 images at 640x640 (train 2605 augmented / valid 114 / test ~82).
YOLO-format labels. 10 class IDs:
0 Hardhat, 1 Mask, 2 NO-Hardhat, 3 NO-Mask, 4 NO-Safety Vest,
5 Person, 6 Safety Cone, 7 Safety Vest, 8 machinery, 9 vehicle
The model is trained on all 10 classes. Our compliance logic uses only:
Person (5), Hardhat (0), NO-Hardhat (2), Safety Vest (7), NO-Safety Vest (4).
Other classes are ignored in the logic (they may be drawn faintly as context).

## Model and training (done OUTSIDE this IDE)
- Training happens in a Kaggle GPU notebook with YOLOv8n (ultralytics).
- We download best.pt and place it at models/best.pt.
- My laptop has NO GPU. Everything in this repo must run on CPU.
- Do not write training code here unless I ask. Until best.pt exists, use the
  pretrained yolov8n.pt only to test the app flow (it will not know our classes).

## Compliance logic (src/compliance.py)
Input: list of detections, each with class name, confidence, and box (x1,y1,x2,y2).
For each Person box:
1. Head zone = top ~35% of the person box. Torso zone = ~20% to 65% of the person box height.
2. Hardhat check: Hardhat box overlapping the head zone -> has hardhat.
   NO-Hardhat box overlapping the head zone -> missing hardhat.
3. Vest check: Safety Vest box overlapping the torso zone -> has vest.
   NO-Safety Vest box overlapping the torso zone -> missing vest.
4. Assign each gear box to the person with the greatest overlap, so one box
   is never given to two people.
5. If no evidence either way, mark that item "unknown" (do not guess compliant).
6. Worker status: Compliant, Missing hardhat, Missing vest, Missing both, or Unknown.
7. Site summary: total workers, compliant count, violation count, compliance rate %,
   site status (Safe / Warning / Critical). Thresholds must be easy to change (constants).
The overlap rule (e.g. IoU or intersection over gear-box area) and thresholds
should be named constants at the top of the file with comments.

## Streamlit app (app.py)
- Sidebar: confidence threshold slider; choice of required gear
  (hardhat only, or hardhat + vest).
- Upload an image (jpg/png), click Detect.
- Show the annotated image: green boxes for compliant workers, red for
  violations, gear boxes drawn too.
- Show summary metrics (workers, compliant, violations, compliance rate, site status).
- Show a per-worker table and a suggested action message.
- Handle edge cases: no detections, wrong file type, missing model file.
- Load the model once (cache it) so the app stays fast on CPU.
Images only, no video.

## Planned folder structure
ppe-compliance/
  src/detector.py     (load model, run inference, return detections)
  src/compliance.py   (logic above, pure Python, no Streamlit imports)
  src/utils.py        (drawing and summary helpers)
  app.py
  models/best.pt
  tests/test_compliance.py   (pytest, made-up boxes for every rule and edge case)
  eval/               (metrics and system-level test results)
  requirements.txt
  README.md

## How I want you to work
- Work in small steps. One file or one function at a time, then wait for me.
- Before writing code, tell me the plan in 3-5 lines.
- After writing code, explain it simply, since I must explain it in a viva.
- Keep compliance.py independent from Streamlit and YOLO so it can be unit tested.
- Write pytest tests for compliance.py and run them. Don't claim something works
  unless you ran it.
- Python 3.10/3.11, CPU only, minimal dependencies
  (ultralytics, streamlit, opencv-python, pillow, pandas, pytest).
- Ask me if something is unclear instead of guessing.

## First task
Create the folder structure, requirements.txt, and an empty skeleton of each
file with docstrings only. Then stop and wait for my next instruction.