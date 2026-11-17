# Hardhat and Safety-Vest Compliance Detection for Construction Workers Using YOLOv8

AI2002 Course Project (Path A: Software-Based)

## Overview
A Streamlit web application that uses a trained YOLOv8 model and spatial rule-based compliance logic to detect construction workers, identify whether they are wearing hardhats and safety vests, and summarize site-wide safety compliance.

## Project Structure
```
.
├── PROJECT_CONTEXT.md       # Project specifications, rules, and guidelines
├── requirements.txt         # Python project dependencies
├── README.md                # Project documentation
├── app.py                   # Streamlit user interface
├── src/
│   ├── detector.py          # Model loading and inference wrapper
│   ├── compliance.py        # Pure Python rule-based compliance logic
│   └── utils.py             # Image annotation and visualization utilities
├── models/                  # Exported model weights (e.g. best.pt)
├── tests/
│   └── test_compliance.py  # Isolated pytest suite for compliance rules
└── eval/                    # Evaluation logs, test results, and metrics
```

## Setup & Running
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
