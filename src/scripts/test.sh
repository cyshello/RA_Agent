#!/bin/bash

# AX 가상환경 활성화 및 main.py 실행
cd /Users/youngseocho/Desktop/AX/RA_Agent
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AX
python3 main.py
