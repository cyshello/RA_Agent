#!/bin/bash

# AX 가상환경 활성화 및 main.py 실행
cd /home/intern/youngseo/RA_Agent
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent
python3 main.py
