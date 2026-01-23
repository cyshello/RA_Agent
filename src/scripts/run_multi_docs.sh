#!/bin/bash

# 여러 문서를 분석하는 예시 스크립트 (Gemini 모델 사용)
# 
# 사용법:
#   bash src/scripts/run_multi_docs.sh

cd "$(dirname "$0")/../.." || exit

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate AX

# 파라미터 설정
COMPANY_NAME="Multi Doc Example"
# 여러 문서를 공백으로 구분
DOCUMENTS="doc1:data/instruction1.pdf doc2:data/IR1.pdf"
EXTRACT_MODEL="gemini"                           # 문서 추출용
EXTRACT_MODEL_NAME="gemini-2.0-flash-exp"
REPORT_MODEL="openai"                            # 보고서 생성용
REPORT_MODEL_NAME="gpt-4o"
OCR_PROVIDER="Upstage"
MAX_RPS="3.0"

# 실행
python3 main.py \
    -c "$COMPANY_NAME" \
    -d $DOCUMENTS \
    -em "$EXTRACT_MODEL" \
    -emn "$EXTRACT_MODEL_NAME" \
    -rm "$REPORT_MODEL" \
    -rmn "$REPORT_MODEL_NAME" \
    --ocr "$OCR_PROVIDER" \
    --web \
    --max-rps "$MAX_RPS" \
    --debug
