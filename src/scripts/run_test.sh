#!/bin/bash

# 문서 분석 파이프라인 실행 스크립트
# 
# 사용법:
#   bash src/scripts/run_example.sh

# 스크립트 디렉토리에서 프로젝트 루트로 이동
cd "$(dirname "$0")/../.." || exit

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate AX

# 파라미터 설정
COMPANY_NAME="Test_Company"
DOCUMENTS="patent1:data/test/patent1.pdf patent2:data/test/patent2.pdf intro:data/test/intro.pdf plan:data/test/business_plan.pdf"  # 여러 문서: "doc1:path1 doc2:path2"
EXTRACT_MODEL="gemini"                           # 문서 추출용: openai 또는 gemini
EXTRACT_MODEL_NAME="gemini-2.5-flash" #"gemini-3-pro-preview"        # 비전모델필요
REPORT_MODEL="gemini" #"openai"                            # 보고서 생성용: openai 또는 gemini
REPORT_MODEL_NAME="gemini-3-pro-preview" #"gpt-4o"                       # 사용 가능: gpt-4o, gpt-4o-mini, o1-preview, o1-mini
WEB_SEARCH_FLAG="--web"                               # 웹 검색 활성화: "--web", 비활성화: ""
MAX_RPS="2.0"
DEBUG_FLAG="--debug"                             # 디버그 모드: "--debug", 비활성화: ""

# Python 스크립트 실행
python3 main.py \
    --company "$COMPANY_NAME" \
    --documents $DOCUMENTS \
    --extract-model "$EXTRACT_MODEL" \
    --extract-model-name "$EXTRACT_MODEL_NAME" \
    --report-model "$REPORT_MODEL" \
    --report-model-name "$REPORT_MODEL_NAME" \
    --max-rps "$MAX_RPS" \
    $WEB_SEARCH_FLAG \
    $DEBUG_FLAG
