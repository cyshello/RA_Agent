#!/bin/bash
# -*- coding: utf-8 -*-
#
# B2G 데이터 파이프라인 예시 스크립트
# 
# 이 스크립트는 PDF 로드 → DB 확인 → 검색 테스트를 순차적으로 실행합니다.
#
# 사용법:
#   chmod +x scripts/run_pipeline_example.sh
#   ./scripts/run_pipeline_example.sh
#

set -e  # 오류 발생시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 가상환경 활성화
cd "$PROJECT_DIR"
source ../.venv/bin/activate

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  B2G 데이터 파이프라인 예시 스크립트${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================================
# 설정 (필요에 따라 수정하세요)
# ============================================================================

# PDF 파일 경로 (예시 - 실제 파일 경로로 변경하세요)
PROJECT_PDF="/Users/youngseocho/Desktop/AX/RA_Agent/data/criteria/presidential_agenda.pdf"
MANAGEMENT_PDF="/Users/youngseocho/Desktop/AX/RA_Agent/data/criteria/management_eval.pdf"
INCLUSIVE_PDF="/Users/youngseocho/Desktop/AX/RA_Agent/data/criteria/inclusive_growth.pdf"

# 페이지 범위 (예시 - 실제 문서에 맞게 수정하세요)
PROJECT_INDEX_PAGES="13-17"
PROJECT_DETAIL_PAGES="21-195"

MANAGEMENT_INDEX_PAGES="20-25"
MANAGEMENT_DETAIL_PAGES="27-46"

INCLUSIVE_INDEX_PAGES="2-2"
INCLUSIVE_DETAIL_PAGES="3-17"

# 출력 디렉토리
OUTPUT_DIR="./output_pipeline"

# OCR 캐시 디렉토리 (OCR 결과 재사용시 사용)
OCR_CACHE_DIR="./ocr_cache"

# OCR 재사용 여부 (true/false)
# true로 설정하면 기존 OCR 결과가 있으면 재사용
REUSE_OCR=false

# ============================================================================
# 함수 정의
# ============================================================================

print_step() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  STEP $1: $2${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================================
# STEP 0: 현재 DB 상태 확인
# ============================================================================

print_step "0" "현재 DB 상태 확인"

python scripts/check_db.py

# ============================================================================
# STEP 1: PDF 파일 처리 (선택적)
# ============================================================================

print_step "1" "PDF 파일 처리 (선택적)"

# OCR 재사용 옵션 설정
OCR_OPT=""
if [ "$REUSE_OCR" = "true" ]; then
    print_info "OCR 재사용 모드 활성화 (기존 OCR 결과 사용)"
fi

# 국정과제 PDF 처리
if [ -f "$PROJECT_PDF" ]; then
    print_info "국정과제 PDF 처리 중: $PROJECT_PDF"
    
    OCR_OPT=""
    if [ "$REUSE_OCR" = "true" ] && [ -d "$OCR_CACHE_DIR/project" ]; then
        OCR_OPT="--reuse-ocr $OCR_CACHE_DIR/project"
        print_info "  → OCR 캐시 사용: $OCR_CACHE_DIR/project"
    fi
    
    python scripts/load_pdf_to_db.py "$PROJECT_PDF" project \
        --index-pages "$PROJECT_INDEX_PAGES" \
        --detail-pages "$PROJECT_DETAIL_PAGES" \
        --output-dir "$OUTPUT_DIR/project" \
        $OCR_OPT
    print_success "국정과제 PDF 처리 완료"
else
    print_info "국정과제 PDF 파일 없음 (스킵): $PROJECT_PDF"
fi

# 경영평가 PDF 처리
if [ -f "$MANAGEMENT_PDF" ]; then
    print_info "경영평가 PDF 처리 중: $MANAGEMENT_PDF"
    
    OCR_OPT=""
    if [ "$REUSE_OCR" = "true" ] && [ -d "$OCR_CACHE_DIR/management_eval" ]; then
        OCR_OPT="--reuse-ocr $OCR_CACHE_DIR/management_eval"
        print_info "  → OCR 캐시 사용: $OCR_CACHE_DIR/management_eval"
    fi
    
    python scripts/load_pdf_to_db.py "$MANAGEMENT_PDF" management_eval \
        --index-pages "$MANAGEMENT_INDEX_PAGES" \
        --detail-pages "$MANAGEMENT_DETAIL_PAGES" \
        --output-dir "$OUTPUT_DIR/management_eval" \
        $OCR_OPT
    print_success "경영평가 PDF 처리 완료"
else
    print_info "경영평가 PDF 파일 없음 (스킵): $MANAGEMENT_PDF"
fi

# 동반성장 PDF 처리
if [ -f "$INCLUSIVE_PDF" ]; then
    print_info "동반성장 PDF 처리 중: $INCLUSIVE_PDF"
    
    OCR_OPT=""
    if [ "$REUSE_OCR" = "true" ] && [ -d "$OCR_CACHE_DIR/inclusive_growth" ]; then
        OCR_OPT="--reuse-ocr $OCR_CACHE_DIR/inclusive_growth"
        print_info "  → OCR 캐시 사용: $OCR_CACHE_DIR/inclusive_growth"
    fi
    
    python scripts/load_pdf_to_db.py "$INCLUSIVE_PDF" inclusive_growth \
        --index-pages "$INCLUSIVE_INDEX_PAGES" \
        --detail-pages "$INCLUSIVE_DETAIL_PAGES" \
        --output-dir "$OUTPUT_DIR/inclusive_growth" \
        $OCR_OPT
    print_success "동반성장 PDF 처리 완료"
else
    print_info "동반성장 PDF 파일 없음 (스킵): $INCLUSIVE_PDF"
fi

# ============================================================================
# STEP 2: 처리 후 DB 상태 확인
# ============================================================================

print_step "2" "처리 후 DB 상태 확인"

python scripts/check_db.py

# ============================================================================
# STEP 3: 검색 테스트
# ============================================================================

print_step "3" "검색 테스트"

# 테스트 검색어 목록
SEARCH_QUERIES=(
    "인공지능 AI"
    "디지털 전환"
    "탄소중립"
    "상생 협력"
)

for query in "${SEARCH_QUERIES[@]}"; do
    echo ""
    echo -e "${YELLOW}▶ 검색어: \"$query\"${NC}"
    echo "------------------------------------------------------------"
    python scripts/search_test.py "$query" --all --limit 5
done

# ============================================================================
# 완료
# ============================================================================

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  파이프라인 실행 완료!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "다음 명령어로 개별 작업을 수행할 수 있습니다:"
echo ""
echo "  # DB 확인"
echo "  python scripts/check_db.py"
echo "  python scripts/check_db.py project --limit 10"
echo ""
echo "  # 검색"
echo "  python scripts/search_test.py \"검색어\" --all"
echo "  python scripts/search_test.py \"검색어\" -t project"
echo ""
echo "  # OCR만 수행 (결과 캐싱)"
echo "  python scripts/run_ocr_only.py <pdf경로> --pages 1-100 --output-dir ./ocr_cache/<타입>"
echo ""
echo "  # PDF 처리 (OCR 재사용)"
echo "  python scripts/load_pdf_to_db.py <pdf경로> <타입> --index-pages 1-3 --detail-pages 4-50 --reuse-ocr ./ocr_cache/<타입>"
echo ""
echo "  # OCR 재사용 모드로 전체 파이프라인 실행"
echo "  REUSE_OCR=true ./scripts/run_pipeline_example.sh"
echo ""
