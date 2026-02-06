#!/bin/bash
# -*- coding: utf-8 -*-
#
# B2G 데이터베이스 구축 스크립트
# PDF → OCR → JSON 저장 → 임베딩 → MySQL DB 저장을 한번에 처리합니다.
#
# 사용법:
#   ./db_scripts/build_db.sh <pdf_path> <data_type> <index_pages> <detail_pages> [options]
#
# 필수 인자:
#   pdf_path      PDF 파일 경로
#   data_type     데이터 종류 (project | management_eval | inclusive_growth)
#   index_pages   목록 페이지 범위 (예: 13-17)
#   detail_pages  세부내용 페이지 범위 (예: 21-195)
#
# 선택 옵션:
#   --reuse-ocr <dir>    기존 OCR 결과 재사용 (OCR 단계 스킵)
#   --output-dir <dir>   중간 결과 저장 디렉토리
#   --db-host <host>     MySQL 호스트 (기본값: localhost)
#   --db-port <port>     MySQL 포트 (기본값: 3306)
#   --db-name <name>     데이터베이스 이름 (기본값: b2g_data)
#   --db-user <user>     MySQL 사용자 (기본값: root)
#   --db-password <pw>   MySQL 비밀번호 (기본값: 빈 문자열)
#
# 예시:
#   # 국정과제 PDF 처리
#   ./db_scripts/build_db.sh ./data/criteria/presidential_agenda.pdf project 13-17 21-195
#
#   # 경영평가 PDF 처리 (중간 결과 저장)
#   ./db_scripts/build_db.sh ./data/criteria/management_eval.pdf management_eval 20-25 27-46 --output-dir ./output/management
#
#   # 동반성장 PDF 처리 (기존 OCR 재사용)
#   ./db_scripts/build_db.sh ./data/criteria/inclusive_growth.pdf inclusive_growth 2-5 6-50 --reuse-ocr ./ocr_cache/inclusive
#

set -e  # 오류 발생시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 스크립트 디렉토리 및 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# 함수 정의
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  B2G 데이터베이스 구축 스크립트${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_usage() {
    echo -e "${YELLOW}사용법:${NC}"
    echo "  $0 <pdf_path> <data_type> <index_pages> <detail_pages> [options]"
    echo ""
    echo -e "${YELLOW}필수 인자:${NC}"
    echo "  pdf_path       PDF 파일 경로"
    echo "  data_type      데이터 종류:"
    echo "                   - project          : 국정과제"
    echo "                   - management_eval  : 경영평가 지표"
    echo "                   - inclusive_growth : 동반성장 평가지표"
    echo "  index_pages    목록 페이지 범위 (예: 13-17)"
    echo "  detail_pages   세부내용 페이지 범위 (예: 21-195)"
    echo ""
    echo -e "${YELLOW}선택 옵션:${NC}"
    echo "  --reuse-ocr <dir>    기존 OCR 결과 재사용"
    echo "  --output-dir <dir>   중간 결과 저장 디렉토리"
    echo "  --db-host <host>     MySQL 호스트 (기본값: localhost)"
    echo "  --db-port <port>     MySQL 포트 (기본값: 3306)"
    echo "  --db-name <name>     DB 이름 (기본값: b2g_data)"
    echo "  --db-user <user>     MySQL 사용자 (기본값: root)"
    echo "  --db-password <pw>   MySQL 비밀번호"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo "  # 국정과제 PDF 처리"
    echo "  $0 ./data/criteria/presidential_agenda.pdf project 13-17 21-195"
    echo ""
    echo "  # 경영평가 PDF 처리 (중간 결과 저장)"
    echo "  $0 ./data/criteria/management_eval.pdf management_eval 20-25 27-46 --output-dir ./output/management"
    echo ""
    echo "  # 동반성장 PDF 처리 (기존 OCR 재사용)"
    echo "  $0 ./data/criteria/inclusive_growth.pdf inclusive_growth 2-5 6-50 --reuse-ocr ./ocr_cache/inclusive"
    echo ""
}

print_step() {
    echo ""
    echo -e "${CYAN}▶ $1${NC}"
    echo -e "${CYAN}$(printf '%.0s─' {1..60})${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ 오류: $1${NC}"
}

print_info() {
    echo -e "  ${YELLOW}→${NC} $1"
}

# ============================================================================
# 인자 파싱
# ============================================================================

# 최소 인자 개수 확인
if [ $# -lt 4 ]; then
    print_header
    print_error "필수 인자가 부족합니다."
    echo ""
    print_usage
    exit 1
fi

# 필수 인자
PDF_PATH="$1"
DATA_TYPE="$2"
INDEX_PAGES="$3"
DETAIL_PAGES="$4"
shift 4

# 선택 옵션 기본값
REUSE_OCR=""
OUTPUT_DIR=""
DB_HOST="localhost"
DB_PORT="3306"
DB_NAME="b2g_data"
DB_USER="root"
DB_PASSWORD=""

# 선택 옵션 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --reuse-ocr)
            REUSE_OCR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --db-host)
            DB_HOST="$2"
            shift 2
            ;;
        --db-port)
            DB_PORT="$2"
            shift 2
            ;;
        --db-name)
            DB_NAME="$2"
            shift 2
            ;;
        --db-user)
            DB_USER="$2"
            shift 2
            ;;
        --db-password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        *)
            print_error "알 수 없는 옵션: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# 유효성 검사
# ============================================================================

print_header

# data_type 유효성 검사
case $DATA_TYPE in
    project|management_eval|inclusive_growth)
        ;;
    *)
        print_error "잘못된 data_type: $DATA_TYPE"
        echo "  허용 값: project, management_eval, inclusive_growth"
        exit 1
        ;;
esac

# PDF 파일 존재 확인
if [ ! -f "$PDF_PATH" ]; then
    print_error "PDF 파일을 찾을 수 없습니다: $PDF_PATH"
    exit 1
fi

# reuse-ocr 디렉토리 확인
if [ -n "$REUSE_OCR" ] && [ ! -d "$REUSE_OCR" ]; then
    print_error "OCR 캐시 디렉토리를 찾을 수 없습니다: $REUSE_OCR"
    exit 1
fi

# ============================================================================
# 설정 출력
# ============================================================================

# 데이터 타입 한글명
case $DATA_TYPE in
    project)
        TYPE_NAME="국정과제"
        TYPE_FIELDS="과제명, 과제_목표, 주요내용, 기대효과"
        ;;
    management_eval)
        TYPE_NAME="경영평가 지표"
        TYPE_FIELDS="지표명, 평가기준, 평가방법, 참고사항, 증빙자료"
        ;;
    inclusive_growth)
        TYPE_NAME="동반성장 평가지표"
        TYPE_FIELDS="지표명, 평가기준, 평가방법"
        ;;
esac

print_step "설정 확인"
print_info "PDF 파일: $PDF_PATH"
print_info "데이터 타입: $TYPE_NAME ($DATA_TYPE)"
print_info "추출 필드: $TYPE_FIELDS"
print_info "목록 페이지: $INDEX_PAGES"
print_info "세부 페이지: $DETAIL_PAGES"

if [ -n "$REUSE_OCR" ]; then
    print_info "OCR 재사용: $REUSE_OCR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    print_info "중간결과 저장: $OUTPUT_DIR"
fi

print_info "DB: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"

# ============================================================================
# 가상환경 활성화
# ============================================================================

print_step "가상환경 활성화"

cd "$PROJECT_DIR"

# 가상환경 경로 확인 (여러 가능한 위치)
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    print_success "가상환경 활성화: ../.venv"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    print_success "가상환경 활성화: .venv"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    print_success "가상환경 활성화: venv"
else
    print_error "가상환경을 찾을 수 없습니다."
    echo "  다음 중 하나의 경로에 가상환경이 있어야 합니다:"
    echo "    - ../.venv/bin/activate"
    echo "    - .venv/bin/activate"
    echo "    - venv/bin/activate"
    exit 1
fi

# ============================================================================
# DB 파이프라인 실행
# ============================================================================

print_step "DB 구축 파이프라인 실행"

# 명령어 구성
CMD="python db_scripts/load_pdf_to_db.py"
CMD="$CMD \"$PDF_PATH\""
CMD="$CMD $DATA_TYPE"
CMD="$CMD --index-pages $INDEX_PAGES"
CMD="$CMD --detail-pages $DETAIL_PAGES"
CMD="$CMD --db-host $DB_HOST"
CMD="$CMD --db-port $DB_PORT"
CMD="$CMD --db-name $DB_NAME"
CMD="$CMD --db-user $DB_USER"

if [ -n "$DB_PASSWORD" ]; then
    CMD="$CMD --db-password \"$DB_PASSWORD\""
fi

if [ -n "$REUSE_OCR" ]; then
    CMD="$CMD --reuse-ocr \"$REUSE_OCR\""
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

echo ""
echo -e "${YELLOW}실행 명령어:${NC}"
echo "  $CMD"
echo ""

# 실행
eval $CMD

# ============================================================================
# 완료
# ============================================================================

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  DB 구축 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}DB 확인 명령어:${NC}"
echo "  python db_scripts/check_db.py $DATA_TYPE"
echo ""
echo -e "${YELLOW}검색 테스트 명령어:${NC}"
echo "  python db_scripts/search_test.py \"검색어\" --type $DATA_TYPE"
echo ""
