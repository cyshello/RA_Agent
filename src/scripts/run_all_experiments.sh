#!/bin/bash

# 4개 기업 전체 실험 스크립트
# 
# 사용법:
#   bash src/scripts/run_all_experiments.sh          # 전체 실행
#   bash src/scripts/run_all_experiments.sh nudge    # 특정 기업만 실행
#   bash src/scripts/run_all_experiments.sh all      # 전체 실행

# 스크립트 디렉토리에서 프로젝트 루트로 이동
cd "$(dirname "$0")/../.." || exit

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate AX

# 공통 설정
# 모델 설정은 하드코딩됨:
#   - 문서 추출 및 Section 1, 2: OpenAI GPT-4o
#   - Section 3, 4, 5: Google Gemini 2.0 Pro
MAX_RPS="2.0"
DEBUG_FLAG="--debug"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_separator() {
    echo -e "${YELLOW}============================================================${NC}"
}

# 실험 실행 함수
run_experiment() {
    local company_name=$1
    local documents=$2
    local web_flag=$3
    
    log_separator
    log_info "실험 시작: $company_name"
    log_info "문서: $documents"
    log_info "모델: 문서 추출/Section1,2=GPT-4o | Section3,4,5=Gemini-2.0-Pro"
    log_separator
    
    python3 main.py \
        --company "$company_name" \
        --documents $documents \
        --max-rps "$MAX_RPS" \
        $web_flag \
        $DEBUG_FLAG
    
    if [ $? -eq 0 ]; then
        log_success "$company_name 실험 완료"
    else
        log_error "$company_name 실험 실패"
        return 1
    fi
}

# Neopharm 실험
run_neopharm() {
    run_experiment "Neopharm" "IR2:data/IR2.pdf" "--web"
}

# Nudge Healthcare 실험
run_nudge() {
    run_experiment "Nudge_Healthcare" "IR1:data/IR1.pdf" ""
}

# Test Company 실험 (여러 문서)
run_test() {
    run_experiment "Test_Company" "intro:data/test/intro.pdf plan:data/test/business_plan.pdf" "--web"
}

# ISens 실험
run_isens() {
    run_experiment "ISens" "IR3:data/IR3.pdf" "--web"
}

# 전체 실험 실행
run_all() {
    log_separator
    echo -e "${YELLOW}       4개 기업 전체 실험 시작${NC}"
    log_separator
    
    local start_time=$(date +%s)
    local success_count=0
    local fail_count=0
    
    # 1. Neopharm
    run_neopharm
    if [ $? -eq 0 ]; then ((success_count++)); else ((fail_count++)); fi
    
    # 2. Nudge Healthcare
    run_nudge
    if [ $? -eq 0 ]; then ((success_count++)); else ((fail_count++)); fi
    
    # 3. Test Company
    run_test
    if [ $? -eq 0 ]; then ((success_count++)); else ((fail_count++)); fi
    
    # 4. ISens
    run_isens
    if [ $? -eq 0 ]; then ((success_count++)); else ((fail_count++)); fi
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    log_separator
    echo -e "${YELLOW}       전체 실험 결과${NC}"
    log_separator
    log_info "성공: $success_count / 4"
    log_info "실패: $fail_count / 4"
    log_info "총 소요 시간: ${elapsed}초"
    log_separator
}

# 메인 로직
case "${1:-all}" in
    neopharm)
        run_neopharm
        ;;
    nudge)
        run_nudge
        ;;
    test)
        run_test
        ;;
    isens)
        run_isens
        ;;
    all)
        run_all
        ;;
    *)
        echo "사용법: $0 {neopharm|nudge|test|isens|all}"
        echo ""
        echo "옵션:"
        echo "  neopharm  - Neopharm 기업만 실행"
        echo "  nudge     - Nudge Healthcare 기업만 실행"
        echo "  test      - Test Company 기업만 실행"
        echo "  isens     - ISens 기업만 실행"
        echo "  all       - 전체 기업 실행 (기본값)"
        exit 1
        ;;
esac
