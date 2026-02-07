#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# JSON 데이터를 MariaDB/MySQL에 로드하는 스크립트 (schema-first 대응)
#
# 사용법:
#   chmod +x load_to_db.sh
#   ./load_to_db.sh                          # 전체 로드
#   ./load_to_db.sh --reset                  # 전체 초기화 후 로드
#   ./load_to_db.sh project                  # 국정과제만 로드
#   ./load_to_db.sh management_eval          # 경영평가만 로드
#   ./load_to_db.sh inclusive_growth         # 동반성장만 로드
#   ./load_to_db.sh --reset project          # 국정과제만 초기화 후 로드
#

set -euo pipefail

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
if [[ -f "../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  JSON → MariaDB 데이터 로드 스크립트 (schema-first)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================================
# DB 설정 (필요시 수정)
# ============================================================================
DB_HOST="localhost"
DB_PORT="3306"
DB_NAME="b2g_data"
DB_USER="root"
DB_PASSWORD=""   # 비번 필요하면 여기에 입력하거나 MYSQL_PWD 사용

# ============================================================================
# 인자 파싱
# ============================================================================
DATA_TYPE="all"
RESET_FLAG=""

show_help() {
  echo "사용법: $0 [옵션] [데이터타입]"
  echo ""
  echo "데이터 타입:"
  echo "  project           국정과제만 로드"
  echo "  management_eval   경영평가만 로드"
  echo "  inclusive_growth  동반성장만 로드"
  echo "  all               전체 로드 (기본값)"
  echo ""
  echo "옵션:"
  echo "  --reset, -r       기존 데이터 초기화 후 로드"
  echo "  --help, -h        도움말 표시"
  echo ""
  echo "예시:"
  echo "  $0"
  echo "  $0 --reset"
  echo "  $0 project"
  echo "  $0 management_eval"
  echo "  $0 --reset inclusive_growth"
}

for arg in "$@"; do
  case "$arg" in
    --reset|-r)
      RESET_FLAG="--reset"
      ;;
    project|management_eval|inclusive_growth|all)
      DATA_TYPE="$arg"
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo -e "${RED}❌ 알 수 없는 인자: $arg${NC}"
      echo ""
      show_help
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}설정:${NC}"
echo "  DB: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
echo "  데이터 타입: $DATA_TYPE"
echo "  초기화: ${RESET_FLAG:-아니오}"
echo ""

# ============================================================================
# JSON 파일 확인
# ============================================================================
echo -e "${YELLOW}JSON 파일 확인:${NC}"

count_json_items() {
  local file_path="$1"
  python3 - "$file_path" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else 0)
PY
}

if [[ -f "$SCRIPT_DIR/DB_data/project.json" ]]; then
  PROJECT_COUNT="$(count_json_items "$SCRIPT_DIR/DB_data/project.json")"
  echo -e "  ✅ project.json: ${GREEN}${PROJECT_COUNT}개 항목${NC}"
else
  echo -e "  ⚠️  project.json: ${RED}파일 없음${NC}"
fi

if [[ -f "$SCRIPT_DIR/DB_data/management.json" ]]; then
  MANAGEMENT_COUNT="$(count_json_items "$SCRIPT_DIR/DB_data/management.json")"
  echo -e "  ✅ management.json: ${GREEN}${MANAGEMENT_COUNT}개 항목${NC}"
else
  echo -e "  ⚠️  management.json: ${RED}파일 없음${NC}"
fi

if [[ -f "$SCRIPT_DIR/DB_data/inclusive.json" ]]; then
  INCLUSIVE_COUNT="$(count_json_items "$SCRIPT_DIR/DB_data/inclusive.json")"
  echo -e "  ✅ inclusive.json: ${GREEN}${INCLUSIVE_COUNT}개 항목${NC}"
else
  echo -e "  ⚠️  inclusive.json: ${RED}파일 없음${NC}"
fi

echo ""

# ============================================================================
# MySQL 서비스 확인
# ============================================================================
echo -e "${YELLOW}MySQL 서비스 확인 중...${NC}"

MYSQL_AUTH_ARGS=(-u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT")
if [[ -n "$DB_PASSWORD" ]]; then
  MYSQL_AUTH_ARGS+=(-p"$DB_PASSWORD")
fi

if ! mysql "${MYSQL_AUTH_ARGS[@]}" -e "SELECT 1;" > /dev/null 2>&1; then
  echo -e "${RED}❌ MySQL 서비스에 연결할 수 없습니다.${NC}"
  echo ""
  echo "확인할 것:"
  echo "  1) MySQL 실행 여부"
  echo "  2) DB_USER/DB_PASSWORD/DB_HOST/DB_PORT"
  echo "  3) root 비밀번호 설정 여부"
  echo ""
  echo "macOS(Homebrew) 예시:"
  echo "  brew services start mysql"
  echo ""
  exit 1
fi

echo -e "${GREEN}✅ MySQL 서비스 정상${NC}"
echo ""

# ============================================================================
# DB 존재 확인 및 생성
# ============================================================================
echo -e "${YELLOW}데이터베이스 확인 중...${NC}"

DB_EXISTS="$(mysql "${MYSQL_AUTH_ARGS[@]}" -Nse "SHOW DATABASES LIKE '$DB_NAME';" | wc -l | tr -d ' ')"

if [[ "$DB_EXISTS" -eq 0 ]]; then
  echo "  데이터베이스 '$DB_NAME' 생성 중..."
  mysql "${MYSQL_AUTH_ARGS[@]}" -e "CREATE DATABASE \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
  echo -e "${GREEN}✅ 데이터베이스 생성 완료${NC}"
else
  echo -e "${GREEN}✅ 데이터베이스 '$DB_NAME' 존재${NC}"
fi

echo ""

# ============================================================================
# Python 스크립트 실행
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  데이터 로드 시작${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PY_ARGS=(
  "$SCRIPT_DIR/load_json_to_db.py"
  --type "$DATA_TYPE"
  --db-host "$DB_HOST"
  --db-port "$DB_PORT"
  --db-name "$DB_NAME"
  --db-user "$DB_USER"
)

if [[ -n "$DB_PASSWORD" ]]; then
  PY_ARGS+=(--db-password "$DB_PASSWORD")
fi

if [[ -n "$RESET_FLAG" ]]; then
  PY_ARGS+=("$RESET_FLAG")
fi

python "${PY_ARGS[@]}"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  완료!${NC}"
echo -e "${BLUE}============================================================${NC}"
