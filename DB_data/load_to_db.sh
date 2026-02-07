#!/bin/bash
# -*- coding: utf-8 -*-
#
# JSON 데이터를 MariaDB/MySQL에 로드하는 스크립트
#
# 사용법:
#   chmod +x load_to_db.sh
#   ./load_to_db.sh              # 전체 로드
#   ./load_to_db.sh --reset      # DB 초기화 후 로드
#   ./load_to_db.sh project      # 국정과제만 로드
#   ./load_to_db.sh management   # 경영평가만 로드
#   ./load_to_db.sh inclusive    # 동반성장만 로드
#

set -e

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
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  JSON → MariaDB 데이터 로드 스크립트${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================================
# DB 설정 (필요시 수정)
# ============================================================================
DB_HOST="localhost"
DB_PORT="3306"
DB_NAME="b2g_data"
DB_USER="root"
DB_PASSWORD=""

# ============================================================================
# 스키마 정보 로드 (동적)
# ============================================================================
SCHEMA_KEYS=$(python3 "$PROJECT_DIR/db_scripts/get_schema_keys.py")
if [ $? -ne 0 ]; then
    echo -e "${RED}스키마 정보를 불러오는데 실패했습니다.${NC}"
    exit 1
fi
SUPPORTED_TYPES="all $SCHEMA_KEYS"

# ============================================================================
# 인자 파싱
# ============================================================================
DATA_TYPE="all"
RESET_FLAG=""

for arg in "$@"; do
    case $arg in
        --reset|-r)
            RESET_FLAG="--reset"
            ;;
        --help|-h)
            echo "사용법: $0 [옵션] [데이터타입]"
            echo ""
            echo "데이터 타입:"
            for key in $SCHEMA_KEYS; do
                echo "  $key"
            done
            echo "  all          전체 로드 (기본값)"
            echo ""
            echo "옵션:"
            echo "  --reset, -r  기존 데이터 삭제 후 로드"
            echo "  --help, -h   도움말 표시"
            echo ""
            exit 0
            ;;
        *)
            # 동적으로 타입 확인
            FOUND=0
            for key in $SUPPORTED_TYPES; do
                if [ "$arg" == "$key" ]; then
                    DATA_TYPE="$arg"
                    FOUND=1
                    break
                fi
            done
            
            if [ $FOUND -eq 0 ]; then
                echo -e "${RED}알 수 없는 인자 또는 데이터 타입: $arg${NC}"
                echo "사용 가능한 타입: $SUPPORTED_TYPES"
                exit 1
            fi
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

CHECK_TYPES=""
if [ "$DATA_TYPE" == "all" ]; then
    CHECK_TYPES="$SCHEMA_KEYS"
else
    CHECK_TYPES="$DATA_TYPE"
fi

for type in $CHECK_TYPES; do
    JSON_PATH=$(python3 "$PROJECT_DIR/db_scripts/get_json_path.py" "$type")
    
    if [ -f "$JSON_PATH" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$JSON_PATH'))))")
        echo -e "  ✅ $type: ${GREEN}${COUNT}개 항목${NC} ($JSON_PATH)"
    else
        echo -e "  ⚠️  $type: ${RED}파일 없음${NC} ($JSON_PATH)"
    fi
done

echo ""

# ============================================================================
# MySQL 서비스 확인
# ============================================================================
echo -e "${YELLOW}MySQL 서비스 확인 중...${NC}"

if ! mysql -u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT" -e "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}❌ MySQL 서비스에 연결할 수 없습니다.${NC}"
    echo ""
    echo "다음 명령어로 MySQL을 시작하세요:"
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

DB_EXISTS=$(mysql -u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT" -e "SHOW DATABASES LIKE '$DB_NAME';" | grep -c "$DB_NAME" || true)

if [ "$DB_EXISTS" -eq 0 ]; then
    echo "  데이터베이스 '$DB_NAME' 생성 중..."
    mysql -u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT" -e "CREATE DATABASE $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
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

python "$PROJECT_DIR/load_json_to_db.py" \
    --type "$DATA_TYPE" \
    --db-host "$DB_HOST" \
    --db-port "$DB_PORT" \
    --db-name "$DB_NAME" \
    --db-user "$DB_USER" \
    $RESET_FLAG

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  완료!${NC}"
echo -e "${BLUE}============================================================${NC}"