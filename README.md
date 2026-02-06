# RA_Agent - 기업 분석 및 보고서 생성 시스템

PDF 문서를 분석하여 기업 보고서를 자동 생성하는 AI 파이프라인입니다.

---

## 1. 환경설정 방법

### 1.1 Python 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 1.2 환경변수 설정

`src/.env` 파일을 생성하고 아래 API 키들을 설정합니다:

```env
# OpenAI API (필수)
OPENAI_KEY=sk-your-openai-api-key

# Google Gemini API (필수)
GEMINI_KEY=your-gemini-api-key

# Naver CLOVA OCR (필수)
CLOVA_api_url=https://your-clova-endpoint.apigw.ntruss.com/custom/v1/...
CLOVA_secret_key=your-clova-secret-key

# AWS S3 (FastAPI 서버 사용 시)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

---

## 2. MariaDB 구축 방법

### 2.1 MariaDB 설치

```bash
# macOS (Homebrew)
brew install mariadb
brew services start mariadb

# Ubuntu/Debian
sudo apt install mariadb-server
sudo systemctl start mariadb
```

### 2.2 데이터베이스 생성

```sql
-- MariaDB 접속
mysql -u root -p

-- 데이터베이스 생성
CREATE DATABASE b2g_data CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2.3 기준 데이터 로드

```bash
cd DB_data

# 전체 데이터 로드 (국정과제, 경영평가, 동반성장)
python load_json_to_db.py

# 특정 타입만 로드
python load_json_to_db.py --type project      # 국정과제
python load_json_to_db.py --type management   # 경영평가
python load_json_to_db.py --type inclusive    # 동반성장

# DB 초기화 후 로드
python load_json_to_db.py --reset
```

### 2.4 DB 연결 정보

기본 연결 정보:
- Host: `localhost`
- Port: `3306`
- Database: `b2g_data`
- User: `root`

---

## 3. FastAPI 실행 방법

### 3.1 서버 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
cd RA_Agent

# 서버 실행 (기본 포트: 8000)
python server.py

# 또는 uvicorn으로 직접 실행
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 3.2 API 문서 확인

서버 실행 후 아래 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 4. /analysis API 설명

### 4.1 개요

S3에서 PDF 파일을 다운로드하여 기업 분석을 실행하고 결과를 JSON으로 반환합니다.

### 4.2 엔드포인트

```
POST /analysis
```

### 4.3 요청 파라미터

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `region` | string | ✅ | AWS 리전 (예: `ap-northeast-2`) |
| `bucket` | string | ✅ | S3 버킷 이름 |
| `object_key` | list[string] | ✅ | S3 오브젝트 키 리스트 |
| `company_name` | string | ❌ | 회사 이름 (기본값: `분석대상기업`) |
| `web_search` | boolean | ❌ | 웹 검색 활성화 여부 (기본값: `false`) |
| `max_rps` | float | ❌ | 초당 최대 API 요청 수 (기본값: `2.0`) |
| `debug` | boolean | ❌ | 디버그 모드 (기본값: `false`) |

### 4.4 요청 예시

```json
{
  "region": "ap-northeast-2",
  "bucket": "my-bucket",
  "object_key": ["reports/company_ir.pdf", "reports/business_plan.pdf"],
  "company_name": "테스트기업",
  "web_search": true,
  "max_rps": 2.0,
  "debug": false
}
```

### 4.5 응답 구조

```json
{
  "section1": {
    "기업명": "...",
    "기업 한줄 요약": "...",
    "기업분류": "...",
    "핵심 역량": ["..."],
    "주요 제품/서비스": ["..."],
    "투자유치 및 재무지표": "...",
    "수상 및 인증 실적": ["..."],
    "지식재산권": "..."
  },
  "section2": {
    "핵심 기술 키워드": ["..."],
    "기술 역량 요약": "...",
    "기술적 차별점 및 경쟁 우위": "...",
    "적용 가능 산업 분야": ["..."],
    "기술 성숙도(TRL)": "..."
  },
  "section3": {
    "시장분석 요약": "...",
    "목표시장 및 성장성": "...",
    "경쟁 현황 및 포지셔닝": "..."
  },
  "section4": {
    "국정과제 관련 지표": [...],
    "경영평가 관련 지표": [...],
    "동반성장 관련 지표": [...]
  },
  "section5": {
    "B2G 전략 방향": "...",
    "공공시장 진입 전략": "..."
  }
}
```

### 4.6 기타 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/schema` | GET | 출력 JSON 스키마 조회 |
| `/docs` | GET | Swagger API 문서 |
| `/redoc` | GET | ReDoc API 문서 |
