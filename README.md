# RA_Agent - 기업 분석 및 보고서 생성 시스템

PDF 문서를 분석하여 기업 보고서를 자동 생성하는 AI 파이프라인입니다.

---
## 빠른 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```
### 2. DB구축
```bash
# macOS (Homebrew)
brew install mariadb
brew services start mariadb

# Ubuntu/Debian
sudo apt install mariadb-server
sudo systemctl start mariadb
```
```bash
# 전체 데이터 로드 (국정과제, 경영평가, 동반성장)
python load_json_to_db.py --reset
```

### 3. FastAPI 실행
```bash
# 프로젝트 루트 디렉토리에서 실행
cd RA_Agent

# 서버 실행 (기본 포트: 8000)
python server.py

# 또는 uvicorn으로 직접 실행
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 루트에 `.env` 파일 추가

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

### 5. 요청 보내기

아래와 같은 형식으로 ```/analysis```에 요청
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
Swagger UI : http://localhost:8000/docs 에서 데모 확인 가능.

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
기존에 생성한 기준 데이터 분석 JSON이 있을 경우, `load_to_db.sh` 스크립트를 사용하여 DB 구성이 가능합니다. 이 스크립트는 `src/db_main.py`의 설정을 동적으로 읽어 동작합니다.

```bash
# 실행 권한 부여
chmod +x DB_data/load_to_db.sh

# 전체 데이터 로드 (국정과제, 경영평가, 동반성장)
./DB_data/load_to_db.sh

# 특정 타입만 로드 (스키마에 정의된 키 사용)
./DB_data/load_to_db.sh project            # 국정과제
./DB_data/load_to_db.sh management_eval    # 경영평가
./DB_data/load_to_db.sh inclusive_growth   # 동반성장

# DB 초기화 후 로드
./DB_data/load_to_db.sh --reset
```

### 2.4 DB 재생성
만약 새로운 문서로 DB를 다시 구축하고 싶다면 다음과 같이 합니다.

```bash
# 사용법
./db_scripts/build_db.sh <pdf_path> <data_type> <index_pages> <detail_pages> [options]

# 필수 인자
#   pdf_path       PDF 파일 경로
#   data_type      데이터 종류 (project | management_eval | inclusive_growth)
#   index_pages    목록 페이지 범위 (예: 13-17)
#   detail_pages   세부내용 페이지 범위 (예: 21-195)

# 선택 옵션
#   --use-table          Table 인식 OCR 사용 (기본값: text만)
#   --reuse-ocr <dir>    기존 OCR 결과 재사용 (OCR 단계 스킵)
#   --output-dir <dir>   중간 결과 JSON 저장 디렉토리
#   --db-host <host>     MySQL 호스트 (기본값: localhost)
#   --db-port <port>     MySQL 포트 (기본값: 3306)
#   --db-name <name>     DB 이름 (기본값: b2g_data)
#   --db-user <user>     MySQL 사용자 (기본값: root)
#   --db-password <pw>   MySQL 비밀번호

# OCR 모드
#   기본값 (--use-table 없음): CLOVA_ocr 사용 → text 필드만 추출
#   --use-table 지정 시:       CLOVA_ocr_with_table 사용 → table 필드 포함
```

이때, 높은 정확도를 위해 국정과제, 세부지표 등이 있는 부분만 입력으로 받아 분석을 진행하며, 페이지 범위는 다음과 같이 설정합니다.

- 목록 페이지 범위 : 국정과제, 세부지표 등이 있는 전체 목록 페이지
- 세부내용 페이지 범위 : 위의 목록에 대한 설명이 있는 페이지

**예시:**

```bash
# 국정과제 PDF 처리 (text OCR)
./db_scripts/build_db.sh ./data/criteria/presidential_agenda.pdf project 13-17 21-195

# 경영평가 PDF 처리 (table 인식 포함)
./db_scripts/build_db.sh ./data/criteria/management_eval.pdf management_eval 20-25 27-46 --use-table

# 동반성장 PDF 처리 (table 인식 + 기존 OCR 재사용)
./db_scripts/build_db.sh ./data/criteria/inclusive_growth.pdf inclusive_growth 2-5 6-50 --use-table --reuse-ocr ./ocr_cache/inclusive
```

이렇게 추출된 기준자료는 `DB_data` 폴더 내에 저장됩니다.

**새로운 문서의 스키마를 변경하려면 `src/db_main.py` 파일 내 `SCHEMA_REGISTRY` 변수만 수정하면 됩니다.**

이곳에서 데이터 타입, 테이블명, 임베딩 테이블명, JSON 파일 경로, 필드 정의 등을 통합 관리합니다.
`load_to_db.sh` 스크립트는 이 설정을 동적으로 읽어 동작하므로, 별도의 스크립트 수정 없이 스키마 변경 사항이 반영됩니다.

### 기본 스키마 예시

다음은 `src/db_main.py`에 정의된 기본 스키마 구조입니다.

국정과제 스키마 : 
```json
[{
  "번호": "과제 번호",
  "과제명": "과제 이름",
  "과제_목표": ["과제의 목표/비전"],
  "주요내용": ["주요 추진 내용 상세"],
  "기대효과": ["기대되는 효과/성과"]
}]
```

동반성장 스키마 : 
```json
[{
    "번호": "지표 번호",
    "세부추진과제명": "세부추진과제 이름",
    "세부내용": ["세부 내용 상세"]
}] 
```

경영평가 스키마 : 
```json
[{
    "번호": "지표 번호",
    "지표명": "지표 이름",
    "평가기준": ["평가 기준 상세"],
    "평가방법": ["평가 방법 상세"],
    "참고사항": ["참고할 내용"],
    "증빙자료": ["필요한 증빙자료"]
}]
```

이렇게 저장된 JSON파일은 `DB_data/load_to_db.sh` 스크립트를 통해 DB에 저장됩니다. 이때 생성되는 테이블은 총 6개(실제로 검색이 진행되는 테이블은 3개)이며, 스키마는 다음과 같습니다. 

#### 1️⃣ 메인 테이블 (3개) - 파이프라인 내 검색에 사용되지 않음

**1. national_projects (국정과제)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| 과제명 | VARCHAR(500) | 과제 이름 |
| 과제번호 | VARCHAR(50) | 과제 번호 |
| 과제_목표 | JSON | 과제 목표 리스트 |
| 주요내용 | JSON | 주요 추진 내용 리스트 |
| 기대효과 | JSON | 기대 효과 리스트 |
| source_document | VARCHAR(255) | 원본 문서명 |
| page_range | VARCHAR(50) | 페이지 범위 |
| extraction_date | DATETIME | 추출 일시 |
| searchable_text | TEXT | 전문검색용 텍스트 (FULLTEXT 인덱스) |
| created_at | TIMESTAMP | 생성 일시 |

**2. management_evals (경영평가)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| 지표명 | VARCHAR(500) | 지표 이름 |
| 평가기준 | JSON | 평가 기준 리스트 |
| 평가방법 | JSON | 평가 방법 리스트 |
| 참고사항 | JSON | 참고 사항 리스트 |
| 증빙자료 | JSON | 필요 증빙자료 리스트 |
| source_document | VARCHAR(255) | 원본 문서명 |
| page_range | VARCHAR(50) | 페이지 범위 |
| extraction_date | DATETIME | 추출 일시 |
| searchable_text | TEXT | 전문검색용 텍스트 (FULLTEXT 인덱스) |
| created_at | TIMESTAMP | 생성 일시 |

**3. inclusive_growth (동반성장)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| 세부추진과제명 | VARCHAR(500) | 세부추진과제 이름 |
| 세부내용 | JSON | 세부 내용 리스트 |
| source_document | VARCHAR(255) | 원본 문서명 |
| page_range | VARCHAR(50) | 페이지 범위 |
| extraction_date | DATETIME | 추출 일시 |
| searchable_text | TEXT | 전문검색용 텍스트 (FULLTEXT 인덱스) |
| created_at | TIMESTAMP | 생성 일시 |

#### 2️⃣ 임베딩 테이블 (3개) - 벡터 검색에 사용됨

메인 테이블의 각 필드를 문장 단위로 분리하여 OpenAI 임베딩(text-embedding-3-small, 1536차원)을 저장합니다.

**4. embedding_chunks_project (국정과제 임베딩)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| source_id | INT | national_projects.id 참조 |
| item_name | VARCHAR(500) | 과제명 |
| field_type | VARCHAR(100) | 필드 유형 (과제_목표, 주요내용, 기대효과) |
| chunk_text | TEXT | 원본 텍스트 청크 |
| embedding | BLOB | 임베딩 벡터 (1536차원 float) |
| created_at | TIMESTAMP | 생성 일시 |

**5. embedding_chunks_management (경영평가 임베딩)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| source_id | INT | management_evals.id 참조 |
| item_name | VARCHAR(500) | 지표명 |
| field_type | VARCHAR(100) | 필드 유형 (평가기준, 평가방법, 참고사항, 증빙자료) |
| chunk_text | TEXT | 원본 텍스트 청크 |
| embedding | BLOB | 임베딩 벡터 (1536차원 float) |
| created_at | TIMESTAMP | 생성 일시 |

**6. embedding_chunks_inclusive (동반성장 임베딩)**
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| id | INT | Primary Key |
| source_id | INT | inclusive_growth.id 참조 |
| item_name | VARCHAR(500) | 세부추진과제명 |
| field_type | VARCHAR(100) | 필드 유형 (세부내용) |
| chunk_text | TEXT | 원본 텍스트 청크 |
| embedding | BLOB | 임베딩 벡터 (1536차원 float) |
| created_at | TIMESTAMP | 생성 일시 |

실제로 검색이 진행되는 벡터 테이블은 정확도 높은 키워드 기반 검색을 위해 각 과제/지표별로 **모든 텍스트를 포함하는 것이 아닌** 모든 필드 안의 텍스트들을 나누어 각각 임베딩하여 저장합니다. 

**파이프라인 실행 순서:**
1. PDF 로드 및 이미지 변환
2. OCR 수행 (또는 기존 OCR 재사용)
3. LLM으로 항목 추출 → JSON 저장
4. OpenAI 임베딩 생성
5. MySQL DB 저장

**DB 확인 및 검색 테스트:**

```bash
# DB 내용 확인
python db_scripts/check_db.py                    # 전체 통계
python db_scripts/check_db.py project            # 국정과제 조회
python db_scripts/check_db.py management_eval    # 경영평가 조회
python db_scripts/check_db.py inclusive_growth   # 동반성장 조회

# 검색 테스트 (임베딩 기반 벡터 검색)
python db_scripts/search_test.py "인공지능 AI"
python db_scripts/search_test.py "상생 협력" --type inclusive_growth
```


### 2.5 DB 연결 정보

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
    "section1" : {
        "scores": {
            "management_eval": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            },
            "national_agenda": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            },
            "co_growth": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            }
        },
        "radar": [
            { "axis": "정책정합성", "score": 0 },
            { "axis": "공공기관적합도", "score": 0 },
            { "axis": "시장성장성", "score": 0 },
            { "axis": "동반성장·협력", "score": 0 },
            { "axis": "신뢰도·레퍼런스", "score": 0 }
        ],
        "overall": {
            "overall_score": 0,
            "grade": "",
            "grade_rule": "",
            "reason_summary": "",
            "needs_more_data_global": [],
            "keywords": []
        }
    },
    "section2" : {
        "finance" : {
            "revenue" : {
                "year" : 0,
                "amount" : ""
            },
            "profit" : {
                "year" : 0,
                "amount" : ""
            },
            "invest" : {
                "year" : 0,
                "amount" : ""
            }
        },
        "performance" : {
            "contents" : []
        },
        "BM" : {
            "contents" : []
        },
        "competencies" : {
            "b2g_keywords" : [],
            "evidences" : []
        }
    },
    "section3" : {
        "market_growth" : 0.0,
        "market_size" : {
            "unit" : "",
            "market_name" : "",
            "reference" : "",
            "data" : {
            }
        },
        "competition" : {
            "competitors" : [],
            "details" : [],
            "differentiation" : []
        },
        "tech_policy_trends" : {
            "keywords" : [],
            "evidences" : [
                {
                    "content" : "",
                    "source" : ""
                },
                {
                    "content" : "",
                    "source" : ""
                }
            ]
        }
    },
    "section4" : {
        "presidential_agenda" : {
            "top10" : [
            {
                "rank" : "",
                "name" : "",
                "description" : ""
            }],
            "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "management_eval" : {
            "top10" : [
                {
                    "rank" : "",
                    "name" : "",
                    "description" : ""
                }
            ],
           "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "inclusive_growth" : {
            "top10" : [
                {
                    "rank" : "",
                    "name" : "",
                    "description" : ""
                }
            ],
            "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "overall" : {
            "rank" : "",
            "expect" : []
        }
    },
    "section5" : {
        "weakness_analysis" : {
            "keyword" : "",
            "evidences" : []
        },
        "strategy" : {
            "keyword" : "",
            "strategy" : "",
            "details" : []
        },
        "to_do_list" : {
            "keyword" : "",
            "tasks" : [
                {
                    "content" : "",
                    "details" : []
                },
                {
                    "content" : "",
                    "details" : []
                }
            ]
        }
    }
}
```

### 4.6 기타 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/schema` | GET | 출력 JSON 스키마 조회 |
| `/docs` | GET | Swagger API 문서 |
| `/redoc` | GET | ReDoc API 문서 |
