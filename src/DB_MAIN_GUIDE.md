# B2G 기준데이터 DB 모듈 (db_main.py) 사용 가이드

## 개요

`db_main.py`는 정부 과제 PDF를 분석하여 MySQL에 저장하고 키워드 기반 검색을 제공하는 모듈입니다.

### 지원 데이터 타입

| 타입 | 테이블명 | 설명 |
|------|----------|------|
| `project` | national_projects | 국정과제 |
| `management_eval` | management_evals | 경영평가 지표 |
| `inclusive_growth` | inclusive_growth | 동반성장 세부추진과제 |

---

## 사전 요구사항

### 1. MySQL 설치 및 실행

```bash
# macOS (Homebrew)
brew install mysql
brew services start mysql

# 데이터베이스 생성
mysql -u root -e "CREATE DATABASE IF NOT EXISTS b2g_data CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### 2. Python 패키지 설치

```bash
pip install pymysql
```

---

## 데이터 스키마

### 국정과제 (StructuredProject)

```json
{
    "과제명": "세계에서 AI를 가장 잘 쓰는 나라 구현",
    "과제번호": "21",
    "과제 목표": ["AI 기술 선도국 도약", "AI 산업 생태계 육성"],
    "주요내용": ["AI 인재 양성", "AI 규제 혁신", "AI 인프라 구축"],
    "기대효과": ["글로벌 AI 경쟁력 확보", "새로운 일자리 창출"],
    "source_document": "presidential_agenda.pdf",
    "page_range": "15-18"
}
```

### 경영평가 (StructuredManagementEval)

```json
{
    "지표명": "리더십 및 전략기획",
    "평가기준": ["경영진의 전략적 비전 제시", "조직 목표 달성 노력"],
    "평가방법": ["정성평가", "계량평가 병행"],
    "참고사항": ["전년도 대비 개선 여부 확인"],
    "증빙자료": ["이사회 회의록", "전략 수립 보고서"],
    "source_document": "management_eval.pdf",
    "page_range": "26-28"
}
```

### 동반성장 (StructuredInclusiveGrowth)

```json
{
    "세부추진과제명": "상생금융 프로그램 확대",
    "세부내용": [
        "기업･금융권이 출연하고, 보증기관이 연계해 협력사 등을 지원",
        "대기업 등이 상생협력을 위하여 무역보험기금에 출연시 세액공제"
    ],
    "source_document": "inclusive_growth.pdf",
    "page_range": "26-28"
}
```

---

## 사용 방법

### 1. 파이프라인 생성

```python
from src.db_main import create_pipeline, create_store

# 전체 파이프라인 (PDF 처리 + 검색)
pipeline = create_pipeline(
    db_host="localhost",
    db_port=3306,
    db_name="b2g_data",
    db_user="root",
    db_password=""
)

# 검색만 사용 (DB 저장소만)
store = create_store(
    db_host="localhost",
    db_port=3306,
    db_name="b2g_data",
    db_user="root",
    db_password=""
)
```

### 2. PDF 처리 (데이터 추출 및 저장)

```python
import asyncio

async def process_document():
    pipeline = create_pipeline()
    
    # 국정과제 PDF 처리
    results = await pipeline.process_pdf(
        pdf_path="/path/to/national_agenda.pdf",
        data_type="project",           # "project" | "management_eval" | "inclusive_growth"
        index_pages=(3, 5),            # 목록이 있는 페이지 범위 (시작, 끝)
        detail_pages=(6, 100),         # 세부내용이 있는 페이지 범위
        save_intermediate=True,        # 중간 결과 저장 여부
        output_dir="./output",         # 중간 결과 저장 디렉토리
        reuse_ocr_dir="./output"       # 기존 OCR 결과 재사용 (선택)
    )
    
    print(f"처리된 항목: {len(results)}개")

asyncio.run(process_document())
```

### 3. 데이터 검색 (20개 후보 반환)

```python
from src.db_main import create_pipeline

pipeline = create_pipeline()

# 국정과제 검색
projects = pipeline.search_projects("인공지능 AI 기술", k=20)
for p in projects:
    print(f"[{p['score']:.2f}] {p['과제명']}")

# 경영평가 검색
evals = pipeline.search_management_evals("리더십 전략", k=20)
for e in evals:
    print(f"[{e['score']:.2f}] {e['지표명']}")

# 동반성장 검색
growths = pipeline.search_inclusive_growth("상생 협력", k=20)
for g in growths:
    print(f"[{g['score']:.2f}] {g['세부추진과제명']}")

# 전체 검색 (모든 타입)
all_results = pipeline.search_all("디지털 전환", k=20)
print(f"국정과제: {len(all_results['projects'])}개")
print(f"경영평가: {len(all_results['management_evals'])}개")
print(f"동반성장: {len(all_results['inclusive_growth'])}개")
```

### 4. DB 통계 확인

```python
stats = pipeline.get_stats()
print(stats)
# {'national_projects': 100, 'management_evals': 50, 'inclusive_growth': 30}
```

### 5. 데이터 삭제

```python
store = create_store()

# 특정 테이블만 삭제
store.delete_all_data("national_projects")

# 전체 삭제
store.delete_all_data()
```

---

## 파이프라인 처리 흐름

```
PDF 파일
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. PDF → 이미지 변환 (PyMuPDF)                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 2. 이미지 → OCR (CLOVA OCR, 표 인식 포함)        │
│    - ocr_page_1.json, ocr_page_2.json, ...      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 3. 목록 페이지 → 항목 리스트 추출 (LLM)          │
│    - 과제명/지표명 목록 추출                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 4. 세부 페이지 → 각 항목 상세정보 채우기 (LLM)   │
│    - 페이지별 순차 처리 (이전 페이지 컨텍스트)   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 5. 구조화 JSON → MySQL 저장                      │
│    - FULLTEXT 인덱스로 검색 지원                 │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 6. 검색: 키워드 → 20개 후보 반환                 │
│    - LLM이 그 중 10개 선별 (호출 측에서 처리)    │
└─────────────────────────────────────────────────┘
```

---

## 기존 코드 호환성

기존 `db.py` 코드와의 호환성을 위해 별칭이 제공됩니다:

```python
# 기존 코드
from src.db import B2GDataPipeline, InclusiveGrowthPipeline

# 새 코드 (동일하게 사용 가능)
from src.db_main import B2GDataPipeline, InclusiveGrowthPipeline

# 내부적으로 B2GPipeline으로 매핑됨
```

| 기존 이름 | 새 이름 (별칭) |
|-----------|---------------|
| `B2GDataPipeline` | `B2GPipeline` |
| `B2GVectorStore` | `MySQLStore` |
| `InclusiveGrowthPipeline` | `B2GPipeline` |
| `InclusiveGrowthVectorStore` | `MySQLStore` |
| `StructuredIndicator` | `StructuredManagementEval` |

---

## 검색 결과 구조

### 국정과제 검색 결과

```python
{
    "id": 1,
    "score": 15.5,
    "과제명": "세계에서 AI를 가장 잘 쓰는 나라 구현",
    "과제번호": "21",
    "과제 목표": ["AI 기술 선도국 도약", ...],
    "주요내용": ["AI 인재 양성", ...],
    "기대효과": ["글로벌 AI 경쟁력 확보", ...],
    "source_document": "presidential_agenda.pdf",
    "page_range": "15-18"
}
```

### 경영평가 검색 결과

```python
{
    "id": 1,
    "score": 12.3,
    "지표명": "리더십 및 전략기획",
    "평가기준": ["경영진의 전략적 비전 제시", ...],
    "평가방법": ["정성평가", ...],
    "참고사항": ["전년도 대비 개선 여부 확인"],
    "증빙자료": ["이사회 회의록", ...],
    "source_document": "management_eval.pdf",
    "page_range": "26-28"
}
```

### 동반성장 검색 결과

```python
{
    "id": 1,
    "score": 10.8,
    "세부추진과제명": "상생금융 프로그램 확대",
    "세부내용": ["기업･금융권이 출연하고...", ...],
    "source_document": "inclusive_growth.pdf",
    "page_range": "26-28"
}
```

---

## 주의사항

1. **MySQL 설정**: ngram 파서를 사용한 FULLTEXT 인덱스가 필요합니다. MySQL 5.7.6+ 또는 MariaDB 10.0.15+ 권장.

2. **OCR 재사용**: `reuse_ocr_dir` 옵션으로 이미 OCR 처리된 결과를 재사용하면 처리 시간이 크게 단축됩니다.

3. **검색 결과 20개**: 검색 결과는 항상 20개(기본값)까지 반환되며, LLM이 그 중 관련성 높은 10개를 선별하는 로직은 호출 측에서 구현해야 합니다.

4. **메모리 관리**: 대용량 PDF 처리 시 `_pdf_images`가 메모리에 유지되므로, 처리 완료 후 파이프라인 인스턴스를 삭제하거나 `pipeline._pdf_images = None`으로 해제하세요.

---

## 문의 및 지원

문제가 발생하면 로그를 확인하세요:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
