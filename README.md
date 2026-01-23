# RA_Agent

This repository is implementation of Agentic Pipeline using LLM that analyzes reports.

## 사용법

### 기본 실행

```bash
python main.py \
    --company "회사 이름" \
    --documents "문서이름1:경로1" "문서이름2:경로2" \
    --model openai \
    --model-name gpt-4o \
    --ocr CLOVA \
    --web \
    --max-rps 2.0 \
    --debug
```

### 명령줄 인자 설명

#### 필수 인자
- `-c, --company`: 회사 이름
- `-d, --documents`: 문서 이름과 경로 (형식: `이름:경로`)
  - 예시: `"doc1:data/doc1.pdf" "doc2:data/doc2.pdf"`

#### 선택 인자
- `-m, --model`: AI 모델 제공자 (기본값: `openai`)
  - 선택: `openai`, `gemini`
- `-mn, --model-name`: 구체적인 모델명 (기본값: 모델 제공자에 따라 자동 설정)
  - OpenAI: `gpt-4o` (기본값)
  - Gemini: `gemini-2.0-flash-exp` (기본값)
- `--ocr`: OCR API 제공자 (기본값: `CLOVA`)
  - 선택: `CLOVA`, `Upstage`
- `--web`: 웹 검색 활성화 (플래그)
- `--max-rps`: 초당 최대 요청 수 (기본값: `2.0`)
  - Upstage Tier 0: 1 RPS
  - Upstage Tier 1: 3 RPS
  - Upstage Tier 2: 10 RPS
- `--debug`: 디버그 모드 활성화 (플래그)

### 예시 스크립트

프로젝트에 포함된 bash 스크립트를 사용할 수 있습니다:

#### 1. 단일 문서 분석 (OpenAI + CLOVA)
```bash
bash src/scripts/run_example.sh
```

#### 2. 여러 문서 분석 (Gemini + Upstage)
```bash
bash src/scripts/run_multi_docs.sh
```

### 직접 실행 예시

#### OpenAI + CLOVA OCR
```bash
python main.py \
    -c "Example Corp" \
    -d "instruction1:data/instruction1.pdf" \
    -m openai \
    -mn gpt-4o \
    --ocr CLOVA \
    --web \
    --max-rps 2.0 \
    --debug
```

#### Gemini + Upstage OCR (여러 문서)
```bash
python main.py \
    -c "Multi Doc Company" \
    -d "doc1:data/doc1.pdf" "doc2:data/doc2.pdf" \
    -m gemini \
    -mn gemini-2.0-flash-exp \
    --ocr Upstage \
    --max-rps 3.0
```

## 출력 파일

분석 결과는 `src/results/{회사명}_{문서명1}_{문서명2}_...` 폴더에 저장됩니다:

### 폴더 구조 예시
```
src/results/
└── Example_instruction1/
    ├── Example_instruction1.json          # 최종 보고서 (모든 report_type 포함)
    ├── instruction1.json                  # 페이지별 분석 결과
    ├── instruction1_ocr.json              # OCR 추출 텍스트 (페이지별)
    └── debug.txt                          # 디버그 로그 (--debug 옵션 사용시)
```

### 파일 설명
- **`{회사명}_{문서명들}.json`**: 최종 보고서
  - 모든 보고서 유형(competencies, b2g_strategy, market)의 결과 포함
- **`{문서명}.json`**: 각 문서의 페이지별 상세 분석 결과
- **`{문서명}_ocr.json`**: 각 문서의 OCR 추출 텍스트
  - 형식: `{"page_0": "텍스트...", "page_1": "텍스트...", ...}`
- **`debug.txt`**: 디버그 로그 (--debug 옵션 사용시에만 생성)
  - 각 페이지 분석 시작/완료 시간, 전체 소요 시간 등 기록

### 여러 문서 처리 시
```
src/results/
└── Multi_Doc_Company_doc1_doc2/
    ├── Multi_Doc_Company_doc1_doc2.json   # 최종 통합 보고서
    ├── doc1.json                          # doc1 페이지별 분석
    ├── doc1_ocr.json                      # doc1 OCR 결과
    ├── doc2.json                          # doc2 페이지별 분석
    ├── doc2_ocr.json                      # doc2 OCR 결과
    └── debug.txt                          # 디버그 로그
```

