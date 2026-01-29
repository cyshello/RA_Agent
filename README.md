# RA_Agent: LangChain 기반 기업 분석 및 보고서 생성 파이프라인

## 📋 프로젝트 개요

RA_Agent는 기업 IR 문서(PDF)를 자동으로 분석하고 전문가 수준의 분석 보고서를 생성하는 LangChain 기반 AI 파이프라인입니다.

### 주요 기능

- **PDF 문서 처리**: PDF를 페이지 단위 이미지로 변환 및 분석
- **OCR 통합**: CLOVA OCR 및 Upstage OCR 지원
- **멀티모달 분석**: 이미지와 텍스트를 동시에 분석하여 구조화된 JSON 추출
- **보고서 생성**: 3가지 유형의 전문 분석 보고서 자동 생성
  - 회사 현황 및 핵심역량 분석
  - 사업시장 현황 분석
  - B2G 전략 방향 수립
- **다중 LLM 지원**: OpenAI (GPT-4o 등), Google Gemini 모델 선택 가능
- **웹 검색 통합**: Gemini의 grounding 기능을 통한 실시간 정보 검색
- **Rate Limiting**: API 호출 제어로 비용 최적화
- **캐싱**: 분석 결과 저장 및 재사용

---

## 🏗️ 아키텍처

### LangChain 기반 구조

```
┌─────────────────────────────────────────────────────┐
│                   Main Pipeline                      │
│              (main.py - Company 클래스)              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ├─► Document Processing
                  │   ├─ PDF → Images (pdf2image)
                  │   ├─ OCR Chain (CLOVA/Upstage)
                  │   └─ Page Extraction Chain
                  │      ├─ LangChain Prompt Template
                  │      ├─ ChatModel (OpenAI/Gemini)
                  │      └─ JSON Output Parser
                  │
                  └─► Report Generation Chains
                      ├─ Competencies Report Chain
                      ├─ Market Analysis Chain
                      └─ B2G Strategy Chain
```

### 디렉토리 구조

```
RA_Agent/
├── main.py                 # 메인 파이프라인
├── requirements.txt        # Python 의존성
├── README.md              # 이 문서
├── data/                  # 입력 PDF 문서
├── src/
│   ├── api.py            # LangChain 모델 래퍼 & Dispatcher
│   ├── prompts.py        # ChatPromptTemplate 정의
│   ├── utils.py          # OCR, JSON 파싱, 추출 Chain
│   ├── scripts/          # 실행 스크립트
│   └── results/          # 분석 결과 저장
└── .env                  # API 키 설정
```

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 `src/` 디렉토리에 생성:

```env
# OpenAI API
OPENAI_KEY=your_openai_api_key

# Google Gemini API
GEMINI_KEY=your_gemini_api_key

# CLOVA OCR API
CLOVA_api_url=your_clova_api_url
CLOVA_secret_key=your_clova_secret_key

# Upstage OCR API (선택)
UPSTAGE_api_key=your_upstage_api_key
```

### 3. 기본 사용법

```bash
python main.py \
  -c "Example Corp" \
  -d IR_deck:data/ir_deck.pdf \
  -em openai -emn gpt-4o \
  -rm openai -rmn gpt-4o \
  --ocr CLOVA \
  --max-rps 2.0
```

---

## 📖 명령줄 인터페이스 (CLI)

### 필수 인자

| 인자 | 설명 | 예시 |
|------|------|------|
| `-c, --company` | 회사 이름 | `"Tech Startup"` |
| `-d, --documents` | 문서명:경로 쌍 (공백 구분) | `IR1:data/ir1.pdf IR2:data/ir2.pdf` |

### 선택 인자

#### 추출 모델 (문서 분석용)
| 인자 | 설명 | 기본값 |
|------|------|--------|
| `-em, --extract-model` | AI 제공자 (`openai` 또는 `gemini`) | `openai` |
| `-emn, --extract-model-name` | 모델명 | `gpt-4o` (OpenAI)<br>`gemini-2.0-flash-exp` (Gemini) |

#### 보고서 모델 (보고서 생성용)
| 인자 | 설명 | 기본값 |
|------|------|--------|
| `-rm, --report-model` | AI 제공자 (`openai` 또는 `gemini`) | `openai` |
| `-rmn, --report-model-name` | 모델명 | `gpt-4o` (OpenAI)<br>`gemini-2.0-flash-exp` (Gemini) |

#### 기타 옵션
| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--ocr` | OCR API (`CLOVA` 또는 `Upstage`) | `CLOVA` |
| `--web` | 웹 검색 활성화 (Gemini만 지원) | `False` |
| `--max-rps` | 초당 최대 LLM API 요청 수 | `2.0` |
| `--debug` | 디버그 모드 (상세 로깅) | `False` |

---

## 💡 사용 예시

### 예시 1: OpenAI 기본 사용
```bash
python main.py \
  -c "Tech Startup" \
  -d pitch_deck:data/pitch.pdf \
  -em openai -emn gpt-4o \
  -rm openai -rmn gpt-4o \
  --ocr CLOVA \
  --max-rps 2.0
```

### 예시 2: Gemini 추출 + OpenAI 보고서
```bash
python main.py \
  -c "HealthTech Inc" \
  -d IR:data/ir.pdf \
  -em gemini -emn gemini-2.0-flash-exp \
  -rm openai -rmn gpt-4o \
  --ocr CLOVA \
  --max-rps 3.0
```

### 예시 3: 웹 검색 + 디버그
```bash
python main.py \
  -c "AI Company" \
  -d intro:data/intro.pdf \
  -em gemini -emn gemini-2.5-pro \
  -rm gemini -rmn gemini-2.5-pro \
  --ocr CLOVA \
  --web \
  --max-rps 2.0 \
  --debug
```

### 예시 4: 여러 문서 동시 분석
```bash
python main.py \
  -c "Enterprise Corp" \
  -d IR1:data/ir1.pdf IR2:data/ir2.pdf IR3:data/ir3.pdf \
  -em openai -emn gpt-4o \
  -rm openai -rmn gpt-4o \
  --ocr Upstage \
  --max-rps 2.0
```

---

## 📊 출력 구조

`src/results/` 폴더에 다음 형식으로 저장:

```
results/
└── 회사명_문서명_extract_모델_report_모델_OCR_옵션_rps값/
    ├── 문서명.json               # 페이지별 추출 결과
    ├── 문서명_ocr.json          # OCR 텍스트
    ├── 회사명_문서명.json        # 최종 보고서 (3종)
    └── debug.txt                # 디버그 로그 (--debug 시)
```

### 보고서 유형

1. **회사 현황 및 핵심역량** (`competencies`)
   - 재무현황 (매출, 영업이익, 누적투자)
   - 주요성과
   - 비즈니스 모델
   - 핵심역량 (B2G 키워드 포함)

2. **사업시장 현황** (`market`)
   - 시장분석 (성장률, 규모)
   - 연도별 시장규모
   - 경쟁구도 및 포지셔닝
   - 기술/정책 트렌드

3. **B2G 전략 방향** (`b2g_strategy`)
   - 약점분석
   - 추천전략
   - To-do 리스트

---

## 🛠️ LangChain 구성

### API 래퍼 (`src/api.py`)

- **ModelFactory**: OpenAI/Gemini ChatModel 생성
- **Dispatcher**: Rate limiting + 멀티모달 지원
- **ChatRequest**: 통일된 요청 인터페이스

### 프롬프트 템플릿 (`src/prompts.py`)

- **ChatPromptTemplate**: 시스템/유저 메시지 구조화
- 4가지 프롬프트: extraction, competencies, market, b2g_strategy

### 추출 Chain (`src/utils.py`)

- **extractJSON()**: 이미지 → OCR → LLM → JSON
- 비동기 처리, 자동 JSON 파싱

---

## 🔧 고급 기능

### Rate Limiting
```bash
--max-rps 1.0   # 느림, 저비용
--max-rps 2.0   # 균형 (권장)
--max-rps 5.0   # 빠름, 고비용
```

### 캐싱
- 한 번 분석된 문서는 자동 저장
- 동일 설정 재실행 시 캐시 재사용

### 디버그 모드
```bash
--debug  # 상세 로깅 + debug.txt 생성
```

---

## 📝 문제 해결

### API 키 오류
```
ValueError: API key not found
```
→ `.env` 파일의 API 키 확인

### Rate Limit 초과
```
openai.RateLimitError
```
→ `--max-rps` 값 낮추기 (예: `1.0`)

### JSON 파싱 오류
→ 모델 변경 또는 `--debug`로 원시 응답 확인

---

## 📚 참고 자료

- [LangChain 문서](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/)
- [CLOVA OCR](https://www.ncloud.com/product/aiService/ocr)
- [Upstage API](https://www.upstage.ai/)

---

## 📄 라이선스

별도 명시 없음.

---

## 👥 문의

이슈를 통해 문의해주세요.
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

