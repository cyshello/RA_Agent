# main pipeline
from src.api import (
    ChatRequest,
    Dispatcher,
)
import os
import json
import logging
import time
from src.utils import extractJSON
import asyncio

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class Document():
    """
    이 클래스는 하나의 문서를 나타냅니다.
    file_path: 문서(PDF) 파일 경로

    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pages = [] 
        self.analysis = []
        self.ocr_texts = []
    
    def convert_images(self):
        """
        PDF 문서를 페이지 단위로 나누어 이미지(PIL.Image 객체) 리스트로 변환
        """
        from pdf2image import convert_from_path
        self.pages = convert_from_path(self.file_path)
        return self.pages
    
    def get_pages(self):
        return self.pages
    
    async def analyze_doc(self, dispatcher: Dispatcher, start_time: float = None):
        """
        문서의 각 페이지 이미지를 분석하여 JSON 데이터 추출 (병렬 처리)
        """
        # 모든 페이지를 동시에 분석 (명시적으로 태스크 생성)
        tasks = [asyncio.create_task(extractJSON(page, dispatcher, i, start_time)) 
                 for i, page in enumerate(self.pages)]
        results = await asyncio.gather(*tasks)
        
        # 결과를 페이지 번호와 함께 저장
        self.analysis = [{i: result[0]} for i, result in enumerate(results)]
        self.ocr_texts = [{i:ocr_text} for i, (_, ocr_text) in enumerate(results)]
        
        return self.analysis
    

class Company():
    """
    여러 문서를 넣어서 요청한 회사 단위
    """
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.dispatcher = Dispatcher()
    
    def add_document(self, document: Document):
        self.documents.append(document)

    async def process_documents(self):
        for document in self.documents:
            document.convert_images()
            await document.analyze_doc(self.dispatcher)
        return [doc.analysis for doc in self.documents]

# TODO : 회사 단위 보고서 생성 구현

def main():
    doc1 = Document("data/IR1.pdf")

    doc1.convert_images()
    dispatcher = Dispatcher()
    
    # 타이머 시작
    start_time = time.time()
    logger.info(f"분석 시작 - 총 {len(doc1.pages)}페이지")
    
    asyncio.run(doc1.analyze_doc(dispatcher, start_time))
    
    # 전체 소요 시간
    total_time = time.time() - start_time
    logger.info(f"전체 분석 완료 - 소요 시간: {total_time:.2f}초")
    
    # print("result :", doc1.analysis)
    
    # results 디렉토리 생성 (없으면)
    results_dir = "src/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # PDF 파일 이름 추출 (확장자 제외)
    pdf_name = os.path.splitext(os.path.basename(doc1.file_path))[0]
    
    output_path = os.path.join(results_dir, f"{pdf_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(doc1.analysis, f, ensure_ascii=False, indent=2)
    
    ocr_path = os.path.join(results_dir, f"{pdf_name}_ocr.json")
    with open(ocr_path, 'w', encoding='utf-8') as f:
        json.dump(doc1.ocr_texts, f, ensure_ascii=False, indent=2)
    print(f"Analysis saved to: {output_path}")

main()
