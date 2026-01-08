# main pipeline
from RA_Agent.src.api import (
    ChatRequest,
    Dispatcher,
)
import os
from RA_Agent.src.utils import extractJSON
import asyncio

class Document():
    """
    이 클래스는 하나의 문서를 나타냅니다.
    file_path: 문서(PDF) 파일 경로

    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pages = [] 
        self.analysis = []
    
    def convert_images(self):
        """
        PDF 문서를 페이지 단위로 나누어 이미지(PIL.Image 객체) 리스트로 변환
        """
        from pdf2image import convert_from_path
        self.pages = convert_from_path(self.file_path)
        return self.pages
    
    def get_pages(self):
        return self.pages
    
    async def analyze_doc(self, dispatcher: Dispatcher):
        """
        문서의 각 페이지 이미지를 분석하여 JSON 데이터 추출
        """
        async def analyze_pages():
            for i,page in enumerate(self.pages):
                json_data = await extractJSON(page, dispatcher)
                self.analysis.append({i: json_data})
        
        asyncio.run(analyze_pages())
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