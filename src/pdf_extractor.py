"""
PDF 문서 구조화 추출 모듈

PDF 파일과 페이지 범위를 지정하면 extractJSON 함수를 이용해 
각 페이지를 분석하여 구조화된 JSON 데이터를 반환합니다.
"""

import asyncio
import json
import logging
import time
import os
from typing import Optional, Union
from pdf2image import convert_from_path
from PIL import Image

from src.api import Dispatcher, ModelFactory
from src.utils import extractJSON

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    PDF 문서에서 페이지별로 구조화된 JSON 데이터를 추출하는 클래스
    """
    
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        max_rps: float = 1.0
    ):
        """
        Args:
            model_provider: AI 모델 제공자 ("openai" 또는 "gemini")
            model_name: 사용할 구체적인 모델명
            max_rps: 초당 최대 요청 수 (기본값: 1 RPS)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.max_rps = max_rps
        self.dispatcher = Dispatcher(max_rps=max_rps)
        
        # 카운터
        self.ocr_call_count = 0
        self.llm_call_count = 0
    
    def _convert_pdf_to_images(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> list:
        """
        PDF를 이미지 리스트로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            start_page: 시작 페이지 (1부터 시작, None이면 처음부터)
            end_page: 끝 페이지 (포함, None이면 끝까지)
            
        Returns:
            PIL.Image 리스트
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # pdf2image는 first_page, last_page가 1부터 시작
        kwargs = {}
        if start_page is not None:
            kwargs['first_page'] = start_page
        if end_page is not None:
            kwargs['last_page'] = end_page
            
        logger.info(f"PDF 변환 중: {pdf_path} (페이지 {start_page or 1} ~ {end_page or '끝'})")
        images = convert_from_path(pdf_path, **kwargs)
        logger.info(f"총 {len(images)}개 페이지 이미지로 변환 완료")
        
        return images
    
    async def extract_pages(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        debug: bool = False
    ) -> dict:
        """
        PDF의 지정된 페이지 범위에서 구조화된 JSON 데이터 추출
        
        Args:
            pdf_path: PDF 파일 경로
            start_page: 시작 페이지 (1부터 시작, None이면 처음부터)
            end_page: 끝 페이지 (포함, None이면 끝까지)
            debug: 디버그 모드 여부
            
        Returns:
            dict: {
                "metadata": {
                    "pdf_path": str,
                    "start_page": int,
                    "end_page": int,
                    "total_pages": int,
                    "model": str,
                    "processing_time": float
                },
                "pages": [
                    {
                        "page_number": int,
                        "content": dict (추출된 JSON),
                        "ocr_text": str
                    },
                    ...
                ]
            }
        """
        start_time = time.time()
        
        # PDF를 이미지로 변환
        images = self._convert_pdf_to_images(pdf_path, start_page, end_page)
        
        # 실제 페이지 번호 계산 (1-based)
        actual_start_page = start_page or 1
        
        # Rate limiting 설정
        min_interval = 1.0 / self.max_rps
        last_request_time = [0]
        lock = asyncio.Lock()
        
        async def rate_limited_extract(image: Image.Image, page_idx: int):
            """Rate limiting이 적용된 추출 함수"""
            async with lock:
                elapsed = time.time() - last_request_time[0]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                last_request_time[0] = time.time()
            
            actual_page_num = actual_start_page + page_idx
            result, ocr_text = await extractJSON(
                image=image,
                dispatcher=self.dispatcher,
                page_num=actual_page_num,
                start_time_base=start_time,
                debug=debug,
                model_provider=self.model_provider,
                model_name=self.model_name,
                company=self  # API 카운터 증가용
            )
            
            return {
                "page_number": actual_page_num,
                "content": result,
                "ocr_text": ocr_text
            }
        
        # 모든 페이지 병렬 처리 (rate limiting 적용)
        logger.info(f"페이지 분석 시작 (총 {len(images)}개, RPS: {self.max_rps})")
        tasks = [
            asyncio.create_task(rate_limited_extract(img, idx))
            for idx, img in enumerate(images)
        ]
        results = await asyncio.gather(*tasks)
        
        # 결과 정렬 (페이지 번호 순)
        results = sorted(results, key=lambda x: x["page_number"])
        
        elapsed_time = time.time() - start_time
        
        # 결과 구성
        output = {
            "metadata": {
                "pdf_path": pdf_path,
                "start_page": actual_start_page,
                "end_page": actual_start_page + len(images) - 1,
                "total_pages": len(images),
                "model": f"{self.model_provider}/{self.model_name}",
                "processing_time": round(elapsed_time, 2),
                "ocr_calls": self.ocr_call_count,
                "llm_calls": self.llm_call_count
            },
            "pages": results
        }
        
        logger.info(f"추출 완료 - 총 {len(results)}개 페이지, 소요시간: {elapsed_time:.2f}초")
        
        return output
    
    def extract_pages_sync(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        debug: bool = False
    ) -> dict:
        """
        동기 버전의 extract_pages
        
        Args:
            pdf_path: PDF 파일 경로
            start_page: 시작 페이지 (1부터 시작, None이면 처음부터)
            end_page: 끝 페이지 (포함, None이면 끝까지)
            debug: 디버그 모드 여부
            
        Returns:
            구조화된 JSON dict
        """
        return asyncio.run(
            self.extract_pages(pdf_path, start_page, end_page, debug)
        )


def extract_pdf_to_json(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    max_rps: float = 1.0,
    output_path: Optional[str] = None,
    debug: bool = False
) -> dict:
    """
    PDF 파일에서 구조화된 JSON 데이터 추출 (편의 함수)
    
    Args:
        pdf_path: PDF 파일 경로
        start_page: 시작 페이지 (1부터 시작, None이면 처음부터)
        end_page: 끝 페이지 (포함, None이면 끝까지)
        model_provider: AI 모델 제공자 ("openai" 또는 "gemini")
        model_name: 사용할 구체적인 모델명
        max_rps: 초당 최대 요청 수
        output_path: 결과 저장 경로 (None이면 저장 안 함)
        debug: 디버그 모드 여부
        
    Returns:
        구조화된 JSON dict
        
    Example:
        >>> result = extract_pdf_to_json(
        ...     "data/document.pdf",
        ...     start_page=1,
        ...     end_page=5,
        ...     output_path="output/result.json"
        ... )
    """
    extractor = PDFExtractor(
        model_provider=model_provider,
        model_name=model_name,
        max_rps=max_rps
    )
    
    result = extractor.extract_pages_sync(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        debug=debug
    )
    
    # 결과 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"결과 저장 완료: {output_path}")
    
    return result


# CLI 지원
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PDF 문서에서 구조화된 JSON 데이터 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 PDF 추출
  python -m src.pdf_extractor -p data/document.pdf -o output/result.json
  
  # 특정 페이지 범위 추출
  python -m src.pdf_extractor -p data/document.pdf --start 1 --end 10
  
  # Gemini 모델 사용
  python -m src.pdf_extractor -p data/document.pdf --provider gemini --model gemini-2.0-pro-exp
        """
    )
    
    parser.add_argument(
        '-p', '--pdf',
        required=True,
        help='PDF 파일 경로'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='시작 페이지 (1부터 시작, 기본값: 처음부터)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='끝 페이지 (포함, 기본값: 끝까지)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='결과 JSON 저장 경로'
    )
    parser.add_argument(
        '--provider',
        default='openai',
        choices=['openai', 'gemini'],
        help='AI 모델 제공자 (기본값: openai)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='모델명 (기본값: gpt-4o)'
    )
    parser.add_argument(
        '--max-rps',
        type=float,
        default=1.0,
        help='초당 최대 요청 수 (기본값: 1.0)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    args = parser.parse_args()
    
    result = extract_pdf_to_json(
        pdf_path=args.pdf,
        start_page=args.start,
        end_page=args.end,
        model_provider=args.provider,
        model_name=args.model,
        max_rps=args.max_rps,
        output_path=args.output,
        debug=args.debug
    )
    
    # 출력 경로가 지정되지 않았으면 stdout으로 출력
    if not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
