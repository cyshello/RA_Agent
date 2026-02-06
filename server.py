"""
FastAPI 서버 - 기업 분석 및 보고서 생성 API

S3에서 PDF를 다운로드하여 분석 파이프라인을 실행하고 결과를 반환합니다.
"""

import os
import tempfile
import asyncio
from typing import Optional
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 모듈 import
from main import Company, Document
from src.utils import OUTPUT_JSON_SCHEMA

# FastAPI 앱 생성
app = FastAPI(
    title="기업 분석 API",
    description="S3에서 PDF를 다운로드하여 기업 분석 및 보고서를 생성하는 API",
    version="1.0.0"
)


# 요청/응답 모델 정의
class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    region: str = Field(..., description="AWS 리전", json_schema_extra={"example": "ap-northeast-2"})
    bucket: str = Field(..., description="S3 버킷 이름", json_schema_extra={"example": "example-bucket"})
    object_key: list[str] = Field(..., description="S3 오브젝트 키 리스트", json_schema_extra={"example": ["reports/2024/final.pdf"]})
    company_name: Optional[str] = Field(default="분석대상기업", description="회사 이름")
    web_search: Optional[bool] = Field(default=False, description="웹 검색 활성화 여부")
    max_rps: Optional[float] = Field(default=2.0, description="초당 최대 API 요청 수")
    debug: Optional[bool] = Field(default=False, description="디버그 모드")


class AnalysisResponse(BaseModel):
    """분석 응답 모델 - OUTPUT_JSON_SCHEMA 구조"""
    section1: dict
    section2: dict
    section3: dict
    section4: dict
    section5: dict


def get_s3_client(region: str):
    """
    S3 클라이언트 생성
    
    .env에서 AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 읽어옵니다.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key_id or not aws_secret_access_key:
        raise HTTPException(
            status_code=500,
            detail="AWS 자격 증명이 설정되지 않았습니다. .env 파일을 확인하세요."
        )
    
    return boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


def download_from_s3(s3_client, bucket: str, object_key: str, local_path: str) -> str:
    """
    S3에서 파일 다운로드
    
    Args:
        s3_client: boto3 S3 클라이언트
        bucket: S3 버킷 이름
        object_key: S3 오브젝트 키
        local_path: 로컬 저장 경로
        
    Returns:
        다운로드된 파일의 로컬 경로
    """
    try:
        s3_client.download_file(bucket, object_key, local_path)
        return local_path
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise HTTPException(
                status_code=404,
                detail=f"S3 오브젝트를 찾을 수 없습니다: {bucket}/{object_key}"
            )
        elif error_code == '403':
            raise HTTPException(
                status_code=403,
                detail=f"S3 오브젝트에 접근 권한이 없습니다: {bucket}/{object_key}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"S3 다운로드 중 오류 발생: {str(e)}"
            )


async def run_analysis_pipeline(
    company_name: str,
    documents: list[tuple[str, str]],
    web_search: bool = False,
    max_rps: float = 2.0,
    debug: bool = False
) -> dict:
    """
    분석 파이프라인 실행
    
    Args:
        company_name: 회사 이름
        documents: 문서 이름과 경로 리스트 [(name1, path1), ...]
        web_search: 웹 검색 활성화 여부
        max_rps: 초당 최대 요청 수
        debug: 디버그 모드
        
    Returns:
        분석 결과 JSON (OUTPUT_JSON_SCHEMA 형식)
    """
    # Company 객체 생성
    com = Company(company_name, max_rps=max_rps)
    
    # 모든 문서 추가
    for name, doc_path in documents:
        com.add_document(name, Document(doc_path))
    
    # 결과 저장 폴더 설정
    result_dir = com.setup_result_directory(
        web_search=web_search,
        max_rps=max_rps,
        debug=debug
    )
    
    # 문서 처리 (OCR 및 분석)
    await com.process_documents(
        debug=debug,
        max_rps=max_rps,
        web_search=web_search
    )
    
    # Section 1~3 보고서 생성
    await com.generate_all_reports(
        web=web_search,
        debug=debug
    )
    
    # Section 4: B2G 평가 검색 및 분석
    await com.search_b2g_evaluations(debug=debug)
    
    # Section 5: B2G 전략 방향 수립
    await com.generate_b2g_strategy(
        web=web_search,
        debug=debug
    )
    
    # 결과 반환
    return com.result_json


@app.post(
    "/analysis",
    response_model=AnalysisResponse,
    summary="기업 분석 실행",
    description="S3에서 PDF를 다운로드하여 기업 분석을 실행하고 결과를 반환합니다."
)
async def analyze(request: AnalysisRequest):
    """
    기업 분석 API 엔드포인트
    
    S3에서 PDF 파일들을 다운로드하여 분석 파이프라인을 실행하고,
    분석 결과를 JSON으로 반환합니다.
    """
    # S3 클라이언트 생성
    s3_client = get_s3_client(request.region)
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        
        # S3에서 PDF 파일들 다운로드
        for i, object_key in enumerate(request.object_key):
            # 파일 이름 추출 (경로에서 마지막 부분)
            file_name = Path(object_key).stem  # 확장자 제외한 파일명
            local_path = os.path.join(temp_dir, f"{file_name}.pdf")
            
            # S3에서 다운로드
            download_from_s3(s3_client, request.bucket, object_key, local_path)
            
            # 문서 리스트에 추가 (이름, 경로)
            doc_name = f"doc{i+1}_{file_name}"
            documents.append((doc_name, local_path))
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="분석할 PDF 파일이 없습니다."
            )
        
        try:
            # 분석 파이프라인 실행
            result = await run_analysis_pipeline(
                company_name=request.company_name,
                documents=documents,
                web_search=request.web_search,
                max_rps=request.max_rps,
                debug=request.debug
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"분석 중 오류 발생: {str(e)}"
            )


@app.get("/schema", summary="출력 스키마 조회")
async def get_output_schema():
    """분석 결과의 JSON 스키마를 반환합니다."""
    return JSONResponse(content=OUTPUT_JSON_SCHEMA)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
