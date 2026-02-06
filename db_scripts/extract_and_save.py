#!/usr/bin/env python3
"""
OCR 결과에서 과제 데이터 추출 및 DB 저장 스크립트
"""

import os
import sys
import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# .env 파일 로드
env_path = project_root / "src" / ".env"
load_dotenv(env_path)

# 환경변수 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY", "")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
OUTPUT_DIR = "/Users/youngseocho/Desktop/AX/RA_Agent/data/output"

# DB 설정
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "b2g_data",
    "user": "youngseocho",
    "password": "",
}
COLLECTION_NAME = "b2g_projects"


def load_ocr_results() -> List[Dict]:
    """모든 OCR 결과 파일 로드"""
    results = []
    for i in range(1, 196):
        ocr_file = os.path.join(OUTPUT_DIR, f"ocr_page_{i}.json")
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
    
    # 페이지 번호로 정렬
    results.sort(key=lambda x: x['page_num'])
    return results


def detect_project_pages(ocr_results: List[Dict]) -> List[Dict]:
    """
    과제 페이지 감지
    
    과제 페이지 특징:
    1. 첫 부분에 숫자 (과제 번호)
    2. "과제목표" 또는 "과제 목표" 키워드 포함
    3. "주요내용" 키워드 포함
    """
    project_pages = []
    
    for ocr in ocr_results:
        text = ocr.get('text', '')
        if not text:
            continue
        
        # 텍스트 정리 (줄바꿈을 공백으로)
        text_clean = text.replace('\n', ' ')
        
        # 과제 페이지 여부 판단
        has_project_goal = '과제목표' in text_clean or '과제 목표' in text_clean
        has_main_content = '주요내용' in text_clean
        has_expected_effect = '기대효과' in text_clean
        
        # 과제 번호 패턴 찾기 (페이지 시작 부분에 숫자)
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ''
        
        # 과제 번호 패턴: 1-3자리 숫자
        project_num_match = re.match(r'^(\d{1,3})$', first_line)
        
        if has_project_goal and has_main_content and project_num_match:
            project_num = project_num_match.group(1)
            project_pages.append({
                'page_num': ocr['page_num'],
                'project_num': project_num,
                'text': text,
                'has_expected_effect': has_expected_effect
            })
    
    return project_pages


def extract_project_data(page_data: Dict) -> Optional[Dict]:
    """
    단일 페이지에서 과제 데이터 추출
    """
    text = page_data['text']
    project_num = page_data['project_num']
    
    # 텍스트를 줄 단위로 분리
    lines = text.split('\n')
    
    # 과제명 추출 (과제 번호 다음 줄부터 "과제목표" 전까지)
    project_name_lines = []
    start_collecting = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if line == project_num:
            start_collecting = True
            continue
        
        if start_collecting:
            if '과제목표' in line or '과제 목표' in line:
                break
            project_name_lines.append(line)
    
    project_name = ' '.join(project_name_lines).strip()
    
    # 섹션 분리
    text_joined = text.replace('\n', ' ')
    
    # 과제 목표 추출
    objectives = []
    goal_match = re.search(r'과제\s*목표\s*(.+?)(?:주요\s*내용|$)', text_joined, re.DOTALL)
    if goal_match:
        goal_text = goal_match.group(1)
        # "○" 또는 "O"로 시작하는 항목 분리
        items = re.split(r'[○O]\s*', goal_text)
        objectives = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
    
    # 주요내용 추출
    main_contents = []
    content_match = re.search(r'주요\s*내용\s*(.+?)(?:기대\s*효과|$)', text_joined, re.DOTALL)
    if content_match:
        content_text = content_match.group(1)
        items = re.split(r'[○O]\s*', content_text)
        main_contents = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
    
    # 기대효과 추출
    effects = []
    effect_match = re.search(r'기대\s*효과\s*(.+?)(?:-\s*\d+\s*-|$)', text_joined, re.DOTALL)
    if effect_match:
        effect_text = effect_match.group(1)
        items = re.split(r'[○O]\s*', effect_text)
        effects = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
    
    # 검증
    if not project_name:
        return None
    
    return {
        "과제명": project_name,
        "과제번호": project_num,
        "과제 목표": objectives,
        "주요내용": main_contents,
        "기대효과": effects,
        "source_page": page_data['page_num']
    }


async def save_to_db(projects: List[Dict]):
    """
    추출된 과제 데이터를 DB에 저장
    """
    from src.db import B2GVectorStore, StructuredProject
    
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    vector_store = B2GVectorStore(
        connection_string=connection_string,
        collection_name=COLLECTION_NAME
    )
    
    structured_projects = []
    for p in projects:
        sp = StructuredProject(
            과제명=p["과제명"],
            과제번호=str(p["과제번호"]),
            과제_목표=p.get("과제 목표", []),
            주요내용=p.get("주요내용", []),
            기대효과=p.get("기대효과", []),
            source_document="presidential_agenda.pdf",
            page_range=str(p.get("source_page", ""))
        )
        structured_projects.append(sp)
    
    # DB에 저장
    if structured_projects:
        ids = vector_store.add_structured_projects(structured_projects)
        logger.info(f"✅ {len(ids)}개 항목이 DB에 저장되었습니다.")
    
    return structured_projects


def main():
    print("="*60)
    print("🚀 OCR 결과에서 과제 데이터 추출")
    print("="*60)
    
    # 1. OCR 결과 로드
    print("\n📌 Step 1: OCR 결과 로드")
    ocr_results = load_ocr_results()
    print(f"  로드된 페이지: {len(ocr_results)}개")
    
    # 2. 과제 페이지 감지
    print("\n📌 Step 2: 과제 페이지 감지")
    project_pages = detect_project_pages(ocr_results)
    print(f"  감지된 과제 페이지: {len(project_pages)}개")
    
    # 3. 과제 데이터 추출
    print("\n📌 Step 3: 과제 데이터 추출")
    projects = []
    for page_data in project_pages:
        project = extract_project_data(page_data)
        if project:
            projects.append(project)
            # 진행 상황 출력
            if len(projects) % 10 == 0:
                print(f"  추출 진행: {len(projects)}개")
    
    print(f"  총 추출된 과제: {len(projects)}개")
    
    # 4. 결과 저장 (JSON)
    print("\n📌 Step 4: 결과 저장")
    projects_file = os.path.join(OUTPUT_DIR, "extracted_projects.json")
    with open(projects_file, 'w', encoding='utf-8') as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {projects_file}")
    
    # 5. DB 저장
    print("\n📌 Step 5: DB 저장")
    asyncio.run(save_to_db(projects))
    
    # 6. 결과 요약
    print("\n" + "="*60)
    print("📊 추출 결과 요약")
    print("="*60)
    
    for i, p in enumerate(projects[:10], 1):
        print(f"\n{i}. [{p['과제번호']}] {p['과제명'][:50]}...")
        print(f"   - 과제 목표: {len(p.get('과제 목표', []))}개")
        print(f"   - 주요내용: {len(p.get('주요내용', []))}개")
        print(f"   - 기대효과: {len(p.get('기대효과', []))}개")
    
    if len(projects) > 10:
        print(f"\n... 외 {len(projects) - 10}개 과제")
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
