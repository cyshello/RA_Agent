#!/usr/bin/env python
"""
국정과제 검색 및 랭킹 테스트 스크립트
"""

import asyncio
import json
import re
from src.db import B2GVectorStore
from src.api import ModelFactory
from src.prompts import COMPANY_FEATURE_EXTRACTION_PROMPT, PRESIDENTIAL_AGENDA_RANKING_PROMPT
from langchain_core.output_parsers import JsonOutputParser


async def test_presidential_agenda():
    # 테스트용 문서 데이터 로드
    with open('src/results/Nudge_Healthcare_IR1_extract_gemini-gemini-2.5-flash_report_openai-gpt-4o_CLOVA_debug_rps2_0/IR1.json', 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    
    document_data = json.dumps(doc_data[:15], ensure_ascii=False)  # 처음 15페이지만 사용
    company_name = "Nudge Healthcare (넛지헬스케어)"
    
    # 1. 기업 특징 추출
    print("="*60)
    print("Step 1: 기업 특징 추출 중...")
    print("="*60)
    
    feature_model = ModelFactory.create_model_chain(
        provider='openai',
        model_name='gpt-4o-mini',
        output_format='json',
        max_rps=2.0
    )
    
    feature_chain = (
        COMPANY_FEATURE_EXTRACTION_PROMPT
        | feature_model
        | JsonOutputParser()
    )
    
    company_features = await feature_chain.ainvoke({
        'company_name': company_name,
        'document_data': document_data
    })
    
    print(f"추출된 특징:")
    print(json.dumps(company_features, ensure_ascii=False, indent=2))
    
    # 2. 국정과제 검색
    print("\n" + "="*60)
    print("Step 2: 국정과제 검색 중...")
    print("="*60)
    
    connection_string = 'postgresql://youngseocho:@localhost:5432/b2g_data'
    vs = B2GVectorStore(connection_string=connection_string, collection_name='b2g_projects')
    
    search_queries = company_features.get('search_queries', [])
    all_agendas = {}
    
    for query in search_queries:
        print(f"\n검색 쿼리: {query}")
        results = vs.search_unique_projects(query, k=5)
        for agenda in results:
            agenda_num = agenda.get('과제번호', '')
            if agenda_num and agenda_num not in all_agendas:
                all_agendas[agenda_num] = agenda
                print(f"  + [{agenda_num}] {agenda.get('과제명', '')[:50]}")
    
    # 부족하면 기술 키워드로 추가 검색
    agenda_list = list(all_agendas.values())
    if len(agenda_list) < 10:
        for tech in company_features.get('core_technologies', []):
            if len(list(all_agendas.values())) >= 10:
                break
            print(f"\n추가 검색 (기술): {tech}")
            results = vs.search_unique_projects(tech, k=3)
            for agenda in results:
                agenda_num = agenda.get('과제번호', '')
                if agenda_num and agenda_num not in all_agendas:
                    all_agendas[agenda_num] = agenda
                    print(f"  + [{agenda_num}] {agenda.get('과제명', '')[:50]}")
    
    agenda_list = list(all_agendas.values())[:10]
    print(f"\n총 {len(agenda_list)}개 국정과제 검색됨")
    
    # 3. 국정과제 랭킹 (OpenAI o1 모델)
    print("\n" + "="*60)
    print("Step 3: 국정과제 랭킹 중 (OpenAI o1 모델 사용)...")
    print("="*60)
    
    # 국정과제 리스트를 텍스트로 변환
    agenda_text = ""
    for i, agenda in enumerate(agenda_list, 1):
        agenda_text += f"""
{i}. [{agenda.get('과제번호', '')}] {agenda.get('과제명', '')}
   - 과제 목표: {', '.join(agenda.get('과제 목표', [])[:2])}
   - 주요내용: {', '.join(agenda.get('주요내용', [])[:2])}
   - 기대효과: {', '.join(agenda.get('기대효과', [])[:2])}
"""
    
    # OpenAI o1 모델로 랭킹
    ranking_model = ModelFactory.create_model_chain(
        provider='openai',
        model_name='o1',  # 최신 추론 모델
        output_format='text',  # o1은 JSON 모드 미지원
        max_rps=2.0
    )
    
    ranking_chain = (
        PRESIDENTIAL_AGENDA_RANKING_PROMPT
        | ranking_model
    )
    
    print("o1 모델 호출 중... (시간이 걸릴 수 있습니다)")
    
    ranking_response = await ranking_chain.ainvoke({
        'company_name': company_name,
        'document_data': document_data,
        'company_features': json.dumps(company_features, ensure_ascii=False),
        'agenda_list': agenda_text
    })
    
    # o1 응답에서 JSON 추출
    response_text = ranking_response.content if hasattr(ranking_response, 'content') else str(ranking_response)
    
    # JSON 블록 추출
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            print(f"JSON 파싱 실패: {response_text[:500]}")
            json_str = '{"presidential_agenda": []}'
    
    try:
        ranking_result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
        ranking_result = {"presidential_agenda": []}
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("Step 4: 최종 결과")
    print("="*60)
    
    presidential_agenda = ranking_result.get("presidential_agenda", [])
    
    formatted_agenda = []
    for item in presidential_agenda[:10]:
        formatted_agenda.append({
            "rank": item.get("rank", 0),
            "name": item.get("name", ""),
            "description": item.get("description", "")
        })
    
    print(f"\n{company_name}에 적합한 국정과제 Top 10:")
    print("-"*60)
    for item in formatted_agenda:
        print(f"\n{item['rank']}. {item['name']}")
        print(f"   사유: {item['description'][:100]}...")
    
    # JSON 파일로 저장
    output = {
        "company_name": company_name,
        "company_features": company_features,
        "presidential_agenda": formatted_agenda
    }
    
    with open('test_presidential_agenda_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 test_presidential_agenda_result.json에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(test_presidential_agenda())
