#!/usr/bin/env python
"""
B2G 평가 검색 및 분석 테스트 스크립트
(국정과제, 경영평가, 동반성장 통합 테스트)
"""

import asyncio
import json
from src.db import B2GVectorStore, InclusiveGrowthVectorStore
from src.api import ModelFactory
from src.prompts import COMPANY_FEATURE_EXTRACTION_PROMPT, B2G_EVALUATION_ANALYSIS_PROMPT
from langchain_core.output_parsers import JsonOutputParser


async def test_b2g_evaluations():
    # 테스트용 문서 데이터 로드
    with open('src/results/Nudge_Healthcare_IR1_extract_gemini-gemini-2.5-flash_report_openai-gpt-4o_CLOVA_debug_rps2_0/IR1.json', 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    
    document_data = json.dumps(doc_data[:15], ensure_ascii=False)
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
    
    connection_string = 'postgresql://youngseocho:@localhost:5432/b2g_data'
    
    # 평가 유형별 설정
    eval_configs = {
        "presidential_agenda": {
            "name": "국정과제",
            "collection": "b2g_projects",
            "store_class": B2GVectorStore,
            "search_method": "search_unique_projects",
            "name_field": "과제명",
            "id_field": "과제번호",
        },
        "management_eval": {
            "name": "공공기관 경영평가",
            "collection": "management_eval_indicators",
            "store_class": InclusiveGrowthVectorStore,
            "search_method": "search_unique_indicators",
            "name_field": "지표명",
            "id_field": "지표명",
        },
        "inclusive_growth": {
            "name": "동반성장 평가지표",
            "collection": "inclusive_growth_indicators",
            "store_class": InclusiveGrowthVectorStore,
            "search_method": "search_unique_indicators",
            "name_field": "지표명",
            "id_field": "지표명",
        }
    }
    
    search_queries = company_features.get('search_queries', [])
    core_technologies = company_features.get('core_technologies', [])
    
    results = {}
    
    # 2. 각 평가 유형별로 검색 및 분석
    for eval_type, config in eval_configs.items():
        print(f"\n{'='*60}")
        print(f"Step 2-{eval_type}: {config['name']} 검색 중...")
        print("="*60)
        
        # 벡터 스토어 생성
        try:
            vector_store = config["store_class"](
                connection_string=connection_string,
                collection_name=config["collection"]
            )
        except Exception as e:
            print(f"  벡터 스토어 연결 실패: {e}")
            continue
        
        search_func = getattr(vector_store, config["search_method"])
        
        # 검색
        all_items = {}
        
        for query in search_queries:
            try:
                search_results = search_func(query, k=5)
                for item in search_results:
                    item_id = item.get(config["id_field"], "") or item.get("full_data", {}).get(config["id_field"], "")
                    if item_id and item_id not in all_items:
                        all_items[item_id] = item
                        name = item.get(config["name_field"], "") or item.get("full_data", {}).get(config["name_field"], "")
                        print(f"  + {name[:50]}...")
            except Exception as e:
                print(f"  검색 오류: {e}")
                continue
        
        # 추가 검색
        if len(all_items) < 10:
            for tech in core_technologies:
                if len(all_items) >= 10:
                    break
                try:
                    search_results = search_func(tech, k=3)
                    for item in search_results:
                        item_id = item.get(config["id_field"], "") or item.get("full_data", {}).get(config["id_field"], "")
                        if item_id and item_id not in all_items:
                            all_items[item_id] = item
                except:
                    continue
        
        items_list = list(all_items.values())[:10]
        print(f"\n총 {len(items_list)}개 항목 검색됨")
        
        if not items_list:
            print(f"  {config['name']} 검색 결과 없음 - 스킵")
            continue
        
        # 3. 항목 리스트를 텍스트로 변환
        items_text = ""
        for i, item in enumerate(items_list, 1):
            name = item.get(config["name_field"], "") or item.get("full_data", {}).get(config["name_field"], "")
            matched_text = item.get("matched_text", "")[:200]
            
            if eval_type == "presidential_agenda":
                goals = item.get("과제 목표", [])[:2]
                contents = item.get("주요내용", [])[:2]
                items_text += f"""
{i}. [{item.get(config["id_field"], "")}] {name}
   - 과제 목표: {', '.join(goals) if goals else '정보 없음'}
   - 주요내용: {', '.join(contents) if contents else matched_text}
"""
            else:
                full_data = item.get("full_data", {})
                eval_criteria = full_data.get("평가기준", [])[:2]
                eval_method = full_data.get("평가방법", [])[:1]
                items_text += f"""
{i}. {name}
   - 평가기준: {', '.join(eval_criteria) if eval_criteria else matched_text}
   - 평가방법: {', '.join(eval_method) if eval_method else ''}
"""
        
        # 4. LLM 분석
        print(f"\nStep 3-{eval_type}: {config['name']} 분석 중...")
        
        analysis_model = ModelFactory.create_model_chain(
            provider='openai',
            model_name='gpt-4o',
            output_format='json',
            max_rps=2.0
        )
        
        analysis_chain = (
            B2G_EVALUATION_ANALYSIS_PROMPT
            | analysis_model
            | JsonOutputParser()
        )
        
        try:
            analysis_result = await analysis_chain.ainvoke({
                "company_name": company_name,
                "company_summary": company_features.get("company_summary", ""),
                "core_technologies": ", ".join(company_features.get("core_technologies", [])),
                "target_sectors": ", ".join(company_features.get("target_sectors", [])),
                "eval_type_name": config["name"],
                "items_list": items_text
            })
            
            results[eval_type] = {
                "top10": analysis_result.get("top10", []),
                "analysis": analysis_result.get("analysis", {})
            }
            
            # 결과 출력
            print(f"\n{config['name']} 분석 결과:")
            print("-" * 40)
            
            top10 = analysis_result.get("top10", [])
            for item in top10[:5]:
                print(f"  {item.get('rank', '-')}. {item.get('name', '')[:40]}...")
            
            analysis = analysis_result.get("analysis", {})
            insight = analysis.get("insight", {})
            risk = analysis.get("risk", {})
            consider = analysis.get("consider", [])
            
            print(f"\n  [Insight] {insight.get('title', '')}")
            for d in insight.get('details', [])[:2]:
                print(f"    - {d[:60]}...")
            
            print(f"\n  [Risk] {risk.get('title', '')}")
            for d in risk.get('details', [])[:2]:
                print(f"    - {d[:60]}...")
            
            print(f"\n  [Consider]")
            for q in consider[:3]:
                print(f"    - {q[:60]}...")
                
        except Exception as e:
            print(f"  분석 오류: {e}")
            results[eval_type] = {"top10": [], "analysis": {}}
    
    # 5. 최종 결과 저장
    print("\n" + "="*60)
    print("Step 5: 최종 결과 저장")
    print("="*60)
    
    with open('test_b2g_evaluations_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("결과가 test_b2g_evaluations_result.json에 저장되었습니다.")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_b2g_evaluations())
