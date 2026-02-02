#!/usr/bin/env python3
"""
동반성장 평가지표 모듈 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Import 테스트"""
    print("1. Import 테스트...")
    
    from src.db import (
        StructuredIndicator,
        InclusiveGrowthVectorStore,
        InclusiveGrowthPipeline,
        create_inclusive_growth_pipeline,
        create_inclusive_growth_vector_store
    )
    print("   ✅ 모든 클래스/함수 import 성공!")
    return True


def test_structured_indicator():
    """StructuredIndicator 테스트"""
    print("\n2. StructuredIndicator 테스트...")
    
    from src.db import StructuredIndicator
    
    # 객체 생성
    indicator = StructuredIndicator(
        지표명='중소기업 협력 이익공유',
        평가기준=['협력 중소기업과의 이익공유 여부', '이익공유 규모'],
        평가방법=['관련 계약서 및 증빙 자료 확인'],
        참고사항=['이익공유 범위는 직전 사업연도 기준'],
        증빙자료=['협력 계약서', '이익공유 내역서']
    )
    
    print(f"   ✅ 객체 생성: {indicator.지표명}")
    
    # to_dict 테스트
    data = indicator.to_dict()
    print(f"   ✅ to_dict(): {len(data)} 필드")
    
    # to_embedding_items 테스트
    items = indicator.to_embedding_items()
    print(f"   ✅ to_embedding_items(): {len(items)}개 항목")
    
    for item in items:
        print(f"      - [{item['field_type']}] {item['text'][:30]}...")
    
    # from_dict 테스트
    indicator2 = StructuredIndicator.from_dict({
        "지표명": "테스트 지표",
        "평가기준": ["기준1"],
        "평가방법": ["방법1"],
        "참고사항": [],
        "증빙사료": ["자료1"]  # 오타 테스트 (증빙사료 -> 증빙자료)
    })
    print(f"   ✅ from_dict(): {indicator2.지표명}")
    print(f"      증빙자료: {indicator2.증빙자료}")
    
    return True


def test_vector_store_connection():
    """벡터 저장소 연결 테스트"""
    print("\n3. VectorStore 연결 테스트...")
    
    from src.db import InclusiveGrowthVectorStore
    
    connection_string = "postgresql://youngseocho:@localhost:5432/b2g_data"
    
    try:
        store = InclusiveGrowthVectorStore(
            connection_string=connection_string,
            collection_name="inclusive_growth_indicators"
        )
        print(f"   ✅ VectorStore 초기화 성공")
        return True
    except Exception as e:
        print(f"   ⚠️ VectorStore 연결 실패: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("동반성장 평가지표 모듈 테스트")
    print("="*60)
    
    success = True
    
    # 1. Import 테스트
    success = test_imports() and success
    
    # 2. StructuredIndicator 테스트
    success = test_structured_indicator() and success
    
    # 3. VectorStore 연결 테스트
    success = test_vector_store_connection() and success
    
    print("\n" + "="*60)
    if success:
        print("✅ 모든 테스트 통과!")
    else:
        print("❌ 일부 테스트 실패")
    print("="*60)
