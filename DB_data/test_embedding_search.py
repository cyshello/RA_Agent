#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""임베딩 기반 검색 테스트"""

import sys
sys.path.insert(0, '/Users/youngseocho/Desktop/AX/RA_Agent')

from src.db_main import MySQLStore

store = MySQLStore()

# 국정과제 검색 테스트
print('=' * 60)
print('🔍 국정과제 임베딩 검색: "인공지능 AI 기술"')
print('=' * 60)
results = store.search_projects_by_embedding('인공지능 AI 기술', k=5)
for i, r in enumerate(results, 1):
    print(f'{i}. [{r["score"]:.4f}] {r["과제명"]}')

print()

# 경영평가 검색 테스트
print('=' * 60)
print('🔍 경영평가 임베딩 검색: "리더십 전략"')
print('=' * 60)
results = store.search_management_evals_by_embedding('리더십 전략', k=5)
for i, r in enumerate(results, 1):
    print(f'{i}. [{r["score"]:.4f}] {r["지표명"]}')

print()

# 동반성장 검색 테스트
print('=' * 60)
print('🔍 동반성장 임베딩 검색: "상생협력 기업"')
print('=' * 60)
results = store.search_inclusive_growth_by_embedding('상생협력 기업', k=5)
for i, r in enumerate(results, 1):
    print(f'{i}. [{r["score"]:.4f}] {r["지표명"]}')

print()
print('✅ 임베딩 기반 검색 테스트 완료!')
