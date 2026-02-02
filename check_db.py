#!/usr/bin/env python
"""DB 현황 확인 스크립트"""
import psycopg2

conn = psycopg2.connect('postgresql://youngseocho:@localhost:5432/b2g_data')
cur = conn.cursor()

# 컬렉션 목록
cur.execute("SELECT uuid, name FROM langchain_pg_collection")
print('='*60)
print('컬렉션 목록:')
print('='*60)
collections = cur.fetchall()
for uuid, name in collections:
    print(f'  - {name}')

# 각 컬렉션별 데이터 수와 샘플
for col_uuid, col_name in collections:
    print()
    print('='*60)
    print(f'{col_name}')
    print('='*60)
    
    # 총 개수
    cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s", (col_uuid,))
    count = cur.fetchone()[0]
    print(f'총 {count}개 row')
    
    # 샘플 2개
    cur.execute("""
        SELECT document, cmetadata 
        FROM langchain_pg_embedding 
        WHERE collection_id = %s 
        LIMIT 2
    """, (col_uuid,))
    
    print('\n샘플 데이터:')
    for i, (doc, meta) in enumerate(cur.fetchall(), 1):
        doc_preview = doc[:80] + '...' if len(doc) > 80 else doc
        print(f'\n  [{i}] document: {doc_preview}')
        if meta:
            if '과제명' in meta:
                print(f'      과제명: {meta.get("과제명", "")[:40]}')
                print(f'      과제번호: {meta.get("과제번호", "")}')
                print(f'      field_type: {meta.get("field_type", "")}')
            elif '지표명' in meta:
                print(f'      지표명: {meta.get("지표명", "")[:40]}')
                print(f'      field_type: {meta.get("field_type", "")}')

conn.close()
