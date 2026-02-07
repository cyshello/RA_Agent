#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/db_main.py의 SCHEMA_REGISTRY에서 json_path를 출력하는 유틸리티
사용법: python get_json_path.py [data_type]
"""
import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.db_main import SCHEMA_REGISTRY
    
    if len(sys.argv) < 2:
        sys.exit(1)
        
    data_type = sys.argv[1]
    
    if data_type in SCHEMA_REGISTRY:
        print(SCHEMA_REGISTRY[data_type].get("json_path", ""))
    else:
        sys.exit(1)

except ImportError:
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.exit(1)
