#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/db_main.py의 SCHEMA_REGISTRY 키를 출력하는 유틸리티
"""
import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.db_main import SCHEMA_REGISTRY
    # 스페이스로 구분하여 출력
    print(" ".join(SCHEMA_REGISTRY.keys()))
except ImportError:
    # src 모듈을 찾지 못할 경우 fallback (현재는 에러 출력)
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.exit(1)
