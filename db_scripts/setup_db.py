#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB 설정 및 데이터 로드 스크립트 (Python 버전 load_to_db.sh)
"""
import argparse
import sys
import os
import subprocess
import pymysql
import json

# 상위 디렉토리를 path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.db_main import SCHEMA_REGISTRY

def main():
    schema_keys = list(SCHEMA_REGISTRY.keys())
    
    parser = argparse.ArgumentParser(description="DB 설정 및 데이터 로드")
    parser.add_argument("type", nargs="?", default="all", choices=schema_keys + ["all"], help="로드할 데이터 타입")
    parser.add_argument("--reset", "-r", action="store_true", help="기존 데이터 삭제 후 로드")
    parser.add_argument("--db-host", default="localhost", help="DB 호스트")
    parser.add_argument("--db-port", type=int, default=3306, help="DB 포트")
    parser.add_argument("--db-name", default="b2g_data", help="DB 이름")
    parser.add_argument("--db-user", default="root", help="DB 사용자")
    parser.add_argument("--db-password", default="", help="DB 비밀번호")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Python DB Setup & Load Script")
    print("=" * 60)
    print(f"  DB: {args.db_user}@{args.db_host}:{args.db_port}/{args.db_name}")
    print(f"  Type: {args.type}")
    print("=" * 60)
    
    # 1. Check/Create Database
    print("\n[Checking Database]")
    try:
        conn = pymysql.connect(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW DATABASES LIKE '{args.db_name}'")
            if not cursor.fetchone():
                print(f"  Creating database '{args.db_name}'...")
                cursor.execute(f"CREATE DATABASE {args.db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                print("  Database created successfully.")
            else:
                print(f"  Database '{args.db_name}' already exists.")
        conn.close()
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        sys.exit(1)
        
    # 2. Check JSON Files
    print("\n[Checking JSON Files]")
    targets = schema_keys if args.type == "all" else [args.type]
    for key in targets:
        json_path = SCHEMA_REGISTRY[key].get("json_path")
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"  ✅ {key}: {len(data)} items ({json_path})")
            except Exception as e:
                print(f"  ⚠️  {key}: Error reading file ({e})")
        else:
            print(f"  ⚠️  {key}: File not found ({json_path})")
            
    # 3. Run Load Script
    print("\n[Running Load Script]")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, "load_json_to_db.py"),
        "--type", args.type,
        "--db-host", args.db_host,
        "--db-port", str(args.db_port),
        "--db-name", args.db_name,
        "--db-user", args.db_user,
        "--db-password", args.db_password
    ]
    if args.reset:
        cmd.append("--reset")
        
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Setup and Load Complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Load script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()