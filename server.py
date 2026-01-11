from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import os
import json
from main import Company, Document
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RA Agent Report Viewer")

# 전역 변수로 회사 객체 저장
company = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지 - 실행 버튼과 결과 표시"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RA Agent - Report Generator</title>
        <meta charset="UTF-8">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .control-panel {
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 15px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 18px;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                font-weight: bold;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                display: none;
            }
            .status.show {
                display: block;
            }
            .status.loading {
                background: #fff3cd;
                color: #856404;
                border: 2px solid #ffc107;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 2px solid #28a745;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 2px solid #dc3545;
            }
            .results {
                display: none;
            }
            .results.show {
                display: block;
            }
            .section {
                margin-bottom: 30px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 15px;
                border-left: 5px solid #667eea;
            }
            .section h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8em;
            }
            .section h3 {
                color: #764ba2;
                margin: 20px 0 10px 0;
                font-size: 1.3em;
            }
            .page-item {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
            .page-title {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }
            .content-box {
                background: white;
                padding: 20px;
                border-radius: 10px;
                white-space: pre-wrap;
                font-family: monospace;
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .report-section {
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 10px;
                border: 2px solid #667eea;
            }
            .report-title {
                font-size: 1.5em;
                color: #764ba2;
                margin-bottom: 15px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 RA Agent - Report Generator</h1>
            
            <div class="control-panel">
                <button id="runBtn" onclick="runAnalysis()">🚀 분석 및 보고서 생성 실행</button>
                <div id="status" class="status"></div>
            </div>
            
            <div id="results" class="results">
                <div class="section">
                    <h2>📄 OCR 결과</h2>
                    <div id="ocrResults"></div>
                </div>
                
                <div class="section">
                    <h2>📊 분석 결과</h2>
                    <div id="analysisResults"></div>
                </div>
                
                <div class="section">
                    <h2>📑 생성된 보고서</h2>
                    <div id="reportResults"></div>
                </div>
            </div>
        </div>
        
        <script>
            function showStatus(message, type) {
                const status = document.getElementById('status');
                status.className = 'status show ' + type;
                status.innerHTML = message;
            }
            
            function hideStatus() {
                const status = document.getElementById('status');
                status.className = 'status';
            }
            
            async function runAnalysis() {
                const btn = document.getElementById('runBtn');
                const results = document.getElementById('results');
                
                btn.disabled = true;
                results.className = 'results';
                showStatus('<div class="spinner"></div><p>분석 중입니다... 잠시만 기다려주세요.</p>', 'loading');
                
                try {
                    const response = await fetch('/run', {
                        method: 'POST'
                    });
                    
                    if (!response.ok) {
                        throw new Error('분석 실행 실패');
                    }
                    
                    const data = await response.json();
                    showStatus('✅ 분석 및 보고서 생성 완료!', 'success');
                    
                    // 결과 표시
                    displayResults(data);
                    results.className = 'results show';
                    
                } catch (error) {
                    showStatus('❌ 오류 발생: ' + error.message, 'error');
                } finally {
                    btn.disabled = false;
                }
            }
            
            function displayResults(data) {
                // OCR 결과 표시
                const ocrDiv = document.getElementById('ocrResults');
                ocrDiv.innerHTML = '';
                if (data.ocr && data.ocr.length > 0) {
                    data.ocr.forEach((page, idx) => {
                        const pageData = page[idx];
                        const pageDiv = document.createElement('div');
                        pageDiv.className = 'page-item';
                        pageDiv.innerHTML = `
                            <div class="page-title">페이지 ${idx + 1}</div>
                            <div class="content-box">${escapeHtml(pageData || '내용 없음')}</div>
                        `;
                        ocrDiv.appendChild(pageDiv);
                    });
                } else {
                    ocrDiv.innerHTML = '<p>OCR 결과가 없습니다.</p>';
                }
                
                // 분석 결과 표시
                const analysisDiv = document.getElementById('analysisResults');
                analysisDiv.innerHTML = '';
                if (data.analysis && data.analysis.length > 0) {
                    data.analysis.forEach((page, idx) => {
                        const pageData = page[idx];
                        const pageDiv = document.createElement('div');
                        pageDiv.className = 'page-item';
                        pageDiv.innerHTML = `
                            <div class="page-title">페이지 ${idx + 1} 분석</div>
                            <div class="content-box">${JSON.stringify(pageData, null, 2)}</div>
                        `;
                        analysisDiv.appendChild(pageDiv);
                    });
                } else {
                    analysisDiv.innerHTML = '<p>분석 결과가 없습니다.</p>';
                }
                
                // 보고서 결과 표시
                const reportDiv = document.getElementById('reportResults');
                reportDiv.innerHTML = '';
                if (data.reports && Object.keys(data.reports).length > 0) {
                    for (const [reportType, reportContent] of Object.entries(data.reports)) {
                        const reportSection = document.createElement('div');
                        reportSection.className = 'report-section';
                        reportSection.innerHTML = `
                            <div class="report-title">📋 ${reportType}</div>
                            <div class="content-box">${JSON.stringify(reportContent, null, 2)}</div>
                        `;
                        reportDiv.appendChild(reportSection);
                    }
                } else {
                    reportDiv.innerHTML = '<p>생성된 보고서가 없습니다.</p>';
                }
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/run")
async def run_analysis():
    """분석 및 보고서 생성 실행"""
    try:
        global company
        
        logger.info("분석 시작")
        
        # 회사 및 문서 설정
        company = Company("example")
        name = "IR1"
        doc = "data/IR1.pdf"
        
        # PDF 파일 존재 확인
        if not os.path.exists(doc):
            raise HTTPException(status_code=404, detail=f"PDF 파일을 찾을 수 없습니다: {doc}")
        
        company.add_document(name, Document(doc))
        
        # 결과 파일 경로
        results_dir = "src/results"
        result_file = os.path.join(results_dir, f"{name}.json")
        ocr_file = os.path.join(results_dir, f"{name}_ocr.json")
        
        # 기존 결과가 있으면 로드, 없으면 분석 실행
        if os.path.exists(result_file) and os.path.exists(ocr_file):
            logger.info("기존 분석 결과 로드")
            with open(result_file, 'r', encoding='utf-8') as f:
                company.documents[name].analysis = json.load(f)
            with open(ocr_file, 'r', encoding='utf-8') as f:
                company.documents[name].ocr_texts = json.load(f)
        else:
            logger.info("새로운 분석 실행")
            await company.process_documents(debug=True)
        
        # 보고서 생성
        logger.info("보고서 생성 시작")
        await company.generate_all_reports(model="gemini", web=True)
        
        # 결과 반환
        return JSONResponse({
            "status": "success",
            "ocr": company.documents[name].ocr_texts,
            "analysis": company.documents[name].analysis,
            "reports": company.reports
        })
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    """저장된 결과 조회"""
    try:
        results_dir = "src/results"
        name = "IR1"
        
        result_file = os.path.join(results_dir, f"{name}.json")
        ocr_file = os.path.join(results_dir, f"{name}_ocr.json")
        
        if not os.path.exists(result_file) or not os.path.exists(ocr_file):
            raise HTTPException(status_code=404, detail="저장된 결과가 없습니다. 먼저 분석을 실행해주세요.")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_texts = json.load(f)
        
        # 보고서 파일도 찾아보기
        reports = {}
        for file in os.listdir(results_dir):
            if file.endswith("_reports.json"):
                with open(os.path.join(results_dir, file), 'r', encoding='utf-8') as f:
                    reports = json.load(f)
                break
        
        return JSONResponse({
            "status": "success",
            "ocr": ocr_texts,
            "analysis": analysis,
            "reports": reports
        })
        
    except Exception as e:
        logger.error(f"결과 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
