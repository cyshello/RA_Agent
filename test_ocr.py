from src.utils import pyt_ocr, paddle_ocr, paddle_ocr_with_bbox, CLOVA_ocr, Upstage_ocr, parse_upstage, parse_upstage_to_html
import PIL
import json
import io


image_path = "/home/intern/youngseo/RA_Agent/data/image.png"

def test_pyt_ocr():
    image = PIL.Image.open(image_path)
    texts = pyt_ocr(image)
    assert isinstance(texts, list) 
    assert len(texts) > 0
    for text in texts:
        assert isinstance(text, str)
    print("=== Pytesseract OCR 결과 ===")
    for text in texts:
        print(text)
    print()

def test_paddle_ocr():
    image = PIL.Image.open(image_path)
    texts = paddle_ocr(image)
    assert isinstance(texts, list)
    assert len(texts) > 0
    for text in texts:
        assert isinstance(text, str)
    print("=== PaddleOCR 결과 (텍스트만) ===")
    for text in texts:
        print(text)
    print()

def test_paddle_ocr_with_bbox():
    image = PIL.Image.open(image_path)
    results = paddle_ocr_with_bbox(image)
    assert isinstance(results, list)
    assert len(results) > 0
    print(f"=== PaddleOCR 결과 (Bounding Box 포함) - 총 {len(results)}개 ===")
    for i, item in enumerate(results):
        print(f"{i+1}. 텍스트: '{item['text']}'")
        print(f"   신뢰도: {item['confidence']:.4f}")
        print(f"   위치: {item['bbox']}")
        print()

def test_clova_ocr():
    image = PIL.Image.open(image_path)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    texts = CLOVA_ocr(img_byte_arr)
    assert isinstance(texts, list)
    assert len(texts) > 0
    for text in texts:
        assert isinstance(text, str)
    
    # 결과를 JSON으로 저장
    import os
    results_dir = "/home/intern/youngseo/RA_Agent/src/results"
    os.makedirs(results_dir, exist_ok=True)
    json_output_path = os.path.join(results_dir, "clova_ocr_result.json")
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump({"texts": texts}, f, indent=2, ensure_ascii=False)
    
    print("=== CLOVA OCR 결과 ===")
    print(f"JSON 결과가 저장되었습니다: {json_output_path}")
    print(f"총 {len(texts)}개의 텍스트가 추출되었습니다.")
    print()

def test_upstage_ocr(model="ocr"):
    image = PIL.Image.open(image_path)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    result = Upstage_ocr(img_byte_arr, model=model)
    assert isinstance(result, dict)
    
    # JSON 결과를 파일로 저장
    import os
    results_dir = "/home/intern/youngseo/RA_Agent/src/results"
    os.makedirs(results_dir, exist_ok=True)
    json_output_path = os.path.join(results_dir, "upstage_ocr_result.json")
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("=== Upstage OCR 원본 결과 ===")
    print(f"JSON 결과가 저장되었습니다: {json_output_path}")
    if "elements" in result:
        print(f"총 {len(result['elements'])}개의 요소가 추출되었습니다.")
    print()

    print("=== parse_upstage로 추출된 텍스트 ===")
    texts = parse_upstage(result)
    assert isinstance(texts, list)
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    print()
    
    if model == "document-parse":
        print("=== parse_upstage_to_html로 HTML 생성 ===")
        html_output_path = "/home/intern/youngseo/RA_Agent/data/upstage_result.html"
        html_content = parse_upstage_to_html(result, html_output_path)
        print(f"HTML 파일이 저장되었습니다: {html_output_path}")
        print()

# print("\nCLOVA OCR 테스트:")
# test_clova_ocr()

print("\nUpstage OCR 테스트:")
test_upstage_ocr()
