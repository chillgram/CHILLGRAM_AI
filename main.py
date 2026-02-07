# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from typing import Optional
import os

# 전개도 생성 모듈
from services.dieline_generate import DielineAnalyzer

# 배너 생성 모듈
from services.banner_generate import AdBannerGenerator

# 영상 생성 모듈
from services.video_generate import generate_video_for_product

# SNS 이미지 생성 모듈
from services.sns_image_generate import SNSImageGenerator

# 패키지 이미지 생성
from services.package_generate import PackageGenerator

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="AI Product Media Server")

# 허용된 이미지 타입
ALLOWED = {"package", "video", "poster", "dieline", "banner", "sns", "sns_background"}

# ===============================================================================================================================================================================================================
#  경로 체크
# ===============================================================================================================================================================================================================

BASE_DIR = Path(__file__).resolve().parent
AI_DIR = BASE_DIR / "ai"


def ensure_product_dir(project_id: int) -> Path:
    product_dir = AI_DIR / str(project_id)
    product_dir.mkdir(parents=True, exist_ok=True)
    return product_dir


@app.get("/hello")
def hello():
    return {"message": "hello"}

# ===============================================================================================================================================================================================================
# 1) 이미지 조회 API
# ===============================================================================================================================================================================================================
@app.get("/ai/{project_id}/images/{img_type}")
def get_img(project_id: int, img_type: str):
    if img_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="invalid type")

    filename_map = {
        "package": "package.png",
        "poster": "poster.png",
        "dieline": "dieline.png",
        "banner": "banner.png",
        "sns": "sns.png",
        "sns_background": "sns_background.png",
    }

    if img_type == "video":
        raise HTTPException(status_code=400, detail="video is not an image type")

    filename = filename_map.get(img_type, f"{img_type}.png")
    product_dir = ensure_product_dir(project_id)
    path = product_dir / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(path, media_type="image/png")


# ===============================================================================================================================================================================================================
# 2) 배너 생성 API
# 배너 생성 API 호출 예시 
#js 호출
# const fd = new FormData();
# fd.append("headline", headline); 
# fd.append("typo_text", typoText);
# await fetch(`/ai/${projectId}/banner`, {
#     method: "POST",
#     body: fd
# });

#curl 방식

#curl -X POST "http://localhost:8000/ai/{project_id}/banner" \
#     -F "headline=맛있는 간식" \
#     -F "typo_text=손이가요 손이가." \
#     --output banner_result.png

# ===============================================================================================================================================================================================================
class BannerRequest(BaseModel):
    headline: str = Field(..., example="맛있는 간식")
    typo_text: str = Field(..., example="손이가요 손이가.")


@app.post("/ai/{project_id}/banner")
def create_banner_from_file(
    project_id: int,
    headline: str = Form(...),
    typo_text: str = Form(...),
):
    product_dir = ensure_product_dir(project_id)
    input_path = product_dir / "package.png"
    output_path = product_dir / "banner.png"

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"{input_path} not found")

    try:
        # API Key는 BannerGenerator 내부 혹은 환경변수 관리 권장
        generator = AdBannerGenerator(api_key=API_KEY)
        generator.process(
            image_path=str(input_path),
            headline=headline,
            typo_text=typo_text,
            output_path=str(output_path),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Banner generation failed: {e}")

    return FileResponse(output_path, media_type="image/png", filename="banner.png")


# ===============================================================================================================================================================================================================
# 3) 영상 생성 API
# ===============================================================================================================================================================================================================
class VideoGenRequest(BaseModel):
    food_name: str = Field(..., example="새우깡")
    food_type: str = Field(..., example="스낵")
    ad_concept: str = Field(..., example="감성+트렌디")
    ad_req: str = Field(..., example="바삭함, 중독성, 가벼운 간식")


@app.post("/ai/{project_id}/video")
async def create_video(
    project_id: int, req: VideoGenRequest, file: UploadFile = File(...)
):
    product_dir = ensure_product_dir(project_id)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")
    package_path = product_dir / "package.png"
    with package_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        final_mp4_path = await generate_video_for_product(
            project_id=project_id, req=req, product_image=file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video generation failed: {e}")

    return FileResponse(
        final_mp4_path, media_type="video/mp4", filename="final_video.mp4"
    )


# ===============================================================================================================================================================================================================
# 4) 전개도(Dieline) 분석 API
# ===============================================================================================================================================================================================================
@app.post("/ai/{project_id}/dieline")
def analyze_dieline(project_id: int, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    product_dir = ensure_product_dir(project_id)
    input_path = product_dir / "dieline_input.png"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analyzer = DielineAnalyzer()
    try:
        result = analyzer.analyze(image_path=str(input_path), output_dir=product_dir)
        result["result_image_url"] = f"/ai/{project_id}/images/dieline"
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===============================================================================================================================================================================================================
# 5) SNS 인스타 광고 이미지 생성 API
# ===============================================================================================================================================================================================================
# curl 호출 예시:
# curl -X POST "http://localhost:8000/ai/123/sns" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "main_text": "나야 새우깡",
#     "sub_text": "바삭함의 정석",
#     "guideline": "브랜드 컬러는 빨강과 노랑을 사용하고, 활기찬 느낌을 강조하세요.",
#     "save_background": true
#   }'
#
# 필수 INPUT:
#   - main_text (str): SNS 이미지에 들어갈 메인 텍스트 (필수)
#
# 선택 INPUT:
#   - sub_text (str): 서브 텍스트, 기본값 ""
#   - guideline (str): 팀에서 제공하는 가이드라인 프롬프트 (배경 스타일, 브랜드 컬러, 콘셉트 등)
#   - custom_prompt (str): 배경 생성용 커스텀 프롬프트 (guideline이 없을 때만 사용됨)
#   - preset (str): 배경 프리셋 이름 (예: "ocean_sunset"), guideline과 custom_prompt 둘 다 없을 때 사용
#   - save_background (bool): 배경 이미지 별도 저장 여부, 기본값 True
#
# 프롬프트 우선순위:
#   1. guideline (최우선) → 팀에서 제공하는 가이드라인 사용
#   2. custom_prompt → guideline 없을 때 커스텀 프롬프트 사용
#   3. preset (기본) → 둘 다 없을 때 내장 프리셋 사용
#   ※ 어떤 경우든 PRODUCT_SPACE_INSTRUCTION과 typography_instruction은 자동으로 추가됩니다.
#
# 사전 요구사항:
#   - package.png가 먼저 생성되어 있어야 함 (패키지 이미지 업로드 또는 생성 필요)
#
# 응답 형식:
#   {
#     "project_id": 123,
#     "sns_image_url": "/ai/123/images/sns",
#     "background_image_url": "/ai/123/images/sns_background",
#     "output_path": "ai/123/sns.png"
#   }
# ===============================================================================================================================================================================================================
class SNSGenRequest(BaseModel):
    main_text: str = Field(..., example="나야 새우깡")
    sub_text: str = Field("", example="바삭함의 정석")
    preset: Optional[str] = Field(None, example="ocean_sunset")
    custom_prompt: Optional[str] = Field(
        None, example="A dramatic night beach scene..."
    )
    guideline: Optional[str] = Field(
        None, example="브랜드 컬러는 빨강과 노랑을 사용하고, 활기찬 느낌을 강조하세요."
    )
    save_background: bool = Field(True, example=True)


@app.post("/ai/{project_id}/sns")
def create_sns_image(project_id: int, req: SNSGenRequest):
    product_dir = ensure_product_dir(project_id)
    product_path = product_dir / "package.png"
    if not product_path.exists():
        raise HTTPException(
            status_code=404, detail="package.png not found. Upload package first."
        )

    background_path = product_dir / "sns_background.png"
    final_path = product_dir / "sns.png"

    try:
        generator = SNSImageGenerator()
        generator.generate(
            product_path=str(product_path),
            main_text=req.main_text,
            sub_text=req.sub_text or "",
            preset=req.preset,
            custom_prompt=req.custom_prompt,
            guideline=req.guideline,
            output_path=str(final_path),
            save_background=req.save_background,
            background_output_path=(
                str(background_path) if req.save_background else None
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SNS generation failed: {e}")

    return {
        "project_id": project_id,
        "sns_image_url": f"/ai/{project_id}/images/sns",
        "background_image_url": (
            f"/ai/{project_id}/images/sns_background" if req.save_background else None
        ),
        "output_path": str(final_path),
    }



# ===============================================================================================================================================================================================================
# 6) 패키지 이미지 생성 API
# ===============================================================================================================================================================================================================
# curl 호출 예시:
# curl -X POST "http://localhost:8000/ai/{project_id}/package" \
#    -H "Content-Type: multipart/form-data" \
#    -F "instruction=Make it pop!" \
#    -F "dieline_file=@box_dieline.png" \
#    -F "concept_file=@shrimp_cracker_concept.png" \
#    --output package_result.png
#
# 필수 INPUT:
#    - dieline_file (File): 박스 전개도 이미지.
#    - concept_file (File): 디자인 레퍼런스 이미지. 
#
# 처리 과정 (Logic):
#    1. 구조 분석: OpenCV로 전개도의 윤곽선을 따서 메인/사이드/상단 패널 좌표(x,y,w,h) 자동 추출
#    2. 배경 생성: 컨셉 이미지의 주조색을 추출하여 자연스러운 종이 질감(Coated Cardboard) 배경 생성
#    3. 디자인 생성: Gemini가 각 패널(앞면: 씨즐/로고, 옆면: 정보, 윗면: 가로형 로고)에 맞는 2D 그래픽 생성
#    4. 최종 합성: 생성된 이미지를 마스킹하여 전개도에 입히고(Masking), 원본 라인을 선명하게 복원(Multiply)
#
# 응답 형식:
#   - image/png (최종 생성된 패키지 디자인)
# ===============================================================================================================================================================================================================
@app.post("/ai/{project_id}/package")
async def create_package_with_gemini(
    project_id: int,
    dieline_file: UploadFile = File(...),
    concept_file: UploadFile = File(...),
):
    # 1. 파일 검증
    if not dieline_file.content_type.startswith("image/") or not concept_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")

    product_dir = ensure_product_dir(project_id)

    # 2. 업로드 원본 저장
    # (PackageGenerator 내부 로직에서 이 파일명들을 참조함)
    dieline_path = product_dir / "dieline_input.png"
    concept_path = product_dir / "concept_input.png"

    with dieline_path.open("wb") as buffer:
        shutil.copyfileobj(dieline_file.file, buffer)
        
    with concept_path.open("wb") as buffer:
        shutil.copyfileobj(concept_file.file, buffer)

    # 3. 결과 저장 경로
    output_path = product_dir / "package.png"

    # 4. Gemini 호출
    try:
        generator = PackageGenerator(api_key=API_KEY)
        generator.edit_package_image(
            product_dir=str(product_dir),
        )

    except Exception as e:
        print(f"Error generation package: {e}")
        raise HTTPException(status_code=500, detail=f"package generation failed: {e}")

    # 5. 생성된 이미지 반환
    return FileResponse(output_path, media_type="image/png", filename="package.png")


class PackageGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # self.client = genai.Client(api_key=self.api_key)
        pass

    def edit_package_image(self, product_dir: str):
        """
        [Main Logic] 전개도(dieline_input.png)와 컨셉(concept_input.png)을 사용하여 패키지 디자인을 생성합니다.
        
        Args:
            product_dir (str): 파일들이 저장된 프로젝트 디렉토리 경로. 
            내부에 'dieline_input.png'와 'concept_input.png'가 있어야 함.
        """
        pass

@app.get("/hello")
def hello():
    return {"message":"hello"}
