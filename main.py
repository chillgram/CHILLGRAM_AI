# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import shutil

# 전개도 생성 모듈
from services.dieline_generate import DielineAnalyzer

# 배너 생성 모듈
from services.banner_generate import AdBannerGenerator

# 영상 생성 모듈
from services.video_generate import generate_video_for_product

app = FastAPI(title="AI Product Media Server")

ROOT = Path("outputs").resolve()
ALLOWED = {"package", "video", "poster", "dieline", "banner"}


# ===============================================================================================================================================================================================================
#  경로 체크
# ===============================================================================================================================================================================================================
def ensure_product_dir(product_id: int) -> Path:
    product_dir = ROOT / "products" / str(product_id)
    product_dir.mkdir(parents=True, exist_ok=True)
    return product_dir


def ensure_job_dir(product_id: int, job_name: str = "default") -> Path:
    job_dir = ROOT / "products" / str(product_id) / "jobs" / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


# ===============================================================================================================================================================================================================
# 1) 이미지 조회 API
# ===============================================================================================================================================================================================================
@app.get("/ai/products/{product_id}/images/{img_type}")
def get_img(product_id: int, img_type: str):
    if img_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="invalid type")
    path = ROOT / "products" / str(product_id) / f"{img_type}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png")


# ===============================================================================================================================================================================================================
# 2) 더미 패키지 생성 API
# ===============================================================================================================================================================================================================
@app.post("/ai/products/{product_id}/package")
async def create_package_dummy(product_id: int, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")
    product_dir = ensure_product_dir(product_id)
    out_path = product_dir / "package.png"
    with out_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return FileResponse(out_path, media_type="image/png")


# ===============================================================================================================================================================================================================
# 3) 배너 생성 API
# ===============================================================================================================================================================================================================
class BannerRequest(BaseModel):
    headline: str = Field(..., example="맛있는 간식")
    typo_text: str = Field(..., example="손이가요 손이가.")


@app.post("/ai/products/{product_id}/banner")
def create_banner_from_file(
    product_id: int,
    headline: str = Form(...),
    typo_text: str = Form(...),
):
    product_dir = ensure_product_dir(product_id)
    input_path = product_dir / "package.png"  # 디렉터리에서 가져오는 이미지
    output_path = product_dir / "banner.png"

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"{input_path} not found")

    try:
        generator = AdBannerGenerator(api_key="김채환")
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
# 4) 영상 생성 API
# ===============================================================================================================================================================================================================
class VideoGenRequest(BaseModel):
    food_name: str = Field(..., example="새우깡")
    food_type: str = Field(..., example="스낵")
    ad_concept: str = Field(..., example="감성+트렌디")
    ad_req: str = Field(
        ..., example="바삭함, 중독성, 가벼운 간식, 자연스러운 일상 무드"
    )


@app.post("/ai/products/{product_id}/video")
async def create_video(
    product_id: int, req: VideoGenRequest, file: UploadFile = File(...)
):
    product_dir = ensure_product_dir(product_id)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload allowed")
    package_path = product_dir / "package.png"
    with package_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        final_mp4_path = await generate_video_for_product(
            product_id=product_id, req=req, product_image=file, root_dir=ROOT
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video generation failed: {e}")

    return FileResponse(
        final_mp4_path, media_type="video/mp4", filename="final_video.mp4"
    )


# ===============================================================================================================================================================================================================
# 5) 전개도(Dieline) 분석 API 
# ===============================================================================================================================================================================================================
@app.post("/ai/products/{product_id}/dieline")
def analyze_dieline(product_id: int, file: UploadFile = File(...)):
    # 1. 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # 2. 경로 설정 및 이미지 저장
    product_dir = ensure_product_dir(product_id)
    input_path = product_dir / "dieline_input.png"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. 분석 모듈 실행
    analyzer = DielineAnalyzer()
    try:
        # analyze 메서드가 결과 Dict를 반환하고, 내부적으로 이미지도 저장함
        result = analyzer.analyze(image_path=str(input_path), output_dir=product_dir)

        # 결과에 이미지 다운로드 URL 힌트 추가 (선택사항)
        result["result_image_url"] = f"/ai/products/{product_id}/images/dieline"

        return result

    except ValueError as ve:
        # 분석 실패 (점선 인식 실패 등)
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(ve)}")
    except Exception as e:
        # 기타 서버 에러
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
