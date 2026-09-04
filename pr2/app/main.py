from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from app.inference import detect_objects
import uvicorn

app = FastAPI(title="Local Object Detection API")

@app.post("/detect/")
async def detect_api(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, description="Поріг упевненості від 0 до 1")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Надісланий файл не є зображенням")

    try:
        image_bytes = await file.read()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл порожній")

        result = detect_objects(image_bytes, conf_threshold)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)