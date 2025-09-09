from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2, numpy as np
from pathlib import Path
from detect_color import ColorPlateDetector
from ocr import PlateOCR
from main import process_image

app=FastAPI(title="ALPR API")
det, ocr = ColorPlateDetector(), PlateOCR()

@app.post("/infer/image")
async def infer_image(file: UploadFile = File(...)):
    b=await file.read()
    tmp=Path("__api.jpg"); tmp.write_bytes(b)
    res=process_image(tmp, det, ocr)
    tmp.unlink(missing_ok=True)
    return JSONResponse(res)
