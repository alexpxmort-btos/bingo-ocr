
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import cv2
import pytesseract
import numpy as np
import re

app = FastAPI(
    title="Bingo OCR API",
    description="API para leitura de cartela de bingo 5x5 via imagem",
    version="1.0.0"
)

class CartelaResponse(BaseModel):
    cartela: List[List[int]]

def ocr_numero(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    texto = pytesseract.image_to_string(
        thresh,
        config="--psm 10 -c tessedit_char_whitelist=0123456789"
    )

    nums = re.findall(r"\d+", texto)
    return int(nums[0]) if nums else 0

@app.post(
    "/ocr/cartela",
    response_model=CartelaResponse,
    summary="Ler cartela de bingo 5x5",
    tags=["OCR"]
)
async def ler_cartela(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    cell_h = h // 5
    cell_w = w // 5

    cartela = []

    for row in range(5):
        linha = []
        for col in range(5):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            cell = img[y1:y2, x1:x2]

            if row == 2 and col == 2:
                linha.append(0)
            else:
                linha.append(ocr_numero(cell))

        cartela.append(linha)

    return {"cartela": cartela}
              
