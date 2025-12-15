from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import cv2
import pytesseract
import numpy as np
import re

app = FastAPI(
    title="Bingo OCR API",
    version="2.0.0"
)

class CartelaResponse(BaseModel):
    cartela: List[List[int]]

# Intervalos válidos por coluna
BINGO_RANGE = [
    (1, 15),   # B
    (16, 30),  # I
    (31, 45),  # N
    (46, 60),  # G
    (61, 75)   # O
]

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
    return int(nums[0]) if nums else None


def corrigir_por_coluna(valor, coluna):
    if valor is None:
        return None

    minimo, maximo = BINGO_RANGE[coluna]

    # Valor já válido
    if minimo <= valor <= maximo:
        return valor

    # Tentativa comum de correção
    # Ex: 2 -> 25, 7 -> 57
    for dezenas in [10, 20, 30, 40, 50, 60, 70]:
        candidato = dezenas + valor
        if minimo <= candidato <= maximo:
            return candidato

    return None


@app.post("/ocr/cartela", response_model=CartelaResponse)
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
            # FREE do meio
            if row == 2 and col == 2:
                linha.append(0)
                continue

            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            cell = img[y1:y2, x1:x2]

            bruto = ocr_numero(cell)
            corrigido = corrigir_por_coluna(bruto, col)

            linha.append(corrigido if corrigido is not None else 0)

        cartela.append(linha)

    return {"cartela": cartela}
    
