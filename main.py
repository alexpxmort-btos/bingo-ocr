
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import cv2
import pytesseract
import numpy as np
import re

app = FastAPI(
    title="Bingo OCR API",
    version="3.0.0"
)

# Modelo da resposta
class CartelaResponse(BaseModel):
    cartela: List[List[int]]

# Intervalos válidos por coluna B I N G O
BINGO_RANGE = [
    (1, 15),   # B
    (16, 30),  # I
    (31, 45),  # N
    (46, 60),  # G
    (61, 75)   # O
]

# =========================
# OCR DE UMA ÚNICA CÉLULA
# =========================
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

    # Remove qualquer coisa que não seja dígito
    texto = re.sub(r"\D", "", texto)

    if texto.isdigit():
        return int(texto)

    return None


# ======================================
# CORREÇÃO USANDO REGRAS DO BINGO
# ======================================
def corrigir_por_coluna(valor, coluna, usados):
    if not isinstance(valor, int):
        return None

    minimo, maximo = BINGO_RANGE[coluna]

    candidatos = [valor]

    # Correções visuais comuns do OCR
    substituicoes = {
        "2": ["5"],
        "5": ["2"],
        "3": ["8"],
        "8": ["3", "6"],
        "6": ["8"]
    }

    valor_str = str(valor)

    for i, dig in enumerate(valor_str):
        if dig in substituicoes:
            for alt in substituicoes[dig]:
                novo = valor_str[:i] + alt + valor_str[i+1:]
                if novo.isdigit():
                    candidatos.append(int(novo))

    # Tentar todos os candidatos
    for c in candidatos:
        if minimo <= c <= maximo and c not in usados:
            return c

    return None


# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.post("/ocr/cartela", response_model=CartelaResponse)
async def ler_cartela(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    cell_h = h // 5
    cell_w = w // 5

    cartela = []
    usados = set()

    for row in range(5):
        linha = []
        for col in range(5):

            # FREE central
            if row == 2 and col == 2:
                linha.append(0)
                continue

            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            cell = img[y1:y2, x1:x2]

            bruto = ocr_numero(cell)
            corrigido = corrigir_por_coluna(bruto, col, usados)

            if corrigido is not None:
                usados.add(corrigido)
                linha.append(corrigido)
            else:
                linha.append(0)

        cartela.append(linha)

    return {"cartela": cartela}
    
