import os
import re
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Faltan variables SUPABASE_URL o SUPABASE_KEY en el entorno.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="InSignia API - Motor PLN y Lengua de Señas",
    version="1.0.1",
    description="Backend desacoplado para traducción Voz -> Glosas LSM/ASL/LSE (< 1.5s)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TraduccionRequest(BaseModel):
    texto: str
    idioma: str = "LSM"

class GlosaAnimada(BaseModel):
    palabra_glosa: str
    categoria: str
    url_animacion: Optional[str]
    encontrada: bool

class TraduccionResponse(BaseModel):
    texto_original: str
    idioma: str
    secuencia_glosas: List[GlosaAnimada]
    tiempo_procesamiento_ms: float

STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "con", "por", "para", "sin",
    "que", "y", "o", "pero", "se", "me", "te", "nos", "le", "les",
    "es", "son", "esta", "estan", "fue", "muy"
}

LEMAS_VERBOS = {
    "compro": "comprar", "comprare": "comprar", "compre": "comprar", "comprando": "comprar",
    "estudio": "estudiar", "estudiare": "estudiar", "estudie": "estudiar", "estudiando": "estudiar",
    "voy": "ir", "vas": "ir", "va": "ir", "vamos": "ir", "fui": "ir",
    "tengo": "tener", "tienes": "tener", "tiene": "tener",
    "quiero": "querer", "quieres": "querer", "quiere": "querer",
    "necesito": "necesitar", "necesitas": "necesitar"
}

def normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"[^\w\sáéíóúñ]", "", texto)
    return texto

def lematizar_token(token: str) -> str:
    if token in LEMAS_VERBOS:
        return LEMAS_VERBOS[token]
    if token.endswith(("ando", "iendo")):
        return token[:-4] + "ar"
    return token

def registrar_palabra_pendiente(lema: str, lengua_id: int):
    try:
        existente = supabase.table("glosas_pendientes")\
            .select("id, veces_solicitada")\
            .eq("lema_espanol", lema)\
            .eq("lengua_id", lengua_id)\
            .execute()

        if existente.data:
            registro = existente.data[0]
            supabase.table("glosas_pendientes")\
                .update({"veces_solicitada": registro["veces_solicitada"] + 1})\
                .eq("id", registro["id"])\
                .execute()
        else:
            supabase.table("glosas_pendientes").insert({
                "lema_espanol": lema,
                "lengua_id": lengua_id,
                "estado": "BUSCANDO"
            }).execute()
    except Exception as e:
        print(f"[Aviso] No se pudo registrar glosa pendiente '{lema}': {e}")

@app.get("/health")
def health_check():
    return {"status": "online", "servicio": "InSignia PLN Backend"}

@app.post("/api/v1/traducir-voz", response_model=TraduccionResponse)
def traducir_voz_a_glosas(req: TraduccionRequest):
    inicio = time.perf_counter()

    try:
        # 1. Validar idioma
        res_lengua = supabase.table("lenguas_senas")\
            .select("id, activa")\
            .eq("codigo", req.idioma.upper())\
            .execute()

        if not res_lengua.data:
            raise HTTPException(status_code=404, detail=f"Idioma '{req.idioma}' no soportado.")
        
        lengua_id = res_lengua.data[0]["id"]

        # 2. Tokenizar y lematizar
        texto_limpio = normalizar_texto(req.texto)
        tokens = [lematizar_token(t) for t in texto_limpio.split() if t not in STOPWORDS_ES]

        if not tokens:
            return TraduccionResponse(
                texto_original=req.texto,
                idioma=req.idioma.upper(),
                secuencia_glosas=[],
                tiempo_procesamiento_ms=round((time.perf_counter() - inicio) * 1000, 2)
            )

        # 3. Cargar categorías gramaticales (sin JOIN complejo para evitar errores de relación)
        res_cats = supabase.table("categorias_gramaticales").select("id, codigo, orden_sintactico").execute()
        mapa_cats = {c["id"]: c for c in res_cats.data}

        # 4. Consultar glosas en Supabase
        res_glosas = supabase.table("glosas")\
            .select("palabra_glosa, lema_espanol, url_animacion_vectorial, categoria_id")\
            .eq("lengua_id", lengua_id)\
            .in_("lema_espanol", tokens)\
            .execute()

        mapa_encontradas = {item["lema_espanol"]: item for item in res_glosas.data}

        # 5. Armar secuencia
        glosas_procesadas = []
        for token in tokens:
            if token in mapa_encontradas:
                dato = mapa_encontradas[token]
                cat_info = mapa_cats.get(dato["categoria_id"], {"codigo": "GENERAL", "orden_sintactico": 90})
                glosas_procesadas.append({
                    "palabra_glosa": dato["palabra_glosa"],
                    "categoria": cat_info["codigo"],
                    "orden": cat_info["orden_sintactico"],
                    "url_animacion": dato["url_animacion_vectorial"],
                    "encontrada": True
                })
            else:
                registrar_palabra_pendiente(token, lengua_id)
                glosas_procesadas.append({
                    "palabra_glosa": token.upper(),
                    "categoria": "DELETREO",
                    "orden": 99,
                    "url_animacion": None,
                    "encontrada": False
                })

        # 6. Reordenamiento T-S-O-V
        glosas_ordenadas = sorted(glosas_procesadas, key=lambda x: x["orden"])

        secuencia_final = [
            GlosaAnimada(
                palabra_glosa=g["palabra_glosa"],
                categoria=g["categoria"],
                url_animacion=g["url_animacion"],
                encontrada=g["encontrada"]
            )
            for g in glosas_ordenadas
        ]

        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)

        return TraduccionResponse(
            texto_original=req.texto,
            idioma=req.idioma.upper(),
            secuencia_glosas=secuencia_final,
            tiempo_procesamiento_ms=latencia_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno detallado: {str(e)}")