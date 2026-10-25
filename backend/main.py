from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analytics_routes import router as analytics_router
from app.translation_routes import router as translation_router

app = FastAPI(
    title="DarijaFlow API",
    description="API de traduction et d'analyse des expériences LSTM",
    version="1.0.0",
)

# Autoriser le frontend React lancé par Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Réutiliser la route d'analyse existante
app.include_router(analytics_router)
app.include_router(translation_router)


@app.get("/")
def root():
    return {
        "message": "DarijaFlow API fonctionne",
        "analytics": "/analytics/data",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}