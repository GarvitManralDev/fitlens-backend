from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.health import router as health_router
from .routes.analyze import router as analyze_router

app = FastAPI(title="FitLens Backend (ML-Only)")

# CORS is open for MVP; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # TODO: replace with your web app's URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers (prefixes matter!)
app.include_router(health_router, prefix="/api")
app.include_router(analyze_router, prefix="/api")
