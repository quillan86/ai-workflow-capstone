from fastapi import FastAPI
#from .routers import
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

# -----------------------------------------------------------------------------
# APPLICATION OBJECT
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Revenue Forecasting Microservice",
    description="A forecasting microservice for streaming revenue",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url=None
)

# -----------------------------------------------------------------------------
# CORS RULES
# -----------------------------------------------------------------------------
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# ADD ROUTERS
# -----------------------------------------------------------------------------
# app.include_router(outlier.router, prefix="/api/v1")

