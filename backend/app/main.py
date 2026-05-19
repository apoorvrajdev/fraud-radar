"""FastAPI application entrypoint for the Fraud Radar service."""
from fastapi import FastAPI

app = FastAPI(
    title="Fraud Radar API",
    description="Real-time fraud detection for financial transactions.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}
