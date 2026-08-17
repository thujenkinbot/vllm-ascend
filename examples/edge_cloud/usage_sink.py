"""No-op backend for the Higress edge usage reporting demo."""

from fastapi import FastAPI, Response, status

app = FastAPI(title="Edge usage sink", docs_url=None, redoc_url=None)


@app.get("/health", status_code=status.HTTP_204_NO_CONTENT)
async def health() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/internal/edge-usage", status_code=status.HTTP_200_OK)
async def accept_usage() -> dict[str, object]:
    """Return a body so Higress AI Statistics finalizes token metrics."""
    return {}
