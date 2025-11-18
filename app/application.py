from fastapi import FastAPI
from .config import get_settings
from .logging import setup_logging
from .api.router import router

settings = get_settings()
setup_logging(settings.LOG_JSON)

app = FastAPI(title="AgentAddon VectorHub", version="0.1.0")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.application:app", host="0.0.0.0", port=8090, reload=True)
