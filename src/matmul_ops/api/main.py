from fastapi import FastAPI
from .routes import router


app = FastAPI(
    title="MatMul API",
    description="Matrix multiplication operator API",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "MatMul API", "docs": "/docs"}
