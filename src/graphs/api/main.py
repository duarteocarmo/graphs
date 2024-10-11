from importlib.metadata import version

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from graphs.common.main import hello_world

app = FastAPI(
    title="graphs API",
    version=version("graphs"),
)


@app.get("/")
async def root():
    return RedirectResponse("/docs")


@app.post(
    "/hello",
)
async def hello():
    return hello_world()
