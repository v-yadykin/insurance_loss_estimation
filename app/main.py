import time

from core.exceptions import InternalServerError
from fastapi import FastAPI, HTTPException, Request
from loguru import logger

app = FastAPI(docs_url="/api/v1/docs")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.exception(e)
        raise InternalServerError(detail=str(e))
