from fastapi import HTTPException


class InternalServerError(HTTPException):
    status_code = 503

    def __init__(self, detail: str):
        super().__init__(status_code=self.status_code, detail=detail)
