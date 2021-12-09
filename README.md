# fastapi-ratelimit
FastAPI Rate Limit

### How to package fastapi-ratelimit python project

These are archives that are uploaded to the Python Package Index and can be installed by pip.
Make sure you have the latest version of PyPA’s build installed:
```bash
py -m pip install --upgrade build
py -m build
```
Use twine to upload the distribution packages. You’ll need to install Twine:
```bash
py -m pip install --upgrade twine
```
Once installed, run Twine to upload all of the archives under dist:
```bash
py -m twine upload --repository testpypi dist/*
```

For more details please click [here](https://packaging.python.org/tutorials/packaging-projects/)

### How to use package fastapi-ratelimit
```bash
# Install package using poetry
poetry add fastapi-ratelimit

# Using pip
pip install fastapi-ratelimit
```
Set rate limit for FastAPI before starting server

Example code
- main.py
```bash
# Import package
from fastapi import FastAPI
from fastapi_ratelimit import RateLimitExceededError, RateLimitMiddleware, Limiter, rate_limit_exceeded_handler
from core.config import settings # settings is class, where the configuration parameters are saved
from api.responses import JSONResponse

...
# Setup FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_PATH}/openapi.json",
    default_response_class=JSONResponse,
)

# Set rate limit for FastAPI
app.state.limiter = Limiter(
    limit_default=settings.RATELIMIT_DEFAULT, # ex: "60/minute" (default is up to 60 requests in 1 minute)
    storage_uri=settings.RATELIMIT_STORAGE_URL # ex: "redis://redis:6379/1" (URL redis)
)
app.add_exception_handler(RateLimitExceededError, rate_limit_exceeded_handler)
app.add_middleware(RateLimitMiddleware)
```
Set limiter for some api
- example.py
```bash
...
from fastapi_ratelimit import limiter
...
@router.post("/example", response_model=schemas.Example)
@limiter.limit("2/minutes") # Up to 2 requests in 1 minute
async def example_request(
    request: Request, db: Session = Depends(deps.get_db)
) -> Any:
...
```
