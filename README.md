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
example code
- main.py
```bash
# Import package
from fastapi import FastAPI
from fastapi_ratelimit import RateLimitExceededError, RateLimitMiddleware, limiter, rate_limit_exceeded_handler, set_settings
from core.config import settings # settings is class, where the configuration parameters are saved

...
# setup FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_PATH}/openapi.json",
    default_response_class=JSONResponse,
)

# set rate limit for FastAPI
set_settings(settings=settings)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceededError, rate_limit_exceeded_handler)
app.add_middleware(RateLimitMiddleware)
```
- example.py
```bash
...
from fastapi_ratelimit import limiter
...
@router.post("/example", response_model=schemas.Example)
@limiter.limit("2/minutes") # up to 2 requests in 1 minute
async def example_request(
    request: Request, db: Session = Depends(deps.get_db)
) -> Any:
...
```
