# fastapi-ratelimit
FastAPI Rate Limit

How to package fastapi-ratelimit python project

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
