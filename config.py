import tomllib
from functools import cache

@cache
def load_config(path="./config.toml"):
    with open(path, "rb") as fp:
        config = tomllib.load(fp)
    return config