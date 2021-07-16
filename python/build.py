from os import system

cmds = [
    "black python/",
    "mypy --strict --ignore-missing-imports python/",
    "pylint python/boolnn",
]

for cmd in cmds:
    system(f"python -m {cmd}")
