default:
    just --list

setup:
    poetry install

tidy path=".":
    poetry run isort {{ path }}
    poetry run ruff format {{ path }}
    just --fmt --unstable

check:
    poetry run ruff check
