"""python -m src.data.verify_compatibility --config ... --references ..."""

from src.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main("compatibility"))
