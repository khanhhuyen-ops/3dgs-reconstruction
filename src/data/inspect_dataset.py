"""python -m src.data.inspect_dataset --config configs/data/ignatius.yaml"""

from src.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main("inspect"))
