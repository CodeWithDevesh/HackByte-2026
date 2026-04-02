import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src...` works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.weapon_detection.model import build_default_weapon_model

def main() -> None:
    p = argparse.ArgumentParser(description="Weapon detection model runner")
    p.add_argument("--list-cameras", action="store_true", help="List available cameras and exit.")
    args = p.parse_args()

    model = build_default_weapon_model()
    model.run(list_cameras=args.list_cameras)

if __name__ == "__main__":
    main()