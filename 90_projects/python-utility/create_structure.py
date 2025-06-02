"""
create_structure.py
-------------------
Quickly bootstrap the folder skeleton for my personal study-library repo.

Usage:
    python create_structure.py [root_path]

If *root_path* is omitted, the structure is created in the CWD.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Edit this list to add/remove folders later
# ---------------------------------------------------------------------
FOLDERS = [
    "00_foundations",
    "10_programming",
    "20_data_science/21_ml_classical",
    "20_data_science/22_deep_learning",
    "20_data_science/23_specialisations/nlp",
    "20_data_science/23_specialisations/time_series",
    "30_data_engineering/spark_big_data",
    "40_software_engineering/dsa_algorithms",
    "40_software_engineering/system_design",
    "50_industry_reports",
    "60_interview_prep/coding_drills",
    "60_interview_prep/case_studies",
    "60_interview_prep/behavioural",
    "90_projects",
    "docs",
]

def touch_gitkeep(dir_path: Path) -> None:
    """Drop a .gitkeep so that empty folders stay under version control."""
    keep_file = dir_path / ".gitkeep"
    if not keep_file.exists():
        keep_file.touch()

def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"Creating study-library skeleton in: {root.resolve()}\n")

    for rel_path in FOLDERS:
        dir_path = root / rel_path
        dir_path.mkdir(parents=True, exist_ok=True)
        touch_gitkeep(dir_path)
        print(f"  └─ {dir_path.relative_to(root)}")

    print("\nAll set! Initialise a git repo, add your README, and start learning.")

if __name__ == "__main__":
    main()

