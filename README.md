# Knowledge-Library Repository

Comprehensive guide to create, maintain, and extend the study archive that spans fundamental mathematics through production-grade system design.

---

## 0. Purpose at a Glance

* **Single source-of-truth** for books, notebooks, scripts, datasets, and research material.
* **Branch-protected workflow** so every change is peer-reviewed (even if the peer is “future me”).
* **Git LFS** for large binaries; repository clones stay lightweight.
* **Decade directory schema (`00_`, `10_`, …)**—scales without renaming.

---

## 1. Sequential Workflow (chronological record)

| #  | Action                               | Command / UI Steps                                                                                                        | Result                                            |
| -- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 1  | Create empty repo `ds-de-se-hub`     | GitHub → **New repository** (no README)                                                                                   | Blank remote                                      |
| 2  | Install tooling                      | `brew install git gh git-lfs pre-commit`                                                                                  | Git, GH CLI, LFS, pre-commit available            |
| 3  | Generate SSH key                     | `ssh-keygen -t ed25519 -C "<your_email@example.com>"` → add public key on GitHub                                            | Password-less Git pushes                          |
| 4  | Clone & scaffold                     | `bash`<br>`git clone git@github.com:<username>/ds-de-se-hub.git`<br>`cd ds-de-se-hub`<br>`python create_structure.py`          | Decade folders (`00_`, `10_`, …) + `.gitkeep` files |
| 5  | Protect `main`                       | GitHub → **Settings ▸ Branches** → Add protection rule (require pull request, require status checks, etc.)              | Direct pushes to `main` blocked                     |
| 6  | Add baseline `.gitignore`            | Create/update `.gitignore` (e.g., from gitignore.io) → `git add .gitignore` → `git commit -m "chore: add gitignore"`      | OS junk, IDE files, virtual envs ignored          |
| 7  | Enable Git LFS                       | `bash`<br>`git lfs install`<br>`git lfs track "*.pdf" "*.csv" "*.png" "*.jpg" "*.epub" "*.pptx" "*.xls" "*.xlsx" "*.gz" "*.zip"`<br>`git add .gitattributes`<br>`git commit -m "chore: enable LFS and track common large file types"` | Large binaries become pointers                    |
| 8  | Store PAT for LFS                    | Create GitHub PAT (scopes: `repo`, `write:packages`) → `git credential approve <<< "protocol=https\nhost=github.com\nusername=<your_username>\npassword=<your_PAT>"` | Silent LFS uploads                                |
| 9  | Populate content in feature branches | (See Commit & PR Workflow section below)                                                                                  | Structured content per topic                      |
| 10 | Open pull requests                   | `gh pr create --fill --web` → Review → Merge via UI                                                                       | Reviewed, linear history maintained               |
| 11 | Install pre-commit hooks             | `bash`<br>`pre-commit sample-config > .pre-commit-config.yaml` (customize as needed)<br>`pre-commit install`                | Auto-strip notebooks, run linters/formatters      |
| 12 | Write master README                  | Copy this document into `README.md` → `git add README.md` → `git commit -m "docs: initialize master README"`               | Single entry-point for the repo                   |

---

## 2. Issue-Resolution Log

| Issue                                                                                         | Attempts                                                                                                                                     | Final solution                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Infinite password loop on LFS push                                                            | 1. Retyped GitHub password (rejected)<br>2. Switched remote to SSH (Git refs ok, LFS still loops)                                         | • Set `credential.helper` (`osxkeychain` / `store` / `manager`) <br>• Generated PAT (scopes: `repo`, `write:packages`)<br>• Stored PAT via `git credential approve` (see step 8 in Sequential Workflow)<br>• Loaded SSH key into agent: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` (macOS) or `ssh-add ~/.ssh/id_ed25519` |
| `git pull` blocked by uncommitted renames                                                     | Considered committing vs ignoring                                                                                                            | • Used `git stash --include-untracked` → `git pull` → `git stash pop` → Resolved conflicts → Committed                                                                                                                                                                                                                                                            |
| Large-file (>50 MB) warning from GitHub                                                       | Ignored initially, then encountered push failures for files >100MB.                                                                          | • Adopted Git LFS.<br>• Migrated existing large files: `git lfs migrate import --everything --include="*.pdf,*.png,*.zip"` (adjust as needed)                                                                                                                                                                                                                  |
| Mac `Icon\r` or `Icon\015` artifacts & inconsistent casing                                      | —                                                                                                                                            | • Added `Icon\r` and `Icon?` to `.gitignore`.<br>• Purged existing artifacts: `find . -name "Icon?" -delete -print`.<br>• Batch-renamed files to snake_case with a script or `rename` utility.<br>• Configured Git for case sensitivity (carefully): `git config core.ignorecase false` (use with caution, especially in mixed OS environments). |
| Should `.gitkeep` be tracked?                                                                 | Debated whether to keep them once a directory has actual content.                                                                            | • Keep in directories that are intentionally empty or are expected to receive content later.<br>• Delete from folders once they are guaranteed to contain other files (unless the `.gitkeep` serves a specific organizational purpose).                                                                                                                         |
| Git LFS Pointer vs. Actual File Conflict:<br>"Encountered X files that should have been pointers, but weren't" | 1. Large file manually placed/overwritten LFS pointer.<br>2. Git LFS not correctly installed/configured for the repo.<br>3. Merge/pull resulted in actual file instead of pointer.<br>4. Incorrect `checkout` of LFS files. | • **Isolate changes:** `git stash push -u -m "temp_lfs_fix"` (if uncommitted changes exist, `-u` includes untracked files).<br>• **Remove local copy of the problematic file(s):** `rm path/to/problematic/file.pdf` (or `del` on Windows).<br>• **Force Git LFS to re-checkout (re-smudge) the file(s):** `git lfs checkout path/to/problematic/file.pdf`. For all LFS files: `git lfs checkout .`<br>• **Verify:** `git lfs status` should show the file correctly. `git status` should be clean regarding this file.<br>• **Proceed with Git operation:** e.g., `git pull`.<br>• **Reapply stashed changes:** `git stash pop` (if stashed). Resolve any conflicts. |

---

## 3. Final Folder Structure & Rationale

```
.
├── 00_foundations
│   ├── books
│   │   └── data_science_from_scratch_2nd_edition_joel_grus.pdf
│   ├── Career_In_DataScience_UltimateGuide_365DS.pdf
│   ├── cv/
│   ├── ds_buzzwords_explained.png
│   ├── ds_www_explained.png
│   └── statistics
│       └── great_books/
├── 10_programming
│   └── python
│       ├── 00_books/
│       ├── 01_reference_pdfs/
│       ├── 02_notebooks/
│       ├── 03_scripts/
│       ├── 90_playgrounds/
│       │   └── python-functional
│       │       ├── 01_demo.py
│       │       ├── 02_data.py
│       │       ├── 03_higher_order.py
│       │       ├── 04_monads.py
│       │       ├── project
│       │       │   ├── blackjack.py
│       │       │   ├── card_games.py
│       │       │   ├── Icon\015
│       │       │   ├── monads.py
│       │       │   └── utils.py
│       │       ├── slides.pptx
│       │       └── solution
│       │           ├── blackjack_solution.py
│       │           ├── blackjack_stream.py
│       │           ├── card_games.py
│       │           ├── Icon\015
│       │           ├── monads.py
│       │           └── utils.py
│       └── data/
├── 20_data_science
│   ├── 21_ml_classical
│   │   ├── 00_books/
│   │   ├── 01_references/
│   │   ├── 02_notebooks/
│   │   └── data/
│   ├── 22_deep_learning
│   │   ├── 00_books/
│   │   └── 01_references
│   │       └── deep_learning_basics.pdf
│   └── 23_specialisations
│       ├── actuarial/
│       ├── nlp/
│       └── time_series/
├── 30_data_engineering
│   └── spark_big_data
│       └── books/
├── 40_software_engineering
│   ├── dsa_algorithms/
│   └── system_design/
├── 50_industry_and_research
│   ├── [2022]_state_of_ai_report.pdf
│   ├── [2023]_state_of_ai_report.pdf
│   ├── [2024]state_of_ai_report.pdf
│   └── 00_ml_papers/
├── 60_interview_prep
│   ├── 365_ds_interview_qs.pdf
│   ├── case_studies/
│   └── coding_drills/
├── 90_projects
│   ├── python_ad_hoc/
│   └── rv-matrix-themed-portfolio/
└── README.md
```

* **Decade prefixes** (`00_`, `10_`, …) provide reserved slots (`70_`, `80_`) for future tracks.
* **Standard sub-folders** (`00_books`, `01_references`, `02_notebooks`, `03_scripts`, `data`) aim to keep each topic self-contained.
* **Git LFS** controls repo size by tracking large files like PDFs, images, and datasets.
* **Protected `main` branch** ensures every change is reviewed via a Pull Request.

*(Note: The `Icon\r` or `Icon\015` files listed in the structure are typically macOS specific metadata files. They should ideally be added to the global or repository `.gitignore` file and removed from the repository. See Issue-Resolution Log.)*

---

## 4. Artefact Type Classification

| Artefact                         | Destination                 | Notes                                                              |
| -------------------------------- | --------------------------- | ------------------------------------------------------------------ |
| Full textbook / long PDF         | `.../<topic>/00_books/`       | Comprehensive texts.                                               |
| Slide deck / cheat-sheet / paper | `.../<topic>/01_references/`  | Quick reference materials, academic papers, concise guides.        |
| Notebook or tutorial series      | `.../<topic>/02_notebooks/`   | Jupyter notebooks, code-along tutorials. Sub-folder by sub-topic.  |
| Reusable helper script / module  | `.../<topic>/03_scripts/`     | Utility scripts, helper functions.                                 |
| Sample dataset                   | `.../<topic>/data/`           | CSVs, Excel files, etc., relevant to the topic. Consider LFS.    |
| One-off exploration / experiment | `90_projects/<short-name>/` | Self-contained projects or explorations with their own structure.  |
| CV / Resume materials            | `00_foundations/cv/`        | Resumes, cover letter templates.                                   |
| Industry Reports / ML Papers     | `50_industry_and_research/` | Annual reports, significant research papers (can also be in `01_references` if topic-specific). |

---

## 5. Commit & PR Workflow

```bash
# 0. Ensure main branch is up-to-date before starting new work
git checkout main
git pull origin main

# 1. Create a descriptive feature branch from main
# Branch naming convention: <type>/<scope>/<short-description>
# Examples: feat/python/add-async-utils, fix/docs/update-readme, chore/lfs/track-epubs
git checkout -b <type>/<scope>/<short-description>

# 2. Make changes: Add, edit, or delete files
# Stage changes thoughtfully:
# git add path/to/specific/file.ext
# git add . # To stage all changes in the current directory and subdirectories
# git add -p # For interactive staging, reviewing each hunk

# 3. Commit with a clear, conventional message
# Format: <type>(<scope>): <imperative short description>
# Example: feat(ml): implement k-means clustering notebook
# Example: docs(readme): update folder structure diagram
# (Body of commit message is optional but useful for more context)
git commit -m "<type>(<scope>): <short description of changes>"
# For multi-line commit messages:
# git commit # This will open your configured editor

# 4. Run pre-commit hooks to ensure quality and formatting
pre-commit run --all-files
# If pre-commit makes changes, review and commit them:
# git add .
# git commit --amend --no-edit # If changes are trivial and part of the last commit
# OR git commit -m "chore: apply pre-commit formatting"

# 5. (Optional but Recommended for long-lived branches) Keep your branch updated with main
# This helps prevent large merge conflicts later.
git fetch origin main
git rebase origin/main # Or git merge origin/main
# Resolve any conflicts, then `git add .` and `git rebase --continue` or `git commit`.

# 6. Push your feature branch to the remote repository
# The -u flag sets the upstream branch for future pushes/pulls on this branch.
git push -u origin <type>/<scope>/<short-description>

# 7. Create a Pull Request (PR) on GitHub
# Using GitHub CLI:
gh pr create --fill --web
# Or navigate to your repository on GitHub to create the PR.
# - Write a clear PR title and detailed description.
# - Link to any relevant issues (e.g., "Closes #123").
# - Explain the "why" and "what" of your changes.
# - If it's a work-in-progress, create a "Draft PR".

# 8. Peer Review (essential, even if it's myself)
# - Collaborators (or you) review the changes.
# - Address feedback by making new commits on your feature branch and pushing them.
# - The PR will update automatically with new commits.

# 9. Merge the PR
# - Once approved and all automated checks (if any) pass, merge the PR (usually via the GitHub UI).
# - Preferred merge strategy: "Squash and merge" or "Rebase and merge" for a clean, linear history on `main`.

# 10. Post-merge cleanup
git checkout main
git pull origin main
git branch -d <type>/<scope>/<short-description> # Delete local branch
# The remote branch might be deleted automatically by GitHub upon merge, or you can delete it manually via UI or:
# git push origin --delete <type>/<scope>/<short-description>
```

## 6. How to Place Future Files

| Decision Question               | Guidance                                                                                                                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Which decade?**               | Map to domain: `00`=foundations, `10`=programming, `20`=data\_science, `30`=data\_engineering, `40`=software\_engineering, `50`=industry, `60`=interview, `90`=projects |
| **Heavy PDF / dataset?**        | Track via Git LFS (`git lfs track "*.pdf" "*.csv"`).                                                                                                                    |
| **Long-form vs reference?**     | `00_books` for full texts; `01_references` for slide decks, cheat-sheets.                                                                                               |
| **Notebook or code tutorial?**  | `02_notebooks` under the appropriate track.                                                                                                                             |
| **Reusable helper or library?** | `03_scripts` under the same track.                                                                                                                                      |
| **Throw-away exploration?**     | `90_projects/<short-desc>`.                                                                                                                                             |
| **Cross-topic resource?**       | Choose the primary learning intent (e.g., ML paper → `50_industry_and_research/00_ml_papers`).                                                                          |
| **Empty directory after move?** | Keep a `.gitkeep` if dir may receive future content; otherwise delete.                                                                                                  |

---


## 7. Do’s and Don’ts

Do:

- Use feature branches for every change—no direct commits to main.
- Follow the Commit & PR Workflow described above.
- Maintain snake_case filenames without spaces (e.g., my_awesome_notebook.ipynb).
- Write clear and conventional commit messages.
- Commit both pyproject.toml and its lockfile (e.g., uv.lock, poetry.lock, pdm.lock) when managing Python dependencies.
- Run pre-commit run --all-files before pushing to catch issues early.
- Track large binaries (PDFs, datasets, images, etc.) with Git LFS.
- Ensure your local main branch is up-to-date (git pull origin main) before creating new feature branches.

Don’t

- Push directly to the main branch.
- Commit virtual environment folders (e.g., venv/, .venv/, __pypackages__). These should be in .gitignore.
- Commit personal/proprietary/sensitive datasets or API keys—keep them local and ignored, or use secure secret management.
- Bypass branch-protection rules unless absolutely necessary and with explicit agreement if collaborating.
- Rename top-level decade folders without updating references and considering the impact on history/links.
- Commit large files that are not tracked by LFS. Double-check git lfs status if unsure.

## 8. Other Resources Link-Board (live)

All external links and reading lists live in Notion, synced manually:
[https://www.notion.so/rv/ds-de-se-knowledge-library](https://www.notion.so/1ef21584c09f80398551fb83167d0a62?v=1fa21584c09f80a3a66a000cf41d6ffd&pvs=4)

---


## 9. Maintainers

**Primary:** rv (Data Science)
For questions: open an issue or email `23rishavsharma@gmail.com`.

---

*Document version: 2025-05-26*
