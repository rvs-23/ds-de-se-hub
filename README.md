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

| #  | Action                               | Command / UI Steps                                                                                                        | Result                                        |
| -- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| 1  | Create empty repo `ds-de-se-hub`     | GitHub → **New repository** (no README)                                                                                   | Blank remote                                  |
| 2  | Install tooling                      | `brew install git gh git-lfs pre-commit`                                                                                  | Git, GH CLI, LFS, pre-commit available        |
| 3  | Generate SSH key                     | `ssh-keygen -t ed25519 -C "<mail>"` → add public key on GitHub                                                            | Password-less Git pushes                      |
| 4  | Clone & scaffold                     | `bash<br>git clone git@github.com:rvs-23/ds-de-se-hub.git<br>cd ds-de-se-hub<br>python create_structure.py`               | Decade folders (`00_`, `10_`, …) + `.gitkeep` |
| 5  | Protect `main`                       | GitHub → **Settings ▸ Branches** → Add protection rule (require pull request)                                             | Direct pushes blocked                         |
| 6  | Add baseline `.gitignore`            | `git add .gitignore` → `git commit -m "chore: add gitignore"`                                                             | OS junk and notebooks ignored                 |
| 7  | Enable Git LFS                       | `bash<br>git lfs install<br>git lfs track "*.pdf" "*.csv"<br>git add .gitattributes<br>git commit -m "chore: enable LFS"` | Binaries become pointers                      |
| 8  | Store PAT for LFS                    | Create token (scopes: `repo`, `write:packages`) → `git credential approve …`                                              | Silent LFS uploads                            |
| 9  | Populate content in feature branches | `git checkout -b feat/<topic>` → Move/rename files → `git add` → `git commit` → `git push`                                | Structured content per topic                  |
| 10 | Open pull requests                   | `gh pr create --fill` → Merge via UI                                                                                      | Reviewed, linear history                      |
| 11 | Install pre-commit hooks             | `bash<br>pre-commit sample-config > .pre-commit-config.yaml<br>pre-commit install`                                        | Auto-strip notebooks, run Black/isort         |
| 12 | Write master README                  | Copy this document into `README.md`                                                                                       | Single entry-point for the repo               |

---

## 2. Issue-Resolution Log

| Issue                                        | Attempts                                                                                          | Final solution                                                                                                                                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Infinite password loop on LFS push           | 1. Retyped GitHub password (rejected)<br>2. Switched remote to SSH (Git refs ok, LFS still loops) | • Set `credential.helper` (`osxkeychain` / `store`) <br>• Generated PAT (`repo`, `write:packages`)<br>• Stored via `git credential approve`<br>• Loaded SSH key into agent: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` |
| `git pull` blocked by uncommitted renames    | Considered committing vs ignoring                                                                 | • Used `git stash` → `git pull` → `git stash pop` → Resolved conflicts → Committed                                                                                                                                           |
| Large-file (>50 MB) warning                  | Ignored initially                                                                                 | • Adopted Git LFS<br>• Migrated existing PDFs: `git lfs migrate import --include="*.pdf"`                                                                                                                                    |
| Mac `Icon\r` artifacts & inconsistent casing | —                                                                                                 | • Added `Icon\r` to `.gitignore`<br>• Purged: `find . -name "Icon?" -delete`<br>• Batch-renamed files to snake\_case with `rename`                                                                                           |
| Should `.gitkeep` be tracked?                | Debated                                                                                           | • Keep in directories that may remain empty<br>• Delete in folders now guaranteed to contain content                                                                                                                         |

---

## 3. Final Folder Structure & Rationale

```
00_foundations/
  books/
  cv/
  statistics/books/
  ds_buzzwords_explained.png

10_programming/
  python/
    00_books/
    01_reference_pdfs/
    02_notebooks/
      basics/
      modules/{numpy,pandas,matplotlib}
      oop/
      optimisation/
      regex/
    03_scripts/
    90_playgrounds/python-functional/
    data/

20_data_science/
  21_ml_classical/{00_books,01_references,02_notebooks,data/}
  22_deep_learning/{00_books,01_references}
  23_specialisations/{nlp/,time_series/,actuarial/}

30_data_engineering/
  spark_big_data/books/

40_software_engineering/
  dsa_algorithms/
  system_design/

50_industry_and_research/
  [2022‒2024] state_of_ai_report.pdf
  00_ml_papers/

60_interview_prep/
  coding_drills/
  case_studies/
  365_ds_interview_qs.pdf

90_projects/
```

* **Decade prefixes** (`00_`, `10_`, …) provide reserved slots (`70_`, `80_`) for future tracks.
* **Standard sub-folders** (`00_books`, `01_references`, `02_notebooks`, `03_scripts`, `data`) keep each topic self-contained.
* **Git LFS** controls repo size.
* **Protected `main`** ensures every change is reviewed.

---

## 4. Artefact Type Classification

| Artefact                         | Destination                 |
| -------------------------------- | --------------------------- |
| Full textbook / long PDF         | `00_books/`                 |
| Slide deck / cheat-sheet         | `01_references/`            |
| Notebook or tutorial             | `02_notebooks/<topic>/`     |
| Reusable helper script / module  | `03_scripts/`               |
| Sample dataset                   | `data/`                     |
| One-off exploration / experiment | `90_projects/<short-name>/` |

---

## 5. Commit & PR Workflow

```bash
# 1. Create a feature branch
git checkout -b feat/<topic>-<desc>

# 2. Move/add/edit files
git add .

# 3. Commit
git commit -m "feat: add <resource> to <path>"

# 4. Run pre-commit
pre-commit run --all-files

# 5. Push and create PR
git push -u origin feat/<topic>-<desc>
gh pr create --fill

# 6. After merge
git checkout main
git pull
git branch -d feat/<topic>-<desc>
```

---

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

**Do**

* Use **feature branches** for every change—no direct commits to `main`.
* Maintain **snake\_case** filenames without spaces.
* Commit both `pyproject.toml` and its lockfile when adding dependencies via `uv`.
* Run **`pre-commit run --all-files`** before pushing.
* Track large binaries with **Git LFS**.

**Don’t**

* Push a virtual-env folder or `__pypackages__`.
* Commit personal/proprietary datasets—keep them under `data/` and ignored.
* Bypass branch-protection rules unless absolutely necessary.
* Rename top-level decade folders without updating references.

---

## 8. Other Resources Link-Board (live)

All external links and reading lists live in Notion, synced manually:
[https://www.notion.so/rv/ds-de-se-knowledge-library](https://www.notion.so/Resources-DS-Software-Eng-1fa21584c09f80b8a571f811fb8e016a?pvs=4)

---


## 9. Maintainers

**Primary:** rv (Data Science)
For questions: open an issue or email `23rishavsharma@gmail.com`.

---

*Document version: 2025-05-21*

