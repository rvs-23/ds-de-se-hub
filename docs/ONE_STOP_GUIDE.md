# ds-de-se-hub • One-Stop Contributor Guide 🚀

A single page that answers:

1. **What’s where?** Folder layout & naming rules  
2. **How do I contribute?** Branch flow, commit style, PR checklist  
3. **Large files?** Git LFS setup  
4. **Local sanity-checks** before pushing

---

## 1. Folder map (top-level)

| Decade | Path | Purpose |
|--------|------|---------|
| `00_`  | `00_foundations/` | Maths, stats, algorithms |
| `10_`  | `10_programming/` | Python, SQL, TypeScript… |
| `20_`  | `20_data_science/` | ML classical → DL → NLP/CV/TS |
| `30_`  | `30_data_engineering/` | Spark, ETL, orchestration |
| `40_`  | `40_software_engineering/` | DSA, system design, DevOps |
| `50_`  | `50_industry_reports/` | Trend reports & papers |
| `60_`  | `60_interview_prep/` | Coding, case, behavioural |
| `90_`  | `90_projects/` | Personal experiments & playgrounds |
| `docs/`| This file, plus `resource_links.md` etc. |

> *Why decades?* Leaves `70_`, `80_` free for future tracks without renaming anything.

### File-naming rules

* Snake-case, no spaces: `hands_on_ml.pdf`, `effective_ml_workflow.ipynb`
* Heavy PDFs/data ➜ track with **Git LFS** (`*.pdf`, `*.csv` patterns)
* Delete Mac trash: `find . -name "Icon?" -delete`

---

## 2. Contribution workflow

```text
main (protected)
  ▲
  │ pull
feature / fix branch
  │ add / move / edit
  ├─ git add …
  ├─ git commit -m "type: summary"
  ├─ git push -u origin <branch>
  └─ open PR  → review → merge

