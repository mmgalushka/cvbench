# Contributing to CVBench

## Getting started

```bash
git clone git@github.com:mmgalushka/cvbench.git
cd cvbench
./helper.sh init          # creates .venv and installs all dependencies
source .venv/bin/activate
./helper.sh test -m 'not tf'   # run the test suite (no GPU required)
```

## Branching

- Branch off `main` for every change: `git checkout -b feat/my-feature`
- Open a pull request back to `main` when ready
- Keep branches short-lived — one feature or fix per branch

## Commit style

CVBench uses [Conventional Commits](https://www.conventionalcommits.org/). The commit
prefix determines whether and how the version number is bumped when your PR merges.

| Prefix | What it means | Version bump |
|--------|--------------|--------------|
| `feat:` | new user-facing feature | minor (0.1.0 → 0.2.0) |
| `fix:` | bug fix | patch (0.1.0 → 0.1.1) |
| `chore:` | maintenance, deps, tooling | none |
| `docs:` | documentation only | none |
| `refactor:` | internal restructure, no behaviour change | none |
| `test:` | test additions or fixes | none |
| `feat!:` or `BREAKING CHANGE:` footer | breaking API change | minor (until v1.0.0) |

**Examples:**

```
feat: add CutMix augmentation preset
fix: checkpoint pruning off-by-one on keep_last_n=1
chore: update tensorflow to 2.17
docs: expand YAML config reference for augmentation placement
test: add coverage for registry loader with missing source file
feat!: rename --from flag to --baseline in train CLI
```

The scope is optional but useful for larger projects: `feat(trainer): ...`

## What happens automatically

**On every push / PR:**
The CI pipeline runs `pytest -m 'not tf'` on GitHub Actions. A green check is required
before merging.

**On every merge to `main`:**
The release pipeline runs `cz bump`, which:
1. Reads all commits since the last release tag
2. Determines the next version number from the commit prefixes
3. Updates `version` in `pyproject.toml`
4. Appends an entry to `CHANGELOG.md`
5. Creates a git commit and tag (e.g. `v0.2.0`), then pushes both back to the repo

If no releasable commits are present (only `chore:`, `docs:`, etc.), the pipeline skips
the bump silently — no version change, no tag.

## Previewing the next version bump

Before opening a PR, you can see what version would be bumped and what the CHANGELOG
entry would look like:

```bash
./helper.sh release --dry-run
```

## Publishing to Docker Hub

Docker images are published automatically — you never need to push manually.

**How it works:**

1. Merge a `feat:` or `fix:` commit to `main`
2. The Release pipeline runs `cz bump`, creates a version tag (e.g. `v0.2.3`), and pushes it
3. The Docker job in the same pipeline immediately builds and pushes to Docker Hub:
   - `mmgalushka/cvbench:0.2.3`
   - `mmgalushka/cvbench:latest`

**To force-publish the current version without bumping** (e.g. after fixing a pipeline bug):

1. Go to GitHub → **Actions** → **Release**
2. Click **Run workflow** → leave "Push current version to Docker Hub" checked → **Run workflow**

**Required repository secrets** (GitHub → Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not your password) |

## What NOT to do

- **Do not manually edit `CHANGELOG.md`** — it is generated automatically by the pipeline.
- **Do not manually edit the `version` field in `pyproject.toml`** — the pipeline owns it.
- **Do not push directly to `main`** — always go through a PR so CI runs first.
