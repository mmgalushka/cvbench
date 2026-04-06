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

## CI pipeline — when it runs

The CI pipeline runs `pytest -m 'not tf'` automatically in two situations:

- **When you open or update a pull request** targeting `main`
- **When a commit lands directly on `main`** (e.g. the version bump commit from a release)

A green CI check is required before a PR can be merged. CI does **not** run on every
push to a feature branch — only when the PR is opened or updated against `main`.

## Versioning

CVBench uses automated versioning — you never edit version numbers or changelogs by hand.

**How the version number is determined:**

When a release is triggered, [Commitizen](https://commitizen-tools.github.io/commitizen/)
scans **all commits since the last release tag** and picks the highest applicable bump:

| Commits present since last release | Version bump |
|------------------------------------|--------------|
| Only `chore:`, `docs:`, `refactor:`, `test:` | none — release is skipped |
| At least one `fix:` | patch: `0.7.0 → 0.7.1` |
| At least one `feat:` | minor: `0.7.0 → 0.8.0` |
| At least one `feat!:` or `BREAKING CHANGE` | minor: `0.7.0 → 0.8.0` (until v1.0.0) |

This means you can accumulate many commits across multiple merged PRs — fixes, features,
chores — and the version is calculated from all of them together when you decide to release.
A single `feat:` among ten `fix:` commits still produces a minor bump.

**To preview what version would be bumped before releasing:**

```bash
./helper.sh release --dry-run
```

## Making a release

Releases are **manual** — nothing is published automatically on merge. You decide when
to release after accumulating the changes you want.

**Steps:**

1. Merge all the PRs you want to include into `main`
2. Go to GitHub → **Actions** → **Release** → **Run workflow** → **Run workflow**
3. The pipeline will:
   - Run `cz bump` to calculate the next version from all commits since the last tag
   - Update `version` in `pyproject.toml` and append an entry to `CHANGELOG.md`
   - Create a version commit and tag (e.g. `v0.8.0`) and push both to the repo
   - Create a GitHub Release at the new tag, populated with the changelog
   - Build and push the Docker image to Docker Hub:
     - `mmgalushka/cvbench:0.8.0`
     - `mmgalushka/cvbench:latest`

If there are no releasable commits (only `chore:`, `docs:`, etc.), `cz bump` skips
the bump and the Docker build is also skipped — nothing is published.

**Required repository secrets** (GitHub → Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not your password) |

## What NOT to do

- **Do not manually edit `CHANGELOG.md`** — it is generated automatically on release.
- **Do not manually edit the `version` field in `pyproject.toml`** — the release pipeline owns it.
- **Do not push directly to `main`** — always go through a PR so CI runs first.
- **Do not create tags manually** — the release pipeline creates and pushes the version tag.
