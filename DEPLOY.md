# Deployment runbook

Step-by-step guide to verify the project locally and ship the two live demos:
the **dashboard** (Streamlit Community Cloud) and the **documentation** (GitHub
Pages), with CI running on every push.

The repo is already a Git repository with remote `git@github.com:AVova/bmmm.git`
on branch `main`.

---

## 1. Verify locally before pushing

This is exactly what CI checks, so run it first to keep CI green.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest                 # all tests, including the slow MCMC smoke test
```

Build the docs the same way Pages will:

```bash
uv sync --group docs
uv run mkdocs build --strict  # fails on broken links/refs -> catches Pages errors early
```

---

## 2. Docker (optional locally, validated by CI)

You do not have Docker installed locally; CI builds the image on every push, so
this section is optional. If you want to run the services in containers:

**Install Docker**

- Windows / macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux: Docker Engine + the Compose plugin
  (<https://docs.docker.com/engine/install/>)

Check it works:

```bash
docker --version
docker compose version
```

**Build and run**

```bash
docker build -t bmmm .        # same build CI runs
docker compose up             # API on :8000, dashboard on :8501
```

- Dashboard: <http://localhost:8501> (works out of the box; assets are baked in).
- API docs: <http://localhost:8000/docs>.

The API needs a trained model bundle. The 92 MB `artifacts/mmm.nc` is **not**
committed or baked into the image; `docker-compose.yml` mounts your local
`./artifacts` into the container. Produce the bundle once with:

```bash
uv run bmmm train             # writes artifacts/mmm.nc + data.csv + metadata.json
```

Without it the API still starts and `/health` reports `model_loaded: false`.

Run only the dashboard (no model needed):

```bash
docker compose up dashboard
```

---

## 3. Push to GitHub

```bash
git add -A
git commit -m "feat: docker, CI/CD, GitHub Pages docs"
git push
```

The `.gitignore` already excludes `artifacts/*.nc` (the 92 MB model) and
`artifacts/*.json`. The small `artifacts/data.csv` and the dashboard assets are
committed on purpose.

---

## 4. Enable CI/CD (GitHub Actions)

Workflows live in `.github/workflows/` and run automatically once pushed:

- `ci.yml` - lint, type-check, tests (incl. a tiny MCMC fit) and a Docker build.
- `docs.yml` - builds the MkDocs site and publishes it to Pages.

If Actions are disabled on the repo: open the **Actions** tab and click
**"I understand my workflows, enable them"**. After the first push, watch the run
status in the **Actions** tab.

---

## 5. Publish the docs to GitHub Pages (live link #2)

One-time setup so `docs.yml` can publish:

1. Repo **Settings -> Pages**.
2. Under **Build and deployment -> Source**, choose **GitHub Actions**
   (not "Deploy from a branch").

On the next push to `main`, `docs.yml` deploys the site to:

```
https://avova.github.io/bmmm/
```

The URL also appears in the **Actions** run (the deploy step) and back on
**Settings -> Pages**.

---

## 6. Deploy the dashboard to Streamlit Community Cloud (live link #1)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** -> select repo **`AVova/bmmm`**, branch **`main`**.
3. Set **Main file path** to **`dashboard/app.py`**.
4. Deploy.

Streamlit installs from `dashboard/requirements.txt` (lightweight: streamlit,
plotly, numpy - no PyMC) because it sits next to the main file. The app reads the
committed `dashboard/assets/` and never loads the heavy model.

After it goes live, copy the app URL into:

- `README.md` - the "Live dashboard" line at the top.

---

## 7. Verify the live demos

- Streamlit URL opens; tabs, sliders and the scenario planner respond.
- Pages URL serves the documentation.
- The CI badge in `README.md` is green.

---

## What ends up where

| Artifact            | Where it runs                | Needs the 92 MB model? |
| ------------------- | ---------------------------- | ---------------------- |
| Dashboard           | Streamlit Community Cloud    | No (compact assets)    |
| Documentation       | GitHub Pages                 | No                     |
| API + dashboard     | Local Docker (`compose up`)  | API yes, dashboard no  |
| CI (tests + build)  | GitHub Actions               | No (trains a tiny one) |
