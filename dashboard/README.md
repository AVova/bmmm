# BMMM Dashboard

Interactive view of a Bayesian Marketing Mix Model: parameter recovery, sales
decomposition, ROAS and an interactive budget planner.

The app is **self-contained and light**: it reads a small precomputed artifact
(`assets/dashboard.json`) and the pre-rendered figures, so it needs only
Streamlit + NumPy + Plotly and starts fast. The model itself is trained offline
in the main project with PyMC-Marketing.

## Run locally

```bash
streamlit run dashboard/app.py
```

## Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the
   repo and set **Main file path** to `dashboard/app.py`.
3. Dependencies come from `dashboard/requirements.txt` (light: no PyMC).

The artifact is regenerated from the trained model with `bmmm export-dashboard`
in the main repository.
