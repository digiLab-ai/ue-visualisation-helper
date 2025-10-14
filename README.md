# Model Viewer App

Interactive visualisation of model predictions with optional uncertainty and validation overlays.  
Supports **1D**, **2D**, and **3D** views using Plotly.

## Data expectations

- **input_df (CSV)** – rows are evaluation points; columns are continuous input variables (e.g., `x,y,z`).
- **pred_df (CSV)** – same number of rows as `input_df`; columns are output predictions (e.g., `c`, `c2`).
- **unc_df (optional, CSV)** – same rows/columns as `pred_df`; predictive **1σ**.
- **val_df (optional, CSV)** – scattered validation points; must include the **selected inputs** and the **selected output** column.
- **val_err_df (optional, CSV)** – scattered validation **errors**; must include the **selected inputs** and an error column named **`<output>_err`** (percentage) or **`<output>`**.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
streamlit run app.py
