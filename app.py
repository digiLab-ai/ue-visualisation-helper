# app.py
import pandas as pd
import streamlit as st
from viewer import PlotlyModelViewer

st.set_page_config(page_title="UE Model Visualisation", layout="wide")

st.title("🔭 Uncertainty Engine Model Visualisation")
st.caption("Upload CSVs, pick up to 3 inputs and one output." \
            "\n Non-selected inputs are **frozen** to specified grid values.\n" \
            " Note: all validation points are overlaid if provided.\n")

# ---------------------- Uploads ----------------------
with st.sidebar:
    st.header("1) Upload CSV files")
    input_file   = st.file_uploader("test input (CSV)", type=["csv"], key="input_df")
    pred_file    = st.file_uploader("test prediction (CSV)",  type=["csv"], key="pred_df")
    unc_file     = st.file_uploader("test uncertainty (CSV, optional)", type=["csv"], key="unc_df")
    val_file     = st.file_uploader("validation (CSV, optional)", type=["csv"], key="val_df")
    valerr_file  = st.file_uploader("validation error (CSV, optional)", type=["csv"], key="val_err_df")

read_csv = lambda f: (pd.read_csv(f) if f else None)
input_df   = read_csv(input_file)
pred_df    = read_csv(pred_file)
unc_df     = read_csv(unc_file)
val_df     = read_csv(val_file)
val_err_df = read_csv(valerr_file)

if input_df is None or pred_df is None:
    st.info("⬅️ Please upload **test input** and **test prediction** to begin.")
    st.stop()
if len(input_df) != len(pred_df):
    st.error("`test input` and `test prediction` must have the **same number of rows** (aligned point-wise).")
    st.stop()

# ---------------------- Column selection ----------------------
with st.sidebar:
    st.header("2) Select inputs & output")
    input_cols = list(input_df.columns)
    output_cols = list(pred_df.columns)

    inputs = st.multiselect("Inputs (≤3)", input_cols, default=input_cols[:min(3, len(input_cols))], max_selections=3)
    if not inputs:
        st.warning("Select at least one input.")
        st.stop()
    output = st.selectbox("Output", output_cols, index=0)

# ---------------------- Frozen values for other inputs ----------------------
other_inputs = [c for c in input_cols if c not in inputs]
frozen = {}

with st.sidebar:
    st.header("3) Freeze non-selected inputs")
    if other_inputs:
        for col in other_inputs:
            options = sorted(pd.unique(input_df[col]))
            # choose middle unique value by default
            default = options[len(options)//2] if options else None
            # Select from actual grid values → always valid (no “nearest” needed)
            frozen[col] = st.select_slider(f"{col}", options=options, value=default)
    else:
        st.caption("All inputs are selected — nothing to freeze.")

# ---------------------- Appearance ----------------------
with st.sidebar:
    st.header("4) Appearance")
    mode3d = "volume"
    if len(inputs) == 3:
        mode3d = st.selectbox("3D Mode", ["volume", "isosurface"], index=0)
    vol_opacity = st.slider("Volume opacity", 0.05, 1.0, 0.15, 0.05, disabled=(len(inputs) != 3 or mode3d != "volume"))
    vol_surface = st.slider("Volume surface count", 4, 32, 12, 1, disabled=(len(inputs) != 3))

# ---------------------- Plot ----------------------
viewer = PlotlyModelViewer(input_df, pred_df, unc_df=unc_df, val_df=val_df, val_err_df=val_err_df)
download_config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": f"{output}_plot",
        # "height": 1600,
        # "width": 2400,
        "scale": 10,
    },
    "displaylogo": False
}

try:
    fig = viewer.build_figure(
        inputs=inputs, output=output, frozen=frozen,
        mode3d=mode3d, vol_opacity=vol_opacity, vol_surface_count=vol_surface
    )
    st.plotly_chart(fig, use_container_width=True, config=download_config)
except Exception as e:
    st.error(f"Plotting failed: {e}")
