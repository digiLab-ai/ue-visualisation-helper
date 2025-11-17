# app.py
from math import prod
import inspect

import numpy as np
import pandas as pd
import streamlit as st
from viewer import PlotlyModelViewer

st.set_page_config(page_title="UE Model Visualisation", layout="wide")

st.title("🔭 Uncertainty Engine Model Visualisation")
st.caption("Upload CSVs, pick up to 3 inputs and one output." \
            "\n Non-selected inputs are **frozen** to specified grid values.\n" \
            " Note: all validation points are overlaid if provided.\n")


def _finite_bounds(values):
    """Return (min, max) of finite numeric values or None if unavailable."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float, na_value=np.nan)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def _meshgrid_summary(df: pd.DataFrame):
    dims = {}
    for col in df.columns:
        vals = pd.unique(df[col].dropna())
        dims[col] = int(len(vals))
    expected = prod(max(1, d) for d in dims.values()) if dims else 0
    looks_regular = len(df) > 0 and expected == len(df)
    return dims, expected, looks_regular


def _uncertainty_pct(pred_series: pd.Series, unc_series: pd.Series):
    """Compute % uncertainty the same way the viewer does."""
    val = pd.to_numeric(pred_series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    sig = pd.to_numeric(unc_series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite_abs = np.abs(val[np.isfinite(val)])
    scale = float(finite_abs.max()) if finite_abs.size else 1.0
    eps = max(1e-12, 1e-6 * max(scale, 1.0))
    denom = np.maximum(np.abs(val), eps)
    pct = 100.0 * sig / denom
    return pct


def _reset_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    return df.reset_index(drop=True) if df is not None else None


def _concat_unique(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined

# ---------------------- Uploads ----------------------
with st.sidebar:
    st.image("assets/digilab.png", use_container_width=True)
    st.header("1) Upload CSV files")
    input_file   = st.file_uploader("test input (CSV)", type=["csv"], key="input_df")
    pred_file    = st.file_uploader("test prediction (CSV)",  type=["csv"], key="pred_df")
    unc_file     = st.file_uploader("test uncertainty (CSV, optional)", type=["csv"], key="unc_df")
    val_input_file  = st.file_uploader("validation input (CSV, optional)", type=["csv"], key="val_input_df")
    val_output_file = st.file_uploader("validation output (CSV, optional)", type=["csv"], key="val_output_df")
    val_unc_file    = st.file_uploader("validation error (CSV, optional)", type=["csv"], key="val_unc_df")

read_csv = lambda f: (pd.read_csv(f) if f else None)
input_df      = read_csv(input_file)
pred_df       = read_csv(pred_file)
unc_df        = read_csv(unc_file)
val_input_df  = read_csv(val_input_file)
val_output_df = read_csv(val_output_file)
val_unc_df    = read_csv(val_unc_file)

if input_df is None or pred_df is None:
    st.info("⬅️ Please upload **test input** and **test prediction** to begin.")
    st.stop()
if len(input_df) != len(pred_df):
    st.error("`test input` and `test prediction` must have the **same number of rows** (aligned point-wise).")
    st.stop()

val_inputs = _reset_df(val_input_df)
val_outputs = _reset_df(val_output_df)
val_uncert_vals = _reset_df(val_unc_df)

if val_outputs is not None and val_inputs is None:
    st.error("Validation output CSV requires a validation input CSV (same row count).")
    st.stop()
if val_uncert_vals is not None and val_inputs is None:
    st.error("Validation error CSV requires a validation input CSV (same row count).")
    st.stop()

if val_inputs is not None and val_outputs is not None and len(val_inputs) != len(val_outputs):
    st.error("Validation input and output CSVs must have the same number of rows.")
    st.stop()
if val_inputs is not None and val_uncert_vals is not None and len(val_inputs) != len(val_uncert_vals):
    st.error("Validation input and uncertainty CSVs must have the same number of rows.")
    st.stop()

val_df = None
if val_inputs is not None:
    frames = [val_inputs]
    if val_outputs is not None:
        frames.append(val_outputs)
    val_df = _concat_unique(frames)
elif val_outputs is not None:
    st.error("Validation output CSV requires a validation input CSV.")
    st.stop()

val_err_df = None
if val_uncert_vals is not None:
    frames = [val_inputs, val_uncert_vals]
    frames = [f for f in frames if f is not None]
    val_err_df = _concat_unique(frames)

grid_dims, expected_points, looks_regular = _meshgrid_summary(input_df)
shape_str = " × ".join(f"{col}:{dim}" for col, dim in grid_dims.items()) or "n/a"
if looks_regular:
    st.success(f"Detected regular meshgrid with {len(grid_dims)} parameters ({shape_str}) covering {expected_points:,} points.")
else:
    st.warning(
        f"The uploaded meshgrid looks irregular. Found {len(grid_dims)} parameters ({shape_str}) "
        f"but expected {expected_points:,} points vs {len(input_df):,} rows."
    )
constant_cols = [col for col, dim in grid_dims.items() if dim <= 1]
tunable_cols = [col for col, dim in grid_dims.items() if dim > 1]
if constant_cols:
    const_data = []
    for col in constant_cols:
        vals = pd.unique(input_df[col].dropna())
        val = vals[0] if len(vals) else "NaN"
        const_data.append(f"{col} = {val}")
    st.info("Non-tunable inputs fixed in the meshgrid: " + "; ".join(const_data))
with st.expander("Meshgrid breakdown", expanded=False):
    st.dataframe(
        pd.DataFrame({
            "Parameter": list(grid_dims.keys()),
            "Unique grid values": list(grid_dims.values())
        })
    )

# ---------------------- Column selection ----------------------
with st.sidebar:
    st.header("2) Select inputs & output")
    input_cols = tunable_cols
    if not input_cols:
        st.error("No tunable inputs detected (all input columns contain a single value).")
        st.stop()
    output_cols = list(pred_df.columns)

    inputs = st.multiselect("Inputs (≤3)", input_cols, default=input_cols[:min(3, len(input_cols))], max_selections=3)
    if not inputs:
        st.warning("Select at least one input.")
        st.stop()
    output = st.selectbox("Output", output_cols, index=0)

    value_range = None
    output_bounds = _finite_bounds(pred_df[output])
    if output_bounds is None:
        st.error(f"Output '{output}' does not contain numeric values.")
        st.stop()
    elif output_bounds[0] == output_bounds[1]:
        st.info(f"{output} is constant at {output_bounds[0]:.3g}; range inputs disabled.")
    else:
        output_min_col, output_max_col = st.columns(2)
        output_min = output_min_col.number_input(
            f"{output} min",
            value=float(output_bounds[0]),
            help="Lower bound for the plotted output axis / colour scale.",
            format="%.6g"
        )
        output_max = output_max_col.number_input(
            f"{output} max",
            value=float(output_bounds[1]),
            help="Upper bound for the plotted output axis / colour scale.",
            format="%.6g"
        )
        if output_min >= output_max:
            st.error("Output min must be strictly less than max.")
            st.stop()
        value_range = (float(output_min), float(output_max))

    uncert_range = None
    unc_mode = None
    if unc_df is not None and output in unc_df.columns:
        unc_display = st.radio(
            "Uncertainty display",
            options=["Percentage (%)", "Absolute (σ)"],
            index=1,
            horizontal=True
        )
        unc_mode = "percentage" if unc_display.startswith("Percentage") else "absolute"
        if unc_mode == "percentage":
            unc_values = _uncertainty_pct(pred_df[output], unc_df[output])
        else:
            unc_values = pd.to_numeric(unc_df[output], errors="coerce")
        uncert_bounds = _finite_bounds(unc_values)
        if uncert_bounds is None or uncert_bounds[0] == uncert_bounds[1]:
            st.info("Uncertainty values are constant; range inputs disabled.")
        else:
            label_prefix = "Uncertainty [%]" if unc_mode == "percentage" else "Uncertainty (σ)"
            unc_min_col, unc_max_col = st.columns(2)
            default_unc_min = float(uncert_bounds[0])
            default_unc_max = float(uncert_bounds[1])
            if unc_mode == "percentage":
                if default_unc_max <= 0:
                    default_unc_max = 100.0
                else:
                    default_unc_max = max(default_unc_max, 100.0)
                default_unc_min = min(default_unc_min, default_unc_max - 1e-9)
            unc_min = unc_min_col.number_input(
                f"{label_prefix} min",
                value=default_unc_min,
                help="Lower bound for the uncertainty colour scale.",
                format="%.6g"
            )
            unc_max = unc_max_col.number_input(
                f"{label_prefix} max",
                value=default_unc_max,
                help="Upper bound for the uncertainty colour scale.",
                format="%.6g"
            )
            if unc_min >= unc_max:
                st.error("Uncertainty min must be strictly less than max.")
                st.stop()
            uncert_range = (float(unc_min), float(unc_max))

# ---------------------- Frozen values for other inputs ----------------------
other_inputs = [c for c in input_cols if c not in inputs]
frozen = {}

with st.sidebar:
    st.header("3) Freeze non-selected inputs")
    if other_inputs:
        for col in other_inputs:
            options = sorted(pd.unique(input_df[col]))
            if not options:
                st.error(f"No valid grid values found for '{col}'. Please check the uploaded meshgrid file.")
                st.stop()
            # choose middle unique value by default
            default = options[len(options)//2] if options else None
            # Select from actual grid values
            frozen[col] = st.select_slider(f"{col}", options=options, value=default)
    else:
        st.caption("All inputs are selected — nothing to freeze.")

# ---------------------- Appearance ----------------------
with st.sidebar:
    st.header("4) Appearance")
    mode3d = "volume"
    if len(inputs) == 3:
        mode3d = st.selectbox("3D Mode", ["volume", "isosurface"], index=0)
    vol_opacity = st.slider("Volume opacity", 0.05, 1.0, 0.50, 0.05, disabled=(len(inputs) != 3 or mode3d != "volume"))
    vol_surface = st.slider("Volume surface count", 4, 32, 12, 1, disabled=(len(inputs) != 3))

# ---------------------- Plot ----------------------
viewer = PlotlyModelViewer(input_df, pred_df, unc_df=unc_df, val_df=val_df, val_err_df=val_err_df)
download_config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": f"{output}_plot",
        "scale": 10,
    },
    "displaylogo": False
}

if unc_mode is None:
    unc_mode = "percentage"

try:
    build_kwargs = dict(
        inputs=inputs,
        output=output,
        frozen=frozen,
        mode3d=mode3d,
        vol_opacity=vol_opacity,
        vol_surface_count=vol_surface,
        value_range=value_range,
        uncert_range=uncert_range,
        uncert_mode=unc_mode,
    )
    sig = inspect.signature(viewer.build_figure)
    allowed = {k: v for k, v in build_kwargs.items() if k in sig.parameters}
    fig = viewer.build_figure(**allowed)
    st.plotly_chart(fig, use_container_width=True, config=download_config)
except Exception as e:
    st.error(f"Plotting failed: {e}")
