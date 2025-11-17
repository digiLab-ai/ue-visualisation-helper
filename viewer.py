# viewer.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Brand colours
INDIGO  = "#16425B"
KEPPEL  = "#16D5C2"
KEPPEL_50 = "#8AEAE1"
KEPPEL_20 = "#D0F7F3"
KEYLIME = "#EBF38B"
CUSTOM_SCALE = [[0.0, INDIGO], [0.5, KEPPEL], [1.0, KEYLIME]]

class PlotlyModelViewer:
    """
    Core plotting class for 1D / 2D / 3D visualisation of model predictions with slicing.

    Data expectations
    -----------------
    input_df : rows are evaluation points; columns are continuous input variables (meshgrid-flattened)
    pred_df  : same number of rows; each column is an output prediction
    unc_df   : (optional) same rows/columns as pred_df; 1σ values
    val_df   : (optional) scattered validation points; must include chosen inputs and chosen output
    val_err_df: (optional) scattered validation errors for chosen output, with a column named
                either "<output>_err" (percentage) or "<output>"
    """

    def __init__(self, input_df: pd.DataFrame, pred_df: pd.DataFrame,
                 unc_df: pd.DataFrame | None = None,
                 val_df: pd.DataFrame | None = None,
                 val_err_df: pd.DataFrame | None = None,
                 slice_validation: bool = False):
        self.input_df   = input_df.copy()
        self.pred_df    = pred_df.copy()
        self.unc_df     = unc_df.copy() if unc_df is not None else None
        self.val_df     = val_df.copy() if val_df is not None else None
        self.val_err_df = val_err_df.copy() if val_err_df is not None else None
        self._slice_validation = slice_validation

    # ---------------------- public API ----------------------
    def build_figure(self, inputs: list[str], output: str,
                     frozen: dict[str, float] | None = None,
                     mode3d: str = "isosurface",
                     vol_opacity: float = 0.15,
                     vol_surface_count: int = 12,
                     title: str | None = None,
                     value_range: tuple[float, float] | None = None,
                     uncert_range: tuple[float, float] | None = None,
                     uncert_mode: str = "percentage") -> go.Figure:
        """Return a Plotly figure for the given selection, applying slicing for non-selected inputs."""
        if not (1 <= len(inputs) <= 3):
            raise ValueError("Select 1, 2, or 3 inputs.")
        for c in inputs:
            if c not in self.input_df.columns:
                raise KeyError(f"Input '{c}' not in input_df.")
        if output not in self.pred_df.columns:
            raise KeyError(f"Output '{output}' not in pred_df.")
        if uncert_mode not in {"percentage", "absolute"}:
            raise ValueError("uncert_mode must be 'percentage' or 'absolute'.")
        if value_range is not None and value_range[0] >= value_range[1]:
            raise ValueError("value_range min must be less than max.")
        if uncert_range is not None and uncert_range[0] >= uncert_range[1]:
            raise ValueError("uncert_range min must be less than max.")

        # Determine frozen values for all *other* inputs
        all_inputs = list(self.input_df.columns)
        frozen = dict(frozen or {})
        for col in all_inputs:
            if col not in inputs:
                if col not in frozen:
                    # default to the median unique grid value
                    u = np.unique(self.input_df[col].to_numpy())
                    frozen[col] = u[len(u)//2] if len(u) else np.nan

        # Apply slice to main tables (meshgrid-flattened)
        inp_s, pred_s, unc_s = self._apply_slice_to_main(inputs, frozen)
        if inp_s.empty:
            frozen_desc = ", ".join(f"{k}={v!r}" for k, v in frozen.items()) or "none"
            raise ValueError(
                "No rows remain after applying the frozen inputs "
                f"({frozen_desc}). Pick different frozen values or upload a grid "
                "that contains these combinations."
            )

        # Apply slice/filter to validation tables (scattered) with tolerance
        # val_s, val_err_s = self._apply_slice_to_validation(inputs, frozen)

        if self._slice_validation:
            val_s, val_err_s = self._apply_slice_to_validation(inputs, frozen or {})
        else:
            val_s  = self.val_df.copy()     if self.val_df is not None else None
            val_err_s = self.val_err_df.copy() if self.val_err_df is not None else None

        # Replace working views
        self._inp_view  = inp_s
        self._pred_view = pred_s
        self._unc_view  = unc_s
        self._val_view  = val_s
        self._valerr_view = val_err_s

        if len(inputs) == 1:
            return self._fig_1d(inputs[0], output, title, value_range)
        if len(inputs) == 2:
            return self._fig_2d(inputs, output, title, value_range, uncert_range, uncert_mode)
        return self._fig_3d(inputs, output, mode3d, vol_opacity, vol_surface_count, title, value_range, uncert_range, uncert_mode)

    # ---------------------- slicing helpers ----------------------
    @staticmethod
    def _nearest_value_in_grid(values: np.ndarray, target: float) -> float:
        if values.size == 0:
            return float(target)
        idx = np.nanargmin(np.abs(values - target))
        return float(values[idx])

    def _column_tol(self, series: pd.Series) -> float:
        # half the minimum positive spacing, or a tiny fallback
        u = np.unique(series.to_numpy())
        if u.size >= 2:
            diffs = np.diff(np.sort(u))
            step = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
            return max(step * 0.51, 1e-12)
        return 1e-12

    def _apply_slice_to_main(self, inputs: list[str], frozen: dict[str, float]):
        mask = np.ones(len(self.input_df), dtype=bool)
        for col, target in frozen.items():
            grid_vals = np.unique(self.input_df[col].to_numpy())
            snapped = self._nearest_value_in_grid(grid_vals, float(target))
            tol = self._column_tol(self.input_df[col])
            mask &= np.isclose(self.input_df[col].to_numpy(), snapped, atol=tol, rtol=0.0)

        inp_s  = self.input_df.loc[mask].reset_index(drop=True)
        pred_s = self.pred_df.loc[mask].reset_index(drop=True)
        unc_s  = self.unc_df.loc[mask].reset_index(drop=True) if self.unc_df is not None else None
        return inp_s, pred_s, unc_s

    def _apply_slice_to_validation(self, inputs: list[str], frozen: dict[str, float]):
        def filter_val(df):
            if df is None:
                return None
            mask = np.ones(len(df), dtype=bool)
            for col, target in frozen.items():
                if col in df.columns:
                    tol = self._column_tol(self.input_df[col])  # use grid spacing for tolerance
                    mask &= np.isfinite(df[col].to_numpy())
                    mask &= np.abs(df[col].to_numpy() - float(target)) <= tol
            return df.loc[mask].reset_index(drop=True)

        return filter_val(self.val_df), filter_val(self.val_err_df)

    # ---------------------- utilities ----------------------
    @staticmethod
    def _grid_2d(x, y, v):
        """Reshape flattened (x,y)->v onto a rectilinear grid inferred from unique coordinates."""
        x = np.asarray(x); y = np.asarray(y); v = np.asarray(v)
        xu, yu = np.unique(x), np.unique(y)
        nx, ny = len(xu), len(yu)
        idx = np.lexsort((y, x))
        try:
            Z = v[idx].reshape(nx, ny)
        except Exception as e:
            raise ValueError("Inputs do not form a regular 2D grid (x,y) after slicing.") from e
        return xu, yu, Z

    @staticmethod
    def _pct_unc(val, sig):
        val = np.asarray(val, float); sig = np.asarray(sig, float)
        eps = max(1e-12, 1e-6 * (np.nanmax(np.abs(val)) or 1.0))
        return 100.0 * sig / np.maximum(np.abs(val), eps)

    def _uncertainty_field(self, out: str, mode: str):
        if self._unc_view is None or out not in self._unc_view.columns:
            return None, None
        sig = np.asarray(self._unc_view[out], float)
        if mode == "percentage":
            data = self._pct_unc(self._pred_view[out], sig)
            label = "Uncertainty [%]"
        else:
            data = sig
            label = "Uncertainty (σ)"
        return data, label

    # ---------------------- 1D ----------------------
    def _fig_1d(self, xcol: str, out: str, title: str | None,
                value_range: tuple[float, float] | None) -> go.Figure:
        x = np.asarray(self._inp_view[xcol], float)
        y = np.asarray(self._pred_view[out], float)
        s = np.argsort(x); x, y = x[s], y[s]

        fig = go.Figure()

        # Uncertainty band (if available)
        if self._unc_view is not None and out in self._unc_view.columns:
            sig = np.asarray(self._unc_view[out], float)[s]
            scale = 1.96 # ~95% CI
            ylo, yhi = y - scale*sig, y + scale*sig
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([ylo, yhi[::-1]]),
                fill='toself', mode='lines',
                line=dict(width=0), fillcolor=KEPPEL_50,  # KEPPEL alpha
                name=f"{out} 95% CI"
            ))

        # Prediction line
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color=KEPPEL, width=2.5),
                                 name=f"{out} (prediction)"))

        # Validation overlay
        if self._val_view is not None and {xcol, out}.issubset(self._val_view.columns):
            xv = np.asarray(self._val_view[xcol], float)
            yv = np.asarray(self._val_view[out], float)
            fig.add_trace(go.Scatter(x=xv, y=yv, mode='markers',
                                     marker=dict(color=INDIGO, symbol='circle', size=10),
                                     name="Validation"))

        fig.update_layout(
            title=title or f"{out} vs {xcol}",
            xaxis_title=xcol, yaxis_title=out,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=60, r=20, t=60, b=50)
        )
        if value_range is not None:
            fig.update_yaxes(range=list(value_range))
        return fig

    # ---------------------- 2D ----------------------
    def _fig_2d(self, inputs: list[str], out: str, title: str | None,
                value_range: tuple[float, float] | None,
                uncert_range: tuple[float, float] | None,
                uncert_mode: str) -> go.Figure:
        xcol, ycol = inputs
        x = np.asarray(self._inp_view[xcol], float)
        y = np.asarray(self._inp_view[ycol], float)
        v = np.asarray(self._pred_view[out], float)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        vmin_plot, vmax_plot = (value_range if value_range is not None else (vmin, vmax))
        X, Y, Z = self._grid_2d(x, y, v)
        scatter_cmin = scatter_cmax = None
        if value_range is not None:
            scatter_cmin, scatter_cmax = value_range
        elif vmin < vmax:
            scatter_cmin, scatter_cmax = vmin, vmax

        # Uncertainty (%)
        U = None
        umin_plot = umax_plot = None
        unc_label = "Uncertainty"
        unc_values, unc_label_candidate = self._uncertainty_field(out, uncert_mode)
        if unc_values is not None:
            unc_label = unc_label_candidate or unc_label
        if unc_values is not None:
            _, _, U = self._grid_2d(x, y, unc_values)
            if U is not None:
                umin = float(np.nanmin(unc_values))
                umax = float(np.nanmax(unc_values))
                umin_plot, umax_plot = (uncert_range if uncert_range is not None else (umin, umax))

        fig = go.Figure()

        # Left: prediction heatmap
        fig.add_trace(go.Heatmap(
            x=X, y=Y, z=Z.T, coloraxis="coloraxis", zsmooth=False, name=out,
            zmin=vmin_plot if value_range is not None else None,
            zmax=vmax_plot if value_range is not None else None
        ))

        # Validation points on left
        if self._val_view is not None and {xcol, ycol, out}.issubset(self._val_view.columns):
            fig.add_trace(go.Scatter(
                x=self._val_view[xcol], y=self._val_view[ycol], mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color=self._val_view[out],     # colour by the same quantity
                    colorscale=CUSTOM_SCALE,       # reuse the same colour scale
                    cmin=scatter_cmin, cmax=scatter_cmax,
                    # line=dict(width=0., color="#000")  # thin outline if you like
                ),
                # marker=dict(symbol='circle', size=6, color=INDIGO),
                  name="Validation"
            ))

        # Right: uncertainty heatmap (if available)
        if U is not None:
            fig.add_trace(go.Heatmap(
                x=X, y=Y, z=U.T, coloraxis="coloraxis2", zsmooth=False,
                xaxis="x2", yaxis="y2", name=unc_label,
                zmin=umin_plot if uncert_range is not None else None,
                zmax=umax_plot if uncert_range is not None else None
            ))

            # Validation error overlay on right (optional)
            if self._valerr_view is not None and {xcol, ycol}.issubset(self._valerr_view.columns):
                err_col = out + "_err" if (out + "_err") in self._valerr_view.columns else (out if out in self._valerr_view.columns else None)
                if err_col:
                    fig.add_trace(go.Scatter(
                        x=self._valerr_view[xcol], y=self._valerr_view[ycol], mode="markers",
                        marker=dict(symbol='x', size=10, color=KEPPEL), name="Val error",
                        xaxis="x2", yaxis="y2"
                    ))

        # Layout with two side-by-side 2D panels
        fig.update_layout(
            title=title or f"{out} — 2D: {xcol} vs {ycol}",
            xaxis=dict(domain=[0.05, 0.48], title=xcol),
            yaxis=dict(
                title=ycol,
                side="left",        # ⬅️ put left-hand y ticks on the left
                ticks="outside",    # nice outward ticks
                mirror=False,
            ),
            xaxis2=dict(domain=[0.52, 0.95], title=xcol, matches="x"),
            yaxis2=dict(
                title=ycol,
                side="right",       # ⬅️ put right-hand y ticks on the right
                ticks="outside",
                mirror=False,
                matches="y",        # share limits with left subplot
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=50, r=50, t=80, b=40),
            # Color scales & top colorbars
            coloraxis=dict(
                colorscale=CUSTOM_SCALE,
                cmin=vmin_plot if value_range is not None else None,
                cmax=vmax_plot if value_range is not None else None,
                colorbar=dict(
                    title=dict(text=out, side="top"),
                    orientation="h",
                    x=0.265, y=1.08, xanchor="center",  # centered over left subplot (mid of 0.05..0.48)
                    len=0.36, thickness=14
                ),
            ),
            coloraxis2=dict(
                colorscale=CUSTOM_SCALE,
                cmin=umin_plot if uncert_range is not None else None,
                cmax=umax_plot if uncert_range is not None else None,
                colorbar=dict(
                    title=dict(text=unc_label, side="top"),
                    orientation="h",
                    x=0.735, y=1.08, xanchor="center",  # centered over right subplot (mid of 0.52..0.95)
                    len=0.36, thickness=14
                ),
            )
        )
        return fig

    # ---------------------- 3D ----------------------
    def _fig_3d(self, inputs: list[str], out: str,
                mode3d: str, vol_opacity: float, vol_surface_count: int,
                title: str | None,
                value_range: tuple[float, float] | None,
                uncert_range: tuple[float, float] | None,
                uncert_mode: str) -> go.Figure:
        xcol, ycol, zcol = inputs
        x = np.asarray(self._inp_view[xcol], float)
        y = np.asarray(self._inp_view[ycol], float)
        z = np.asarray(self._inp_view[zcol], float)
        v = np.asarray(self._pred_view[out], float)

        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        vmin_plot, vmax_plot = (value_range if value_range is not None else (vmin, vmax))
        scatter_cmin = scatter_cmax = None
        if value_range is not None:
            scatter_cmin, scatter_cmax = value_range
        elif vmin < vmax:
            scatter_cmin, scatter_cmax = vmin, vmax

        fig = go.Figure()

        if mode3d == "volume":
            vol_kwargs = dict(
                x=x, y=y, z=z, value=v,
                isomin=vmin_plot, isomax=vmax_plot,
                colorscale=CUSTOM_SCALE, surface_count=vol_surface_count,
                opacity=vol_opacity, showscale=True,
                colorbar=dict(
                    title=dict(text=out, side="top"),
                    orientation="h",
                    x=0.25, y=1.10, xanchor="center",  # centered above scene domain [0.02..0.48]
                    len=0.40, thickness=14
                ),
                name="Prediction"
            )
            if value_range is not None:
                vol_kwargs["cmin"], vol_kwargs["cmax"] = value_range
            fig.add_trace(go.Volume(**vol_kwargs))
        else:  # isosurface
            lvls = np.linspace(vmin_plot, vmax_plot, vol_surface_count+2)[1:-1] if np.isfinite(vmin_plot) and np.isfinite(vmax_plot) and vmin_plot < vmax_plot else []
            for lv in lvls:
                fig.add_trace(go.Isosurface(
                    x=x, y=y, z=z, value=v,
                    isomin=float(lv), isomax=float(lv), surface_count=1,
                    colorscale=CUSTOM_SCALE, showscale=False, opacity=0.9, name=f"Iso {lv:.3g}"
                ))

        # Validation overlay on left scene
        if self._val_view is not None and {xcol, ycol, zcol, out}.issubset(self._val_view.columns):
            fig.add_trace(go.Scatter3d(
                x=self._val_view[xcol], y=self._val_view[ycol], z=self._val_view[zcol],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color=self._val_view[out],     # colour by the same quantity
                    colorscale=CUSTOM_SCALE,       # reuse the same colour scale
                    cmin=scatter_cmin, cmax=scatter_cmax,
                    # line=dict(width=0., color="#000")  # thin outline if you like
                ),
                name="Validation"
            ))

        # Right scene: uncertainty %
        unc_label = "Uncertainty"
        unc_values, unc_label_candidate = self._uncertainty_field(out, uncert_mode)
        if unc_values is not None:
            unc_label = unc_label_candidate or unc_label
        if unc_values is not None:
            umin, umax = float(np.nanmin(unc_values)), float(np.nanmax(unc_values))
            umin_plot, umax_plot = (uncert_range if uncert_range is not None else (umin, umax))
            unc_kwargs = dict(
                x=x, y=y, z=z, value=unc_values,
                isomin=umin_plot, isomax=umax_plot,
                colorscale=CUSTOM_SCALE, surface_count=max(6, vol_surface_count//2),
                opacity=vol_opacity,
                opacityscale=[[0,1],[1,1]],
                showscale=True,
                colorbar=dict(
                    title=dict(text=unc_label, side="top"),
                    orientation="h",
                    x=0.75, y=1.10, xanchor="center",   # centered above scene2 domain [0.52..0.98]
                    len=0.40, thickness=14
                ),
                name=unc_label, scene="scene2"
            )
            if uncert_range is not None:
                unc_kwargs["cmin"], unc_kwargs["cmax"] = uncert_range
            fig.add_trace(go.Volume(**unc_kwargs))

            # Validation error overlay on uncertainty panel (if provided)
            if self._valerr_view is not None and {xcol, ycol, zcol}.issubset(self._valerr_view.columns):
                err_col = out + "_err" if (out + "_err") in self._valerr_view.columns else (out if out in self._valerr_view.columns else None)
                if err_col:
                    fig.add_trace(go.Scatter3d(
                        x=self._valerr_view[xcol], y=self._valerr_view[ycol], z=self._valerr_view[zcol],
                        mode="markers", marker=dict(size=10, color=KEPPEL),
                        name="Val error", scene="scene2"
                    ))

        # Side-by-side 3D scenes
        fig.update_layout(
            title=title or f"{out} — 3D: {xcol} | {ycol} | {zcol} ({mode3d})",
            scene=dict(domain=dict(x=[0.02, 0.48], y=[0.0, 1.0]),
                    xaxis_title=xcol, yaxis_title=ycol, zaxis_title=zcol,
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            scene2=dict(domain=dict(x=[0.52, 0.98], y=[0.0, 1.0]),
                        xaxis_title=xcol, yaxis_title=ycol, zaxis_title=zcol,
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=20, r=20, t=80, b=20)
        )
        return fig
