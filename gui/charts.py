"""
Plotly figure builders for the GUI.

Every chart follows one set of rules, so they read as one system:

  * FORM FIRST. A comparison where one model is the point uses EMPHASIS — the
    model in the accent, the rest in gray — never nine categorical hues.
    Categorical hues are for identity only, in the fixed palette order, and a
    chart never needs a 9th: past eight, it folds into "Other".
  * Polarity (a correlation's sign, a feature pushing a prediction up or down)
    is DIVERGING: blue and red around a neutral gray, never a hue at zero.
    Magnitude (a confusion-matrix count) is SEQUENTIAL: one hue, light to dark.
  * Thin marks: bars capped near 24px with rounded data-ends, 2px lines,
    markers of 8px or more with a 2px ring in the surface colour. Gridlines are
    solid hairlines — never dashed — and recessive.
  * Text wears ink, never a series colour.
  * Dark mode is its own set of steps, not an automatic inversion.

The palette is the validated reference palette (the one the classification
notebook already uses); run the dataviz validator before changing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Tokens ────────────────────────────────────────────────────────────────────
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#008300", "#9085e9", "#e66767"],
}
TOK = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", deemph="#b0aea6",
                  up="#e34948", down="#2a78d6", mid="#f0efec"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", deemph="#5f5e59",
                  up="#e66767", down="#3987e5", mid="#383835"),
}
# Diverging, blue <-> neutral gray <-> red. On the dark surface the poles get
# BRIGHTER towards the extremes and the midpoint sinks into the surface — the
# same rule, restepped, not the light ramp reused.
DIVERGING = {
    "light": ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f5b3b2", "#e34948", "#8f2322"],
    "dark":  ["#b7d3f6", "#5598e7", "#1c5cab", "#383835", "#a13a3a", "#e66767", "#f5b3b2"],
}
SEQUENTIAL = {
    "light": ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark":  ["#1f2630", "#15345e", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
}


def tokens(mode):
    return TOK["dark" if mode == "dark" else "light"]


def series(mode):
    return SERIES["dark" if mode == "dark" else "light"]


def accent(mode):
    return series(mode)[0]


def _scale(stops):
    n = len(stops) - 1
    return [[i / n, c] for i, c in enumerate(stops)]


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp_color(stops, t):
    """Colour at position t in [0, 1] along a list of hex stops."""
    t = float(np.clip(t, 0, 1)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    a, b = _hex_rgb(stops[i]), _hex_rgb(stops[i + 1])
    f = t - i
    return tuple(a[k] + (b[k] - a[k]) * f for k in range(3))


def _text_on(rgb, mode):
    """White or ink, whichever clears contrast on a filled cell."""
    c = [x / 255 for x in rgb]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return "#ffffff" if lum < 0.33 else "#0b0b0b"


def fmt(v, digits=4):
    """Compact number formatting: thousands commas, no false precision."""
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "—"
    v = float(v)
    a = abs(v)
    if a >= 1e6 or (0 < a < 1e-3):
        return f"{v:.3g}"
    if a >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}" if a < 1 else f"{v:,.{max(digits - len(str(int(a))), 1)}f}"


def base_layout(fig, mode, height=None, x_title=None, y_title=None, legend=None):
    """The shared chrome: transparent surface, hairline solid grid, quiet axes."""
    t = tokens(mode)
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["ink2"], size=13),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"],
                        font=dict(color=t["ink"], size=13)),
        showlegend=bool(legend),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=t["ink2"]), bgcolor="rgba(0,0,0,0)"),
    )
    axis = dict(showgrid=True, gridcolor=t["grid"], gridwidth=1, griddash="solid",
                zeroline=False, linecolor=t["axis"], linewidth=1, ticks="outside",
                tickcolor=t["axis"], tickfont=dict(color=t["muted"]),
                title_font=dict(color=t["ink2"]), automargin=True)
    fig.update_xaxes(**axis, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def _marker(color, mode, size=9):
    return dict(color=color, size=size, line=dict(color=tokens(mode)["surface"], width=2))


# =============================================================================
#  DATA TAB
# =============================================================================
def histogram(s, name, mode, bins=30):
    t = tokens(mode)
    fig = go.Figure(go.Histogram(
        x=s.dropna(), nbinsx=bins, marker=dict(color=accent(mode),
                                               line=dict(color=t["surface"], width=1)),
        hovertemplate=f"{name}: %{{x}}<br>rows: %{{y}}<extra></extra>"))
    fig.update_layout(bargap=0.02)
    return base_layout(fig, mode, 340, name, "Rows")


def box_by_group(df, value, group, order, mode, horizontal=False):
    """One box per group, a single hue: the axis labels carry the identity."""
    fig = go.Figure()
    for g in order:
        v = df.loc[df[group].astype(str) == str(g), value].dropna()
        kw = dict(x=v, name=str(g)) if horizontal else dict(y=v, name=str(g))
        fig.add_trace(go.Box(**kw, marker=dict(color=accent(mode), size=5, opacity=0.6),
                             line=dict(color=accent(mode), width=1.5),
                             fillcolor=_rgba(accent(mode), 0.12), boxpoints="outliers"))
    base_layout(fig, mode, 360, value if horizontal else str(group),
                str(group) if horizontal else value)
    fig.update_layout(showlegend=False)
    return fig


def boxplots(df, cols, mode, standardize=True):
    """All numeric features side by side, optionally on one standardised scale."""
    fig = go.Figure()
    for c in cols:
        v = df[c].dropna().astype(float)
        if standardize and v.std() > 0:
            v = (v - v.mean()) / v.std()
        fig.add_trace(go.Box(y=v, name=str(c), marker=dict(color=accent(mode), size=4, opacity=0.6),
                             line=dict(color=accent(mode), width=1.5),
                             fillcolor=_rgba(accent(mode), 0.12), boxpoints="outliers"))
    base_layout(fig, mode, 420, None, "z-score (standardised)" if standardize else "Value")
    fig.update_layout(showlegend=False)
    return fig


def correlation_heatmap(R, P, mode, stars):
    """
    Lower triangle only, diagonal removed — the diagonal is 1 by definition and
    the upper triangle repeats the lower. Diverging scale pinned to [-1, 1] so
    colour means the same thing on every dataset.
    """
    t = tokens(mode)
    k = len(R)
    Z = R.to_numpy(dtype=float).copy()
    mask = np.triu(np.ones_like(Z, dtype=bool))       # includes the diagonal
    Z[mask] = np.nan
    cols = list(R.columns)
    # Row 0 has nothing below the diagonal, and the last column nothing left of
    # it — drop both so the triangle fills the plot instead of floating in it.
    Zp, rows_lbl, cols_lbl = Z[1:, :-1], cols[1:], cols[:-1]
    stops = DIVERGING["dark" if mode == "dark" else "light"]
    fig = go.Figure(go.Heatmap(
        z=Zp, x=cols_lbl, y=rows_lbl, zmin=-1, zmax=1, colorscale=_scale(stops),
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title=dict(text="r", font=dict(color=t["ink2"])), thickness=12,
                      tickfont=dict(color=t["muted"]), outlinewidth=0),
        customdata=P.to_numpy()[1:, :-1],
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<br>p = %{customdata:.2g}<extra></extra>"))
    if k <= 16:
        for i in range(Zp.shape[0]):
            for j in range(Zp.shape[1]):
                z = Zp[i, j]
                if np.isnan(z):
                    continue
                rgb = _lerp_color(stops, (z + 1) / 2)
                fig.add_annotation(x=cols_lbl[j], y=rows_lbl[i],
                                   text=f"{z:.2f}{stars(P.iloc[i + 1, j])}",
                                   showarrow=False, font=dict(size=11, color=_text_on(rgb, mode)))
    base_layout(fig, mode, max(360, 34 * k + 120))
    fig.update_xaxes(showgrid=False, tickangle=-35)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def scatter_vs_target(df, x, y, mode):
    """Feature against target with a least-squares trend line."""
    t = tokens(mode)
    d = df[[x, y]].dropna().astype(float)
    fig = go.Figure(go.Scatter(
        x=d[x], y=d[y], mode="markers", marker=_marker(accent(mode), mode, 8),
        opacity=0.75, hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>"))
    if len(d) > 2 and d[x].nunique() > 1:
        b, a = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", hoverinfo="skip",
                                 line=dict(color=t["ink2"], width=2)))
        r = np.corrcoef(d[x], d[y])[0, 1]
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                           xanchor="left", yanchor="top", text=f"Pearson r = {r:.3f}",
                           font=dict(color=t["ink2"]))
    base_layout(fig, mode, 380, x, y)
    return fig


def category_bars(counts, mode, x_title="Rows", highlight=None):
    """Horizontal bars for category counts; one hue unless one bar is the point."""
    t = tokens(mode)
    counts = counts.sort_values()
    colors = [accent(mode) if (highlight is None or str(k) == str(highlight)) else t["deemph"]
              for k in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.values, y=[str(i) for i in counts.index], orientation="h",
        marker=dict(color=colors, cornerradius=4), text=[f"{v:,}" for v in counts.values],
        textposition="outside", textfont=dict(color=t["ink2"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    return base_layout(fig, mode, 70 + 38 * len(counts), x_title, None)


def missing_bars(df, mode):
    miss = (df.isna().mean() * 100).sort_values()
    t = tokens(mode)
    fig = go.Figure(go.Bar(
        x=miss.values, y=[str(c) for c in miss.index], orientation="h",
        marker=dict(color=accent(mode), cornerradius=4),
        text=[f"{v:.1f}%" for v in miss.values], textposition="outside",
        textfont=dict(color=t["ink2"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>"))
    fig.update_layout(bargap=0.35)
    fig = base_layout(fig, mode, 70 + 30 * len(miss), "Missing (%)", None)
    fig.update_xaxes(range=[0, max(5, miss.max() * 1.25)])
    return fig


def scatter_matrix(df, cols, mode, color_by=None, classes=None):
    """Pairwise scatter; coloured by class only when there are at most three."""
    t = tokens(mode)
    d = df[cols + ([color_by] if color_by else [])].dropna()
    dims = [dict(label=str(c), values=d[c]) for c in cols]
    fig = go.Figure()
    if color_by and classes and len(classes) <= 3:
        for i, c in enumerate(classes):
            sub = d[d[color_by].astype(str) == str(c)]
            fig.add_trace(go.Splom(dimensions=[dict(label=str(k), values=sub[k]) for k in cols],
                                   name=str(c), showupperhalf=False, diagonal_visible=False,
                                   marker=dict(color=series(mode)[i], size=5, opacity=0.7,
                                               line=dict(color=t["surface"], width=0.5))))
        legend = True
    else:
        fig.add_trace(go.Splom(dimensions=dims, showupperhalf=False, diagonal_visible=False,
                               marker=dict(color=accent(mode), size=5, opacity=0.65,
                                           line=dict(color=t["surface"], width=0.5))))
        legend = False
    base_layout(fig, mode, 150 + 130 * len(cols), legend=legend)
    return fig


# =============================================================================
#  MODEL TAB
# =============================================================================
def metric_bars(values, metric, mode, highlight, higher_is_better, best=None):
    """EMPHASIS: the selected model in the accent, every other model in gray."""
    t = tokens(mode)
    v = values.dropna().sort_values(ascending=higher_is_better)
    colors = [accent(mode) if m == highlight else t["deemph"] for m in v.index]
    labels = [f"{m}  ★" if m == best else m for m in v.index]
    fig = go.Figure(go.Bar(
        x=v.values, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[fmt(x) for x in v.values], textposition="outside",
        textfont=dict(color=t["ink2"]), cliponaxis=False,
        hovertemplate="%{y}<br>" + metric + " = %{x:.4f}<extra></extra>"))
    fig.update_layout(bargap=0.32)
    base_layout(fig, mode, 80 + 36 * len(v), metric, None)
    span = (v.max() - min(v.min(), 0)) or 1
    fig.update_xaxes(range=[min(0, v.min() - 0.05 * span), v.max() + 0.18 * span])
    return fig


def actual_vs_predicted(y, p, mode, y_tr=None, p_tr=None, target="target"):
    t = tokens(mode)
    fig = go.Figure()
    if y_tr is not None:
        fig.add_trace(go.Scatter(x=y_tr, y=p_tr, mode="markers", name="Train",
                                 marker=dict(color=t["deemph"], size=7, opacity=0.55,
                                             line=dict(color=t["surface"], width=1)),
                                 hovertemplate="train<br>actual %{x:.3f}<br>predicted %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=y, y=p, mode="markers", name="Test",
                             marker=_marker(accent(mode), mode, 9), opacity=0.85,
                             hovertemplate="test<br>actual %{x:.3f}<br>predicted %{y:.3f}<extra></extra>"))
    allv = np.concatenate([np.ravel(y), np.ravel(p)] +
                          ([np.ravel(y_tr), np.ravel(p_tr)] if y_tr is not None else []))
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = (hi - lo) * 0.04 or 1
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                             name="Perfect prediction", hoverinfo="skip",
                             line=dict(color=t["muted"], width=1)))
    base_layout(fig, mode, 500, f"Actual {target}", f"Predicted {target}",
                legend=y_tr is not None)
    # The same range on both axes, so the diagonal runs corner to corner. Not
    # a locked 1:1 aspect: in a wide panel that pads the x-axis far past the
    # data instead.
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def residuals_vs_predicted(y, p, mode, target="target"):
    t = tokens(mode)
    r = np.ravel(y) - np.ravel(p)
    fig = go.Figure(go.Scatter(x=p, y=r, mode="markers", marker=_marker(accent(mode), mode, 9),
                               opacity=0.85,
                               hovertemplate="predicted %{x:.3f}<br>residual %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1.5))
    return base_layout(fig, mode, 400, f"Predicted {target}", "Residual (actual − predicted)")


def residual_histogram(y, p, mode):
    return histogram(pd.Series(np.ravel(y) - np.ravel(p)), "Residual (actual − predicted)", mode, 30)


def error_by_range(y, p, mode, bins=6, target="target"):
    """Mean absolute error inside quantile bins of the actual value."""
    t = tokens(mode)
    d = pd.DataFrame({"y": np.ravel(y), "e": np.abs(np.ravel(y) - np.ravel(p))})
    d["bin"] = pd.qcut(d["y"], q=min(bins, d["y"].nunique()), duplicates="drop")
    g = d.groupby("bin", observed=True)["e"].agg(["mean", "size"]).reset_index()
    labels = [f"{fmt(iv.left)} – {fmt(iv.right)}" for iv in g["bin"]]
    fig = go.Figure(go.Bar(x=labels, y=g["mean"], marker=dict(color=accent(mode), cornerradius=4),
                           text=[fmt(v) for v in g["mean"]], textposition="outside",
                           textfont=dict(color=t["ink2"]), cliponaxis=False,
                           customdata=g["size"],
                           hovertemplate="%{x}<br>MAE %{y:.3f}<br>rows %{customdata}<extra></extra>"))
    fig.update_layout(bargap=0.45)
    return base_layout(fig, mode, 380, f"Actual {target} (quantile bins)", "Mean absolute error")


def importance_bars(imp, mode, score_name):
    t = tokens(mode)
    d = imp.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=d["importance"], y=d["feature"].astype(str), orientation="h",
        error_x=dict(type="data", array=d["std"], color=t["ink2"], thickness=1.2, width=4),
        marker=dict(color=accent(mode), cornerradius=4),
        hovertemplate="%{y}<br>drop %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    fig.update_layout(bargap=0.35)
    return base_layout(fig, mode, 80 + 34 * len(d), f"Drop in test {score_name} when shuffled", None)


def learning_curve_chart(lc, mode):
    s = series(mode)
    fig = go.Figure()
    for col, name, color in (("train", "Training score", s[1]), ("cv", "Cross-validated score", s[0])):
        m, sd = lc[f"{col}_mean"], lc[f"{col}_std"]
        fig.add_trace(go.Scatter(x=np.r_[lc.train_size, lc.train_size[::-1]],
                                 y=np.r_[m + sd, (m - sd)[::-1]], fill="toself",
                                 fillcolor=_rgba(color, 0.10), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=lc.train_size, y=m, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2), marker=_marker(color, mode, 8),
                                 hovertemplate=f"{name}<br>%{{x}} rows: %{{y:.4f}}<extra></extra>"))
    return base_layout(fig, mode, 400, "Training rows", lc.attrs.get("score", "score"), legend=True)


def taylor_diagram(stats_df, mode, highlight):
    """
    Taylor diagram: correlation as the angle, SD ratio as the radius, so the
    distance to the reference point (R = 1, SD ratio = 1) is the centred RMS
    error. Emphasis again — the selected model in the accent, others gray.
    """
    t = tokens(mode)
    theta = np.degrees(np.arccos(np.clip(stats_df["R"], -1, 1)))
    rmax = max(1.5, float(stats_df["SD ratio"].max()) * 1.15)
    fig = go.Figure()
    for crms in (0.25, 0.5, 0.75, 1.0):        # centred-RMS arcs about the reference
        ang = np.linspace(0, np.pi, 200)
        x, y = 1 + crms * np.cos(ang), crms * np.sin(ang)
        r, th = np.hypot(x, y), np.degrees(np.arctan2(y, x))
        ok = (r <= rmax) & (th <= (180 if (stats_df["R"] < 0).any() else 90))
        fig.add_trace(go.Scatterpolar(r=r[ok], theta=th[ok], mode="lines", hoverinfo="skip",
                                      line=dict(color=t["grid"], width=1), showlegend=False))
    fig.add_trace(go.Scatterpolar(r=[1], theta=[0], mode="markers", name="Observed",
                                  marker=dict(color=t["ink"], size=11, symbol="star"),
                                  hovertemplate="Observed (reference)<extra></extra>"))
    for name, r_val, sd_val, th in zip(stats_df.index, stats_df["R"], stats_df["SD ratio"], theta):
        hi = name == highlight
        fig.add_trace(go.Scatterpolar(
            r=[sd_val], theta=[th], mode="markers+text" if hi else "markers", name=name,
            text=[name] if hi else None, textposition="top center",
            textfont=dict(color=t["ink"]),
            marker=_marker(accent(mode) if hi else t["deemph"], mode, 12 if hi else 9),
            hovertemplate=f"{name}<br>R = {r_val:.3f}<br>SD ratio = {sd_val:.3f}<extra></extra>",
            showlegend=False))
    fig.update_layout(polar=dict(
        sector=[0, 180 if (stats_df["R"] < 0).any() else 90], bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, rmax], gridcolor=t["grid"], linecolor=t["axis"],
                        showticklabels=True, ticks="outside", angle=0, tickangle=0,
                        tickfont=dict(color=t["muted"]),
                        title=dict(text="SD ratio (model / actual)",
                                   font=dict(color=t["ink2"]))),
        angularaxis=dict(gridcolor=t["grid"], linecolor=t["axis"], tickfont=dict(color=t["muted"]),
                         tickvals=[np.degrees(np.arccos(r)) for r in (0, .5, .7, .8, .9, .95, .99)],
                         ticktext=["0", "0.5", "0.7", "0.8", "0.9", "0.95", "0.99"])))
    return base_layout(fig, mode, 520)


def confusion_matrix_chart(cm, classes, mode, normalized):
    t = tokens(mode)
    stops = SEQUENTIAL["dark" if mode == "dark" else "light"]
    Z = cm.astype(float)
    if normalized:
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.nan_to_num(Z / Z.sum(axis=1, keepdims=True)) * 100
    vmax = 100 if normalized else max(Z.max(), 1)
    lbl = [str(c) for c in classes]
    fig = go.Figure(go.Heatmap(
        z=Z, x=lbl, y=lbl, zmin=0, zmax=vmax, colorscale=_scale(stops), xgap=2, ygap=2,
        colorbar=dict(title=dict(text="% of row" if normalized else "rows",
                                 font=dict(color=t["ink2"])),
                      thickness=12, tickfont=dict(color=t["muted"]), outlinewidth=0),
        customdata=cm,
        hovertemplate="true %{y} → predicted %{x}<br>%{customdata:,} rows"
                      + ("<br>%{z:.1f}% of the true class" if normalized else "") + "<extra></extra>"))
    K = len(classes)
    if K <= 12:
        for i in range(K):
            for j in range(K):
                rgb = _lerp_color(stops, Z[i, j] / vmax)
                txt = f"{Z[i, j]:.1f}%" if normalized else f"{int(cm[i, j]):,}"
                fig.add_annotation(x=lbl[j], y=lbl[i], text=txt, showarrow=False,
                                   font=dict(size=13 if K <= 6 else 10, color=_text_on(rgb, mode)))
    base_layout(fig, mode, max(340, 60 * K + 140), "Predicted class", "True class")
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_yaxes(showgrid=False, type="category", autorange="reversed")
    return fig


def roc_chart(curves, mode):
    """curves: [(label, fpr, tpr, auc, is_average)]"""
    t, s = tokens(mode), series(mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                             line=dict(color=t["muted"], width=1), showlegend=False))
    for i, (label, fpr, tpr, auc, avg) in enumerate(curves):
        color = t["ink"] if avg else s[i % 8]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC {auc:.3f})",
                                 line=dict(color=color, width=2.5 if avg else 2),
                                 hovertemplate=f"{label}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>"))
    fig = base_layout(fig, mode, 440, "False positive rate", "True positive rate",
                      legend=len(curves) > 1)
    fig.update_xaxes(range=[-0.01, 1.01]); fig.update_yaxes(range=[-0.01, 1.01])
    return fig


def pr_chart(curves, mode):
    """curves: [(label, recall, precision, ap, prevalence, is_average)]"""
    t, s = tokens(mode), series(mode)
    fig = go.Figure()
    for i, (label, rec, prec, ap, prev, avg) in enumerate(curves):
        color = t["ink"] if avg else s[i % 8]
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{label} (AP {ap:.3f})",
                                 line=dict(color=color, width=2.5 if avg else 2, shape="hv"),
                                 hovertemplate=f"{label}<br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>"))
        if len(curves) == 1 and prev is not None:
            fig.add_hline(y=prev, line=dict(color=t["muted"], width=1),
                          annotation_text=f"no-skill = prevalence {prev:.2f}",
                          annotation_font_color=t["ink2"], annotation_position="bottom left")
    fig = base_layout(fig, mode, 440, "Recall", "Precision", legend=len(curves) > 1)
    fig.update_xaxes(range=[-0.01, 1.01]); fig.update_yaxes(range=[-0.01, 1.03])
    return fig


def calibration_chart(curves, mode):
    """curves: [(label, mean_predicted, fraction_positive)]"""
    t, s = tokens(mode), series(mode)
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                               line=dict(color=t["muted"], width=1), hoverinfo="skip"))
    for i, (label, mp, fp) in enumerate(curves):
        fig.add_trace(go.Scatter(x=mp, y=fp, mode="lines+markers", name=str(label),
                                 line=dict(color=s[i % 8], width=2), marker=_marker(s[i % 8], mode, 8),
                                 hovertemplate=f"{label}<br>predicted %{{x:.3f}}<br>observed %{{y:.3f}}<extra></extra>"))
    fig = base_layout(fig, mode, 440, "Mean predicted probability", "Observed frequency",
                      legend=True)
    fig.update_xaxes(range=[-0.01, 1.01]); fig.update_yaxes(range=[-0.01, 1.01])
    return fig


def threshold_chart(th, prec, rec, f1, best_t, mode):
    t, s = tokens(mode), series(mode)
    fig = go.Figure()
    for y, name, c in ((prec, "Precision", s[0]), (rec, "Recall", s[1]), (f1, "F1", s[2])):
        fig.add_trace(go.Scatter(x=th, y=y, mode="lines", name=name, line=dict(color=c, width=2),
                                 hovertemplate=f"{name} %{{y:.3f}}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=t["muted"], width=1),
                  annotation_text="0.5 default", annotation_font_color=t["ink2"],
                  annotation_position="bottom right")
    fig.add_vline(x=best_t, line=dict(color=t["ink"], width=1.5),
                  annotation_text=f"best F1 at {best_t:.2f}", annotation_font_color=t["ink"],
                  annotation_position="top left")
    fig = base_layout(fig, mode, 420, "Decision threshold on P(positive)", "Score", legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1.02])
    return fig


def density_lines(groups, mode, x_title, bins=30):
    """groups: [(label, values)] -> one 2px density line each, 10% wash beneath."""
    s = series(mode)
    fig = go.Figure()
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for i, (label, v) in enumerate(groups[:8]):
        h, _ = np.histogram(np.clip(v, 0, 1), bins=edges, density=True)
        fig.add_trace(go.Scatter(x=centers, y=h, mode="lines", name=str(label),
                                 line=dict(color=s[i], width=2, shape="spline", smoothing=0.6),
                                 fill="tozeroy", fillcolor=_rgba(s[i], 0.10),
                                 hovertemplate=f"{label}<br>%{{x:.2f}}: density %{{y:.2f}}<extra></extra>"))
    fig = base_layout(fig, mode, 400, x_title, "Density", legend=len(groups) > 1)
    fig.update_xaxes(range=[0, 1])
    return fig


def per_class_bars(report, metric, mode):
    t = tokens(mode)
    d = report[metric]
    fig = go.Figure(go.Bar(
        x=d.values, y=[str(i) for i in d.index], orientation="h",
        marker=dict(color=accent(mode), cornerradius=4), text=[f"{v:.3f}" for v in d.values],
        textposition="outside", textfont=dict(color=t["ink2"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    fig = base_layout(fig, mode, 80 + 40 * len(d), metric, None)
    fig.update_xaxes(range=[0, 1.12])
    return fig


# =============================================================================
#  PREDICT TAB
# =============================================================================
def shapley_waterfall(labels, phi, base, final, mode, value_label, top=10):
    """
    The ledger, read top to bottom: start at the typical-input prediction,
    each feature pushes it up (red) or down (blue), end exactly at this
    prediction. Past `top` features the small ones fold into one "Other
    features" row, so the chart stays readable without losing any of the total.

    The axis is zoomed to the path the prediction actually travels. Waterfall
    totals are drawn from zero, and an axis that included zero would squash a
    few-percent contribution into a sliver beside two enormous total bars; the
    totals are allowed to run off the left edge instead, which is the usual
    convention for a zoomed waterfall.
    """
    t = tokens(mode)
    order = np.argsort(np.abs(phi))[::-1]
    lab, val = [labels[i] for i in order[:top]], [float(phi[i]) for i in order[:top]]
    # One decimal scale for every contribution, set by the largest: +12.44 and
    # -0.19 read as a ledger; +12.44 and -0.1871 read as noise.
    biggest = max((abs(v) for v in val), default=1.0) or 1.0
    dec = int(np.clip(2 - np.floor(np.log10(biggest)), 0, 6))
    if len(order) > top:
        lab.append(f"Other {len(order) - top} features")
        val.append(float(np.sum(phi[order[top:]])))
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(val) + ["total"],
        y=["Typical input"] + lab + ["This prediction"],
        x=[base] + val + [final],
        text=[fmt(base)] + [("+" if v >= 0 else "−") + f"{abs(v):,.{dec}f}" for v in val]
             + [fmt(final)],
        textposition="outside", textfont=dict(color=t["ink2"]), cliponaxis=False,
        increasing=dict(marker=dict(color=t["up"])),
        decreasing=dict(marker=dict(color=t["down"])),
        totals=dict(marker=dict(color=t["ink2"])),
        connector=dict(line=dict(color=t["axis"], width=1)),
        hovertemplate="%{y}<br>%{text}<extra></extra>"))
    fig.update_layout(waterfallgap=0.35)
    path = base + np.concatenate([[0.0], np.cumsum(val)])
    lo, hi = float(min(path.min(), final)), float(max(path.max(), final))
    span = (hi - lo) or (abs(final) * 0.1) or 1.0
    fig = base_layout(fig, mode, 110 + 34 * (len(lab) + 2), value_label, None)
    fig.update_xaxes(range=[lo - 0.45 * span, hi + 0.45 * span])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def what_if_chart(grid, out, current, feature, mode, info, task, classes=None,
                  halfwidth=None, focus_class=None, target="prediction"):
    t, s = tokens(mode), series(mode)
    fig = go.Figure()
    numeric = info["kind"] == "numeric"
    if numeric:
        # the training range as a quiet band: outside it the curve is extrapolation
        fig.add_vrect(x0=info["min"], x1=info["max"], fillcolor=_rgba(t["muted"], 0.08),
                      line_width=0, layer="below")
    if task == "regression":
        y = np.ravel(out)
        if halfwidth is not None and np.isfinite(halfwidth) and numeric:
            fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]],
                                     y=np.r_[y + halfwidth, (y - halfwidth)[::-1]],
                                     fill="toself", fillcolor=_rgba(accent(mode), 0.10),
                                     line=dict(width=0), hoverinfo="skip", name="90% interval"))
        if numeric:
            fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", name=target,
                                     line=dict(color=accent(mode), width=2),
                                     hovertemplate=f"{feature} = %{{x}}<br>{target} %{{y:.3f}}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=[str(g) for g in grid], y=y, mode="markers", name=target,
                                     marker=_marker(accent(mode), mode, 11),
                                     hovertemplate=f"{feature} = %{{x}}<br>{target} %{{y:.3f}}<extra></extra>"))
        cur_y = float(np.interp(float(current), grid.astype(float), y)) if numeric else \
            float(y[list(map(str, grid)).index(str(current))]) if str(current) in map(str, grid) else None
        if cur_y is not None:
            fig.add_trace(go.Scatter(x=[current if numeric else str(current)], y=[cur_y],
                                     mode="markers", name="Your input",
                                     marker=_marker(t["ink"], mode, 13),
                                     hovertemplate=f"your input<br>{feature} = %{{x}}<br>{target} %{{y:.3f}}<extra></extra>"))
        return base_layout(fig, mode, 380, feature, target, legend=False)
    # classification: probability of each class along the sweep
    P = np.asarray(out)
    idx = list(range(P.shape[1])) if P.shape[1] <= 8 else [focus_class]
    for k in idx:
        xs = grid if numeric else [str(g) for g in grid]
        fig.add_trace(go.Scatter(x=xs, y=P[:, k], mode="lines" if numeric else "lines+markers",
                                 name=f"P({classes[k]})",
                                 line=dict(color=s[k % 8], width=2.5 if k == focus_class else 2),
                                 hovertemplate=f"P({classes[k]}) %{{y:.3f}}<extra></extra>"))
    if numeric:
        fig.add_vline(x=current, line=dict(color=t["ink"], width=1.5),
                      annotation_text="your input", annotation_font_color=t["ink"],
                      annotation_position="top")
    else:
        # A category axis has no numeric position to hang an annotated line on
        # (Plotly averages the x values to place the label), so the current
        # category is marked with a point on the focus class's curve instead.
        labels = [str(g) for g in grid]
        if str(current) in labels and focus_class is not None:
            fig.add_trace(go.Scatter(
                x=[str(current)], y=[P[labels.index(str(current)), focus_class]],
                mode="markers", name="Your input", marker=_marker(t["ink"], mode, 13),
                hovertemplate="your input<extra></extra>"))
    fig = base_layout(fig, mode, 380, feature, "Probability", legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, 1.02])
    return fig


def agreement_chart(preds, chosen, mode, task, halfwidth=None, x_title="Prediction",
                    hover_extra=None):
    """Every model's answer for this input, the chosen one emphasised."""
    t = tokens(mode)
    names = list(preds)
    order = sorted(names, key=lambda n: preds[n])
    fig = go.Figure()
    if halfwidth is not None and np.isfinite(halfwidth):
        c = preds[chosen]
        fig.add_vrect(x0=c - halfwidth, x1=c + halfwidth, fillcolor=_rgba(accent(mode), 0.10),
                      line_width=0, layer="below",
                      annotation_text="90% interval", annotation_font_color=t["ink2"],
                      annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[preds[n] for n in order], y=order, mode="markers",
        marker=dict(color=[accent(mode) if n == chosen else t["deemph"] for n in order],
                    size=[14 if n == chosen else 10 for n in order],
                    line=dict(color=t["surface"], width=2)),
        customdata=[(hover_extra or {}).get(n, "") for n in order],
        hovertemplate="%{y}<br>" + x_title + " %{x:.4f}%{customdata}<extra></extra>"))
    fig = base_layout(fig, mode, 70 + 32 * len(names), x_title, None)
    if task == "classification":
        fig.update_xaxes(range=[0, 1.02])
    else:
        # Frame the dots AND the interval with room to spare, so the band's
        # edges are visible — a band wider than the frame reads as background.
        xs = list(preds.values())
        if halfwidth is not None and np.isfinite(halfwidth):
            xs += [preds[chosen] - halfwidth, preds[chosen] + halfwidth]
        lo, hi = min(xs), max(xs)
        pad = 0.12 * ((hi - lo) or abs(hi) or 1.0)
        fig.update_xaxes(range=[lo - pad, hi + pad])
    return fig


def class_probability_bars(P, classes, predicted, mode):
    t = tokens(mode)
    order = np.argsort(P)
    colors = [accent(mode) if k == predicted else t["deemph"] for k in order]
    fig = go.Figure(go.Bar(
        x=P[order], y=[str(classes[k]) for k in order], orientation="h",
        marker=dict(color=colors, cornerradius=4), text=[f"{P[k]:.1%}" for k in order],
        textposition="outside", textfont=dict(color=t["ink2"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    fig = base_layout(fig, mode, 80 + 40 * len(classes), "Probability", None)
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    return fig


def _rgba(hex_color, alpha):
    r, g, b = _hex_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"
