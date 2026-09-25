"""
ML Studio — a GUI for the regression and classification workflows.

    streamlit run gui/app.py          (from the repository root)

Three places, one flow:

  SIDEBAR   load data (or a trained bundle from a notebook), choose what to
            predict, train.
  PREDICT   the point of the whole app: type in each variable, get a
            prediction from the best model — with its uncertainty, the reason
            for it, and how it would change.
  DATA / MODELS   everything behind that answer: the data's statistics and
            graphs, the leaderboard, and any metric or graph for any model.

All modelling lives in engine.py and all figure-building in charts.py; this
file is layout and state only.
"""
from __future__ import annotations

import html
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charts as C   # noqa: E402
import engine as E   # noqa: E402

st.set_page_config(page_title="ML Studio", page_icon="🔮", layout="wide",
                   initial_sidebar_state="expanded")

TAB_PREDICT, TAB_DATA, TAB_MODELS = "🔮  Predict", "📊  Data explorer", "🏆  Model explorer"
ss = st.session_state


def theme_mode():
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


MODE = theme_mode()
TK = C.tokens(MODE)
ACC = C.accent(MODE)


def esc(x):
    """Column names and class labels come from the user's file: escape them."""
    return html.escape(str(x))


def show_chart(fig, key, data=None, data_label="Show the numbers behind this chart"):
    """Every chart ships with its table twin, so no value is hover-only."""
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key,
                    config={"displaylogo": False})
    if data is not None:
        with st.expander(data_label):
            st.dataframe(data, width="stretch")


# ── Global styling: the prediction hero and a few quiet adjustments ─────────
st.markdown(f"""
<style>
  .hero {{ border-radius: 16px; padding: 26px 30px 22px 30px;
           background: {C._rgba(ACC, 0.07 if MODE == "light" else 0.14)};
           border: 1px solid {C._rgba(ACC, 0.30)}; }}
  .hero-label {{ font-size: 1.0rem; color: {TK["ink2"]}; margin-bottom: 2px; }}
  .hero-value {{ font-size: 4.2rem; font-weight: 650; line-height: 1.05;
                 color: {TK["ink"]}; letter-spacing: -0.02em; word-break: break-word; }}
  .hero-sub {{ font-size: 1.05rem; color: {TK["ink"]}; margin-top: 8px; }}
  .hero-meta {{ font-size: 0.92rem; color: {TK["ink2"]}; margin-top: 12px; }}
  .hero-chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
                border: 1px solid {TK["axis"]}; margin: 6px 6px 0 0; font-size: 0.85rem;
                color: {TK["ink2"]}; }}
  .track {{ position: relative; height: 10px; border-radius: 6px; margin: 18px 0 6px 0;
            background: {"#cde2fb" if MODE == "light" else "#15345e"}; }}
  .track-band {{ position: absolute; top: 0; height: 10px; border-radius: 6px;
                 background: {C._rgba(ACC, 0.45)}; }}
  .track-fill {{ position: absolute; top: 0; left: 0; height: 10px; border-radius: 6px;
                 background: {ACC}; }}
  .track-dot {{ position: absolute; top: -5px; width: 20px; height: 20px; margin-left: -10px;
                border-radius: 50%; background: {ACC}; border: 3px solid {TK["surface"]}; }}
  .track-legend {{ display: flex; justify-content: space-between; font-size: 0.8rem;
                   color: {TK["muted"]}; }}
  .empty {{ border: 1px dashed {TK["axis"]}; border-radius: 16px; padding: 36px;
            text-align: center; color: {TK["ink2"]}; }}
  div[data-testid="stTab"] p {{ font-size: 1.05rem; }}
  div[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; }}
</style>""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR — data in, models out
# =============================================================================
def _reset_workspace():
    """New data arrived: the models trained on the old data no longer apply."""
    for k in ("ws", "bundle", "bundle_bytes", "memo"):
        ss.pop(k, None)


def _on_source_change():
    """
    Switching between upload / demo / trained-models forgets the previous
    source completely. Keeping its table around would show the demo data under
    "Upload a data file" before anything is uploaded, and keeping the loaded
    file's name would stop the same bundle from reloading on the way back.
    """
    _reset_workspace()
    for k in ("df", "df_name", "bundle_name"):
        ss.pop(k, None)


def sidebar():
    sb = st.sidebar
    sb.markdown("## 🔮 ML Studio")
    sb.caption("Regression & classification — train, explore, predict.")

    sb.markdown("### 1 · Your data")
    source = sb.radio("Start from", ["Upload a data file", "Demo dataset",
                                     "Load trained models (.pkl)"],
                      key="source", on_change=_on_source_change,
                      help="A trained-models file comes from the Save button below, "
                           "or from export_gui_bundle() at the end of either notebook.")

    if source == "Load trained models (.pkl)":
        up = sb.file_uploader("Trained models file", type=["pkl"], key="bundle_file")
        sb.caption("⚠️ Only open files you created — a .pkl file can run code when loaded.")
        if up is not None and ss.get("bundle_name") != up.name + str(up.size):
            try:
                b = E.load_bundle(up)
                ws = E.Workspace(b)
                ss.update(ws=ws, bundle=b, bundle_name=up.name + str(up.size),
                          df=_frame_from_bundle(ws), df_name=up.name, bundle_bytes=None,
                          memo={})
                ss["main_tabs"] = TAB_PREDICT
            except Exception as exc:
                sb.error(f"Could not load that file: {type(exc).__name__}: {exc}")
        if ss.get("ws") is not None:
            ws = ss["ws"]
            sb.success(f"Loaded **{len(ws.model_names)} models** · {ws.task} · "
                       f"target **{esc(ws.target)}**")
            sb.caption(f"Source: {esc(ws.source)}"
                       + (f" · saved {esc(ws.b['created'])}" if ws.b.get("created") else ""))
            for name, why in (ws.b.get("failures") or {}).items():
                sb.caption(f"⚠️ {esc(name)} was not exported: {esc(why)[:160]}")
        return

    if source == "Demo dataset":
        kind = sb.selectbox("Demo", ["Concrete strength (regression)",
                                     "Concrete grade (classification)"], key="demo_kind")
        k = "regression" if "regression" in kind else "classification"
        if ss.get("df_name") != f"demo:{k}":
            ss.update(df=E.demo_dataset(k), df_name=f"demo:{k}")
            _reset_workspace()
    else:
        up = sb.file_uploader("CSV or Excel file", type=["csv", "txt", "xlsx", "xls", "xlsm"],
                              key="data_file")
        if up is not None:
            sheet = 0
            if up.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
                sheets = E.excel_sheets(up)
                if len(sheets) > 1:
                    sheet = sb.selectbox("Sheet", sheets, key="sheet")
                up.seek(0)
            ident = f"{up.name}|{up.size}|{sheet}"
            if ss.get("df_name") != ident:
                try:
                    df = E.read_table(up, up.name, sheet)
                    df.columns = [str(c).strip() for c in df.columns]
                    ss.update(df=df, df_name=ident)
                    _reset_workspace()
                except Exception as exc:
                    sb.error(f"Could not read that file: {type(exc).__name__}: {exc}")
                    return

    df = ss.get("df")
    if df is None:
        sb.info("Upload a file to begin — or pick the demo dataset.")
        return
    sb.caption(f"**{len(df):,} rows × {df.shape[1]} columns**")

    # ── 2 · What to predict ──────────────────────────────────────────────
    sb.markdown("### 2 · What to predict")
    cols = list(df.columns)
    target = sb.selectbox("Output (target) column", cols, index=len(cols) - 1, key="target")
    auto_task, why = E.detect_task(df[target])
    task = sb.radio("Problem type", ["regression", "classification"],
                    index=0 if auto_task == "regression" else 1, horizontal=True,
                    key=f"task::{target}", format_func=str.capitalize,
                    help="Detected automatically; override it if the detection is wrong.")
    sb.caption(f"Detected: **{auto_task}** — {esc(why)}")
    feats, excluded = E.suggest_features(df, target)
    features = sb.multiselect("Input variables", [c for c in cols if c != target],
                              default=feats, key=f"features::{target}")
    if excluded:
        with sb.expander(f"{len(excluded)} column(s) left out by default"):
            for c, reason in excluded:
                st.markdown(f"- **{esc(c)}** — {esc(reason)}")
    positive = None
    if task == "classification":
        labels = sorted(df[target].dropna().astype(str).unique())
        if len(labels) == 2:
            positive = sb.selectbox("Positive class", labels, index=1, key=f"pos::{target}",
                                    help="Precision, recall and F1 are reported for this class.")
        elif len(labels) > 50:
            sb.warning(f"{len(labels)} distinct classes — is this really a class label?")

    # ── 3 · Train ─────────────────────────────────────────────────────────
    sb.markdown("### 3 · Train the models")
    budget = sb.segmented_control("Tuning depth", list(E.BUDGETS), default="Balanced",
                                  key="budget", required=True,
                                  help="Quick ≈ seconds, Balanced ≈ a minute or two, "
                                       "Thorough ≈ several minutes on ~1,000 rows.")
    roster = E.model_names(task)
    chosen = sb.multiselect("Models", roster, default=roster, key=f"models::{task}")
    c1, c2 = sb.columns(2)
    test_pct = c1.slider("Test set %", 10, 40, 20, 5, key="test_pct")
    seed = c2.number_input("Random seed", 0, 10_000, E.SEED, key="seed")
    smote = False
    if task == "classification":
        smote = sb.toggle("Balance classes with SMOTE", key="smote",
                          help="Oversamples the minority class inside each training fold "
                               "only — never the rows a model is scored on.")
    ensemble = sb.toggle("Add a voting ensemble of the top 3", value=True, key="ensemble")

    ready = bool(features) and bool(chosen)
    if sb.button("🚀  Train models", type="primary", width="stretch", disabled=not ready):
        settings = E.TrainSettings(task=task, target=target, features=list(features),
                                   budget=budget or "Balanced", test_size=test_pct / 100,
                                   seed=int(seed), models=list(chosen), smote=smote,
                                   positive_label=positive, ensemble=ensemble)
        bar = sb.progress(0.0, text="Starting…")
        trained = False
        try:
            b = E.train_workspace(df, settings, progress=lambda f, m: bar.progress(
                min(max(f, 0.0), 1.0), text=m))
            ss.update(ws=E.Workspace(b), bundle=b, bundle_bytes=None, memo={})
            ss["main_tabs"] = TAB_PREDICT
            trained = True
        except Exception as exc:
            sb.error(f"Training failed — {type(exc).__name__}: {exc}")
        finally:
            bar.empty()
        if trained:
            st.rerun()

    ws = ss.get("ws")
    if ws is not None and (ws.target != target or ws.task != task
                           or set(ws.features) != set(features)):
        sb.warning(f"The trained models predict **{esc(ws.target)}** from "
                   f"{len(ws.features)} input(s) as **{ws.task}**. Your choices above have "
                   f"changed since — press **Train models** to apply them.", icon="🔁")
    if ws is not None:
        b = ss["bundle"]
        sb.success(f"**{len(ws.model_names)} models** trained in "
                   f"{b.get('train_seconds', 0):.0f}s — best: **{esc(ws.best)}**")
        for name, why in b.get("skipped", []):
            sb.caption(f"Skipped {esc(name)}: {esc(why)}")
        for name, err in b.get("failures", {}).items():
            sb.caption(f"⚠️ {esc(name)} failed: {esc(err)[:160]}")
        if ss.get("bundle_bytes") is None and sb.button("💾 Prepare a save file", width="stretch"):
            ss["bundle_bytes"] = E.save_bundle(b)
        if ss.get("bundle_bytes") is not None:
            sb.download_button("⬇️ Download trained models (.pkl)", ss["bundle_bytes"],
                               file_name=f"trained_models_{ws.target}.pkl",
                               mime="application/octet-stream", width="stretch")


def _frame_from_bundle(ws):
    """Rebuild a data table from a bundle so the Data tab works on it too."""
    X = pd.concat([ws.X_train, ws.X_test], ignore_index=True)
    y = np.concatenate([ws.y_train, ws.y_test])
    if ws.task == "classification":
        y = np.asarray(ws.classes, dtype=object)[y.astype(int)]
    X[ws.target] = y
    return X


# =============================================================================
#  PREDICT TAB — the highlight
# =============================================================================
def _input_key(ws, f):
    return f"in::{ws.id}::{f}"


def _set_inputs(ws, row):
    for f in ws.features:
        v = row[f].iloc[0]
        if ws.info[f]["kind"] == "numeric":
            v = float(v) if pd.notna(v) else float(ws.info[f]["median"])
        else:
            v = str(v) if pd.notna(v) else ws.info[f].get("mode", "")
        ss[_input_key(ws, f)] = v


def _prefill(ws):
    choice = ss.get(f"prefill::{ws.id}")
    if choice is None or choice.startswith("Typical"):
        _set_inputs(ws, ws.baseline())
        ss.pop(f"actual::{ws.id}", None)
        return
    i = int(choice.split("#")[1].split(" ")[0]) - 1
    _set_inputs(ws, ws.X_test.iloc[[i]].reset_index(drop=True))
    # The actual value is only true of THESE inputs; keep them alongside it so
    # the hero stops claiming it the moment any field is edited.
    ss[f"actual::{ws.id}"] = (ws.y_test[i], _signature(ws))


def _signature(ws):
    return tuple(str(ss.get(_input_key(ws, f))) for f in ws.features)


def _actual_for_current_inputs(ws):
    """The test row's true value — but only while the form still holds that row."""
    rec = ss.get(f"actual::{ws.id}")
    if rec is None or rec[1] != _signature(ws):
        return None
    return rec[0]


def _numeric_widget_args(fi):
    if fi.get("integer"):
        return dict(step=1.0, format="%.0f")
    # Three significant digits of the variable's own range: a 150-500 kg/m3
    # quantity shows one decimal, a 0.25-0.75 ratio four, a 1e-4-scale
    # measurement six — instead of a fixed format that shows it as 0.000.
    span = (fi["max"] - fi["min"]) or abs(fi["max"]) or 1.0
    decimals = int(np.clip(np.ceil(-np.log10(span)) + 3, 0, 6))
    step = 10.0 ** -decimals if decimals else max(1.0, round(span / 200))
    return dict(step=float(step), format=f"%.{decimals}f")


def input_form(ws):
    for f in ws.features:
        if _input_key(ws, f) not in ss:
            _set_inputs(ws, ws.baseline())
            break
    options = ["Typical values (median / most common)"] + [
        f"Test row #{i + 1}  ·  actual = "
        + (C.fmt(ws.y_test[i]) if ws.task == "regression" else str(ws.classes[int(ws.y_test[i])]))
        for i in range(min(len(ws.y_test), 300))]
    st.selectbox("Start from", options, key=f"prefill::{ws.id}",
                 on_change=_prefill, args=(ws,),
                 help="Fill every field from a typical input, or from a real row the "
                      "model never trained on — then compare the prediction with the "
                      "actual value.")
    with st.form(f"predict_form::{ws.id}", border=True):
        n_cols = 2 if len(ws.features) <= 14 else 3
        grid = st.columns(n_cols, gap="small")
        for i, f in enumerate(ws.features):
            fi = ws.info[f]
            with grid[i % n_cols]:
                key = _input_key(ws, f)
                if fi["kind"] == "numeric":
                    st.number_input(str(f), key=key, **_numeric_widget_args(fi),
                                    help=f"Training range {C.fmt(fi['min'])} – {C.fmt(fi['max'])}"
                                         f" · median {C.fmt(fi['median'])}")
                elif fi["kind"] == "categorical":
                    opts = list(fi["categories"])
                    if ss.get(key) not in opts:
                        ss[key] = fi.get("mode", opts[0])
                    st.selectbox(str(f), opts, key=key,
                                 help=f"{len(opts)} categories seen in training")
                else:
                    st.text_input(str(f), key=key)
        st.form_submit_button("🔮  Predict", type="primary", width="stretch")
    return pd.DataFrame([{f: ss[_input_key(ws, f)] for f in ws.features}], columns=ws.features)


def _memo(key, fn):
    """Cache per workspace, so a second bundle can never serve the first one's results."""
    memo = ss.setdefault("memo", {})
    key = (ss["ws"].id,) + tuple(key)
    if key not in memo:
        if len(memo) > 500:
            memo.clear()
        memo[key] = fn()
    return memo[key]


def hero_regression(ws, model, row, pred):
    q = ws.conformal_halfwidth(model, 0.90)
    y = ws.y_train.astype(float)
    lo, hi = float(np.min(y)), float(np.max(y))
    span = (hi - lo) or 1.0
    pos = np.clip((pred - lo) / span * 100, 0, 100)
    pct = float(np.mean(y < pred) * 100)
    m = ws.metrics(model, "test")
    if q is not None and np.isfinite(q):
        a, b = pred - q, pred + q
        band = (f'<div class="track-band" style="left:{np.clip((a - lo) / span * 100, 0, 100):.2f}%;'
                f'width:{np.clip((b - a) / span * 100, 0, 100):.2f}%"></div>')
        interval = (f'90% prediction interval: <b>{C.fmt(a)} – {C.fmt(b)}</b> '
                    f'<span style="color:{TK["ink2"]}">(± {C.fmt(q)})</span>')
    else:
        band = ""
        interval = ("<span>Too few test rows for a 90% interval</span>")
    actual = _actual_for_current_inputs(ws)
    actual_html = (f'<div class="hero-sub">Actual value for this test row: <b>{C.fmt(actual)}</b> '
                   f'— prediction error {pred - actual:+.4g}</div>') if actual is not None else ""
    st.markdown(f"""
    <div class="hero" role="status" aria-live="polite">
      <div class="hero-label">Predicted {esc(ws.target)}</div>
      <div class="hero-value">{C.fmt(pred, 5)}</div>
      <div class="hero-sub">{interval}</div>
      {actual_html}
      <div class="track">{band}<div class="track-dot" style="left:{pos:.2f}%"></div></div>
      <div class="track-legend"><span>training min {C.fmt(lo)}</span>
        <span>higher than {pct:.0f}% of training rows</span><span>max {C.fmt(hi)}</span></div>
      <div class="hero-meta">
        <span class="hero-chip">Model: {esc(model)}{" ★ best" if model == ws.best else ""}</span>
        <span class="hero-chip">test R² {C.fmt(m["R2"])}</span>
        <span class="hero-chip">test RMSE {C.fmt(m["RMSE"])}</span>
      </div>
    </div>""", unsafe_allow_html=True)


def hero_classification(ws, model, row, P):
    k = int(np.argmax(P))
    conf = float(P[k])
    m = ws.metrics(model, "test")
    actual = _actual_for_current_inputs(ws)
    actual_html = ""
    if actual is not None:
        right = int(actual) == k
        actual_html = (f'<div class="hero-sub">Actual class for this test row: '
                       f'<b>{esc(ws.classes[int(actual)])}</b> — '
                       f'{"✅ correct" if right else "❌ misclassified"}</div>')
    runner = np.argsort(P)[::-1][1] if len(P) > 1 else k
    st.markdown(f"""
    <div class="hero" role="status" aria-live="polite">
      <div class="hero-label">Predicted {esc(ws.target)}</div>
      <div class="hero-value">{esc(ws.classes[k])}</div>
      <div class="hero-sub">Probability <b>{conf:.1%}</b>
        <span style="color:{TK["ink2"]}">· next most likely: {esc(ws.classes[runner])}
        ({P[runner]:.1%})</span></div>
      {actual_html}
      <div class="track"><div class="track-fill" style="width:{conf * 100:.1f}%"></div></div>
      <div class="track-legend"><span>0%</span><span>model confidence</span><span>100%</span></div>
      <div class="hero-meta">
        <span class="hero-chip">Model: {esc(model)}{" ★ best" if model == ws.best else ""}</span>
        <span class="hero-chip">test ROC-AUC {C.fmt(m["ROC_AUC"])}</span>
        <span class="hero-chip">test accuracy {m["Accuracy"]:.1%}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    return k


def trust_notes(ws, row):
    flags = ws.extrapolation(row)
    for f, msg in flags:
        st.warning(f"**{esc(f)}**: {esc(msg)} — the model is extrapolating, so treat "
                   f"this prediction with caution.", icon="⚠️")
    nov = ws.novelty(row)
    if nov is not None and not flags:
        if nov >= 99:
            st.warning(f"Every value is inside its training range, but this **combination** "
                       f"is more unusual than {nov:.0f}% of the training rows. The model has "
                       f"seen few inputs like it.", icon="🧭")
        elif nov >= 95:
            st.info(f"This combination of inputs is less typical than {nov:.0f}% of the "
                    f"training rows.", icon="🧭")


def what_if_view(ws, model, row, feat, focus):
    grid, sweep = ws.what_if(model, row, feat)
    q = ws.conformal_halfwidth(model) if ws.task == "regression" else None
    show_chart(C.what_if_chart(grid, sweep, row[feat].iloc[0], feat, MODE, ws.info[feat],
                               ws.task, ws.classes, q, focus, ws.target),
               f"whatif_chart::{ws.id}",
               pd.DataFrame(np.column_stack([grid, sweep]),
                            columns=[feat] + ([ws.target] if ws.task == "regression"
                                              else [f"P({c})" for c in ws.classes])))
    st.caption("Everything else stays at your inputs. The shaded band is the range "
               "seen in training — outside it the model is extrapolating.")


def predict_tab():
    ws = ss.get("ws")
    if ws is None:
        st.markdown(f"""
        <div class="empty">
          <div style="font-size:2.6rem">🔮</div>
          <div style="font-size:1.35rem; font-weight:600; color:{TK["ink"]}; margin:6px 0 10px">
            Predict from your own inputs</div>
          <div>1&nbsp;·&nbsp;Load a data file (or the demo) in the sidebar
            &nbsp;&nbsp;→&nbsp;&nbsp;2&nbsp;·&nbsp;choose the output column
            &nbsp;&nbsp;→&nbsp;&nbsp;3&nbsp;·&nbsp;press <b>Train models</b></div>
          <div style="margin-top:10px">Or load models you already trained in a notebook:
            <i>Start from → Load trained models (.pkl)</i></div>
        </div>""", unsafe_allow_html=True)
        return

    top = st.columns([3, 2], vertical_alignment="bottom")
    names = ws.model_names
    model = top[0].selectbox(
        "Model used for the prediction", names, index=names.index(ws.best),
        key=f"pred_model::{ws.id}",
        format_func=lambda n: f"★ {n}  (best)" if n == ws.best else n)
    top[1].caption(f"Best model chosen by {esc(ws.b.get('selection_note', 'its score'))}.")

    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("#### Enter the input variables")
        row = input_form(ws)

    with right:
        key = (model, tuple(map(str, row.iloc[0].tolist())))
        out = _memo(("pred",) + key, lambda: ws.predict_raw(model, row)[0])
        if ws.task == "regression":
            hero_regression(ws, model, row, float(out))
            focus = None
        else:
            focus = hero_classification(ws, model, row, np.asarray(out))
        trust_notes(ws, row)

        why, whatif, agree = st.tabs(["Why this prediction", "What-if", "Do the models agree?"])
        with why:
            phi, fb, fx = _memo(("shap",) + key, lambda: ws.shapley(model, row, class_index=focus))
            # The value exactly as entered — 1000.5, not a rounded "1,000".
            labels = [f"{f} = {float(row[f].iloc[0]):.6g}" if ws.info[f]["kind"] == "numeric"
                      else f"{f} = {row[f].iloc[0]}" for f in ws.features]
            what = (f"Predicted {ws.target}" if ws.task == "regression"
                    else f"P({ws.target} = {ws.classes[focus]})")
            if np.allclose(phi, 0, atol=1e-9 * max(1.0, abs(fx))):
                st.info("These inputs **are** the typical input — every variable at its median "
                        "or most common value — so there is nothing to attribute yet. Change "
                        "some values and press **Predict**, or start from a test row.", icon="💡")
            else:
                show_chart(C.shapley_waterfall(labels, phi, fb, fx, MODE, what), f"shap::{ws.id}",
                           pd.DataFrame({"input": labels, "contribution": phi})
                           .sort_values("contribution", key=np.abs, ascending=False))
            st.caption("Starts from the prediction for a **typical input** (every variable at "
                       "its median or most common value). Each bar is how much changing that "
                       "variable to your value moves the prediction — red pushes it up, blue "
                       "down — and the bars add up **exactly** to this prediction "
                       "(Shapley values against the typical input).")
        with whatif:
            order = [f for f in np.array(ws.features)[np.argsort(-np.abs(phi))]
                     if ws.info[f]["kind"] != "text"]
            feat = st.selectbox("Vary this variable", order, key=f"whatif::{ws.id}")
            if feat is None:
                st.info("None of the input variables can be swept.")
            else:
                what_if_view(ws, model, row, feat, focus)
        with agree:
            answers = _memo(("all",) + key[1:], lambda: {
                n: ws.predict_raw(n, row)[0] for n in ws.model_names})
            preds, extra = {}, {}
            for n, o in answers.items():
                if ws.task == "regression":
                    preds[n] = float(o)
                else:
                    preds[n] = float(o[focus])
                    extra[n] = f"<br>its own answer: {esc(ws.classes[int(np.argmax(o))])}"
            q = ws.conformal_halfwidth(model) if ws.task == "regression" else None
            xt = (f"Predicted {ws.target}" if ws.task == "regression"
                  else f"P({ws.target} = {ws.classes[focus]})")
            show_chart(C.agreement_chart(preds, model, MODE, ws.task, q, xt, extra),
                       f"agree::{ws.id}", pd.Series(preds, name=xt).to_frame())
            vals = np.array(list(preds.values()))
            if ws.task == "regression":
                st.caption(f"Spread across the {len(vals)} models: {C.fmt(vals.min())} – "
                           f"{C.fmt(vals.max())}. When the models disagree strongly, the input "
                           f"is somewhere the data does not pin the answer down.")
            else:
                agree_n = sum(1 for o in answers.values() if int(np.argmax(o)) == focus)
                st.caption(f"{agree_n} of {len(vals)} models predict "
                           f"**{esc(ws.classes[focus])}** for this input.")

    batch_prediction(ws, model)


def batch_prediction(ws, model):
    with st.expander("📄  Predict many rows at once from a file"):
        st.caption("Upload a CSV/Excel file with these columns: "
                   + ", ".join(f"`{f}`" for f in ws.features))
        up = st.file_uploader("File of new inputs", type=["csv", "txt", "xlsx", "xls"],
                              key=f"batch::{ws.id}")
        if up is None:
            return
        try:
            new = E.read_table(up, up.name)
            new.columns = [str(c).strip() for c in new.columns]
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        missing = [f for f in ws.features if f not in new.columns]
        if missing:
            st.error("Missing column(s): " + ", ".join(missing))
            return
        out = new.copy()
        try:
            res = ws.predict_raw(model, new[ws.features])
        except Exception as exc:
            st.error(f"Could not predict these rows — {type(exc).__name__}: {exc}. Check that "
                     f"each column holds the same kind of values as in the training data.")
            return
        if ws.task == "regression":
            q = ws.conformal_halfwidth(model)
            out[f"predicted_{ws.target}"] = res
            if q is not None and np.isfinite(q):
                out["interval_low_90"], out["interval_high_90"] = res - q, res + q
        else:
            out[f"predicted_{ws.target}"] = np.asarray(ws.classes, dtype=object)[res.argmax(1)]
            for k, c in enumerate(ws.classes):
                out[f"P({c})"] = res[:, k]
        outside = ws.out_of_range(new[ws.features])
        out["outside_training_range"] = outside
        if outside.any():
            st.warning(f"{int(outside.sum())} of {len(new)} rows have at least one value outside "
                       f"the training range (marked in the last column) — treat those "
                       f"predictions with caution.")
        st.dataframe(out, width="stretch")
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"predictions_{ws.target}.csv", mime="text/csv")


# =============================================================================
#  DATA EXPLORER TAB
# =============================================================================
DATA_VIEWS = ["Overview", "Summary statistics", "Target variable", "Distributions",
              "Box plots", "Correlation matrix", "Feature vs target", "Pairwise scatter",
              "Missing values", "Outliers"]


def data_tab():
    df = ss.get("df")
    if df is None:
        st.info("Load a data file in the sidebar to explore it.")
        return
    ws = ss.get("ws")
    if ws is not None and ws.target in df.columns:
        target, feats, task = ws.target, list(ws.features), ws.task
    else:
        target = ss.get("target") if ss.get("target") in df.columns else df.columns[-1]
        feats = ss.get(f"features::{target}") or [c for c in df.columns if c != target]
        task = ss.get(f"task::{target}") or E.detect_task(df[target])[0]
    feats = [c for c in feats if c in df.columns]
    numeric = [c for c in feats if E._is_numeric(df[c])]

    view = st.selectbox("Show", DATA_VIEWS, key="data_view",
                        help="Statistics and graphs for the loaded data")
    if view == "Overview":
        k = st.columns(4)
        k[0].metric("Rows", f"{len(df):,}")
        k[1].metric("Columns", f"{df.shape[1]}")
        k[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        k[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        types = pd.DataFrame({"type": ["numeric" if E._is_numeric(df[c]) else "categorical"
                                       for c in df.columns],
                              "non-null": df.notna().sum().values,
                              "unique": [df[c].nunique() for c in df.columns],
                              "role": ["output" if c == target else
                                       "input" if c in feats else "not used"
                                       for c in df.columns]},
                             index=df.columns)
        st.markdown("##### Columns")
        st.dataframe(types, width="stretch")
        st.markdown("##### First rows")
        st.dataframe(df.head(25), width="stretch")
    elif view == "Summary statistics":
        stats_df = E.summary_statistics(df, feats + [target])
        st.dataframe(stats_df.style.format(precision=4, na_rep="—"), width="stretch")
        st.download_button("⬇️ Download as CSV", stats_df.to_csv().encode(),
                           "summary_statistics.csv", "text/csv")
    elif view == "Target variable":
        if task == "regression" and E._is_numeric(df[target]):
            show_chart(C.histogram(df[target], target, MODE), "t_hist",
                       E.summary_statistics(df, [target]))
        else:
            counts = df[target].astype(str).value_counts()
            show_chart(C.category_bars(counts, MODE), "t_bars",
                       counts.rename("rows").to_frame().assign(
                           share=lambda d: (d["rows"] / d["rows"].sum()).round(4)))
            if len(counts) > 1:
                st.caption(f"Imbalance ratio (largest / smallest class): "
                           f"**{counts.max() / counts.min():.2f}**")
    elif view == "Distributions":
        f = st.selectbox("Variable", feats, key="dist_feat")
        if E._is_numeric(df[f]):
            if task == "classification":
                classes = sorted(df[target].dropna().astype(str).unique())
                show_chart(C.box_by_group(df.assign(**{target: df[target].astype(str)}), f, target,
                                          classes, MODE), "dist_box",
                           df.groupby(df[target].astype(str))[f].describe())
            show_chart(C.histogram(df[f], f, MODE), "dist_hist",
                       E.summary_statistics(df, [f]))
        else:
            counts = df[f].astype(str).value_counts()
            show_chart(C.category_bars(counts, MODE), "dist_cat", counts.to_frame("rows"))
    elif view == "Box plots":
        std = st.toggle("Standardise (put every variable on one scale)", value=True, key="box_std")
        show_chart(C.boxplots(df, numeric, MODE, std), "boxes",
                   E.summary_statistics(df, numeric))
    elif view == "Correlation matrix":
        c1, c2 = st.columns([1, 2])
        method = c1.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_m",
                              format_func=str.capitalize)
        cols = numeric + ([target] if E._is_numeric(df[target]) else [])
        if len(cols) < 2:
            st.info("Needs at least two numeric columns.")
            return
        R, P = E.correlation_with_pvalues(df, cols, method)
        show_chart(C.correlation_heatmap(R, P, MODE, E.significance_stars), "corr",
                   R.round(4).astype(str) + P.map(E.significance_stars))
        st.caption("`*` p < 0.01  ·  `**` p < 0.05  ·  unmarked: not significant at 0.05  ·  "
                   "diagonal and upper triangle omitted (the matrix is symmetric). "
                   "Same convention as the notebooks.")
    elif view == "Feature vs target":
        f = st.selectbox("Variable", feats, key="fvt_feat")
        if task == "regression" and E._is_numeric(df[f]) and E._is_numeric(df[target]):
            show_chart(C.scatter_vs_target(df, f, target, MODE), "fvt",
                       df[[f, target]].describe())
        elif E._is_numeric(df[f]):
            classes = sorted(df[target].dropna().astype(str).unique())
            show_chart(C.box_by_group(df.assign(**{target: df[target].astype(str)}), f, target,
                                      classes, MODE), "fvt_box",
                       df.groupby(df[target].astype(str))[f].describe())
        else:
            tab = pd.crosstab(df[f].astype(str), df[target].astype(str) if task ==
                              "classification" else pd.qcut(df[target], 4, duplicates="drop"))
            st.dataframe(tab, width="stretch")
    elif view == "Pairwise scatter":
        default = numeric[:4]
        pick = st.multiselect("Variables (up to 6)", numeric, default=default,
                              max_selections=6, key="splom_cols")
        if len(pick) >= 2:
            classes = (sorted(df[target].dropna().astype(str).unique())
                       if task == "classification" else None)
            show_chart(C.scatter_matrix(df.assign(**{target: df[target].astype(str)})
                                        if classes else df, pick, MODE,
                                        target if classes else None, classes), "splom")
            if classes and len(classes) > 3:
                st.caption("Coloured by class only for three classes or fewer — past that, "
                           "colours stop being reliably distinguishable.")
    elif view == "Missing values":
        if df.isna().sum().sum() == 0:
            st.success("No missing values in any column.")
        show_chart(C.missing_bars(df, MODE), "missing",
                   df.isna().sum().rename("missing").to_frame()
                   .assign(percent=lambda d: (d["missing"] / len(df) * 100).round(2)))
    elif view == "Outliers":
        tab = E.outlier_table(df, numeric)
        st.dataframe(tab.style.format(precision=3), width="stretch")
        st.caption("Outliers by the 1.5 × IQR rule. They are reported, not removed — "
                   "an extreme mix design is often real data.")


# =============================================================================
#  MODEL EXPLORER TAB
# =============================================================================
REG_GRAPHS = ["Actual vs predicted", "Residuals vs predicted", "Residual distribution",
              "Error by target range", "Feature importance (permutation)", "Learning curve",
              "Metric comparison (all models)", "Taylor diagram (all models)"]
CLF_GRAPHS = ["Confusion matrix", "ROC curve", "Precision–recall curve", "Calibration curve",
              "Threshold analysis", "Probability distribution", "Per-class metrics",
              "Feature importance (permutation)", "Learning curve",
              "Metric comparison (all models)"]


def _table_columns(df, digits=4):
    """
    Column formats for a numeric table: 4 decimals, right-aligned, sortable.

    Streamlit's grid draws a missing number as a grey "None" whatever the
    Styler says — Python's word for it, not the reader's. A column with gaps
    (a notebook's PySR row has no CV score or fit time) is therefore shown as
    right-aligned text with an en dash; complete columns stay numeric so they
    still sort by value.
    """
    cfg = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().any():
            df[c] = [("—" if pd.isna(v) else f"{v:,.{digits}f}") for v in df[c]]
            cfg[c] = st.column_config.TextColumn(c, alignment="right")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format=f"%.{digits}f", alignment="right")
    return cfg


def models_tab():
    ws = ss.get("ws")
    if ws is None:
        st.info("Train models (or load a trained-models file) in the sidebar to see them here.")
        return
    cat = E.metric_catalog(ws.task)
    lb = ws.leaderboard()
    best = ws.best
    bm = ws.metrics(best, "test")

    with st.container(border=True):
        st.markdown(f"#### ★ Best model: {esc(best)}")
        st.caption(f"Chosen by {esc(ws.b.get('selection_note', ''))}.")
        keys = (["R2", "RMSE", "MAE", "MAPE"] if ws.task == "regression"
                else ["ROC_AUC", "Accuracy", "F1", "MCC"])
        k = st.columns(len(keys) + 1)
        for col, m in zip(k, keys):
            col.metric(f"Test {E.metric_label(m)}", C.fmt(bm[m]), help=cat[m][1])
        cv = ws.b["cv_score"].get(best)
        k[-1].metric(f"CV {E.metric_label(ws.primary)}", C.fmt(cv) if cv is not None else "—",
                     help="Cross-validated score on the training split")

    st.markdown("##### Leaderboard")
    shown = lb.copy()
    shown.index = [f"★ {i}" if i == best else i for i in shown.index]
    shown.columns = [E.metric_label(c) if c in cat else
                     f"CV {E.metric_label(c[3:])}" if c.startswith("CV ") else c
                     for c in shown.columns]
    st.dataframe(shown, width="stretch", column_config=_table_columns(shown))

    st.markdown("##### Inspect a model")
    c1, c2, c3 = st.columns([2, 3, 2])
    names = ws.model_names
    model = c1.selectbox("Model", names, index=names.index(best), key=f"mx_model::{ws.id}",
                         format_func=lambda n: f"★ {n}" if n == best else n)
    default_metrics = (["R2", "RMSE", "MAE", "SI"] if ws.task == "regression"
                       else ["ROC_AUC", "Accuracy", "F1", "MCC"])
    metrics = c2.multiselect("Metrics", list(cat), default=default_metrics,
                             key=f"mx_metrics::{ws.id}",
                             format_func=E.metric_label)
    graphs = REG_GRAPHS if ws.task == "regression" else CLF_GRAPHS
    if ws.task == "classification" and ws.n_classes != 2:
        graphs = [g for g in graphs if g != "Threshold analysis"]
    if not ws.can_refit:
        graphs = [g for g in graphs if g != "Learning curve"]
    if ws.b.get("extras", {}).get("pysr"):
        graphs = graphs + ["Symbolic equation (PySR)"]
    graph = c3.selectbox("Graph", graphs, key=f"mx_graph::{ws.id}")

    if metrics:
        te, tr = ws.metrics(model, "test"), ws.metrics(model, "train")
        tiles = st.columns(min(len(metrics), 6))
        for i, m in enumerate(metrics):
            higher = cat[m][0]
            d = te[m] - tr[m] if np.isfinite(te[m]) and np.isfinite(tr[m]) else None
            tiles[i % len(tiles)].metric(
                f"{E.metric_label(m)} (test)", C.fmt(te[m]),
                delta=None if d is None else f"{d:+.3g} vs train",
                delta_color="normal" if higher else "inverse", help=cat[m][1])
        st.caption("The small figure under each value is test minus train: a large gap in "
                   "the wrong direction means the model fits its training data far better "
                   "than new data — overfitting.")

    render_graph(ws, model, graph, metrics or default_metrics)

    with st.expander(f"Hyperparameters of {model}"):
        p = ws.b["params"].get(model) or {}
        if p:
            st.dataframe(pd.DataFrame({"value": [str(v) for v in p.values()]},
                                      index=list(p.keys())), width="stretch")
        else:
            st.caption("None recorded.")


def render_graph(ws, model, graph, metrics):
    y_te, P = ws.y_test, ws.pred(model)
    key = f"g::{ws.id}::{model}::{graph}"
    if graph == "Actual vs predicted":
        tr = st.toggle("Also show training rows", key=f"avp_tr::{ws.id}")
        show_chart(C.actual_vs_predicted(y_te, P, MODE,
                                         ws.y_train if tr else None,
                                         ws.pred(model, "train") if tr else None, ws.target),
                   key, pd.DataFrame({"actual": y_te, "predicted": P, "residual": y_te - P}))
    elif graph == "Residuals vs predicted":
        show_chart(C.residuals_vs_predicted(y_te, P, MODE, ws.target), key,
                   pd.DataFrame({"predicted": P, "residual": y_te - P}))
        st.caption("A healthy model leaves a shapeless cloud around zero. A funnel means the "
                   "error grows with the prediction; a curve means a pattern the model missed.")
    elif graph == "Residual distribution":
        show_chart(C.residual_histogram(y_te, P, MODE), key,
                   pd.Series(y_te - P, name="residual").describe().to_frame())
    elif graph == "Error by target range":
        show_chart(C.error_by_range(y_te, P, MODE, target=ws.target), key)
        st.caption("Is the model equally good across the whole range, or only in the middle?")
    elif graph == "Feature importance (permutation)":
        with st.spinner("Shuffling each variable and re-scoring…"):
            imp = ws.permutation_importance(model)
        show_chart(C.importance_bars(imp, MODE, imp.attrs["score"]), key, imp)
        st.caption(f"How much the test {imp.attrs['score']} drops when one variable is "
                   f"randomly shuffled — i.e. how much the model relies on it to be right.")
    elif graph == "Learning curve":
        with st.spinner("Retraining on growing subsets of the training data…"):
            lc = ws.learning_curve(model)
        if lc is None:
            st.info("Learning curves need the in-app trainer (a loaded bundle cannot retrain).")
        else:
            show_chart(C.learning_curve_chart(lc, MODE), key, lc)
            st.caption("If the two curves are still converging at the right edge, more data "
                       "would help; a persistent gap means overfitting.")
    elif graph == "Metric comparison (all models)":
        m = st.selectbox("Metric", metrics, key=f"cmp_metric::{ws.id}",
                         format_func=E.metric_label)
        vals = pd.Series({n: ws.metrics(n, "test")[m] for n in ws.model_names})
        higher = E.metric_catalog(ws.task)[m][0]
        show_chart(C.metric_bars(vals, f"Test {E.metric_label(m)}", MODE, model, higher, ws.best),
                   key + m, vals.rename(m).to_frame())
        st.caption(f"The highlighted bar is **{esc(model)}**; ★ marks the best model.")
    elif graph == "Taylor diagram (all models)":
        ts = E.taylor_stats(ws)
        show_chart(C.taylor_diagram(ts, MODE, model), key, ts)
        st.caption("Angle = correlation with the actual values, radius = spread relative to "
                   "the actual spread. The closer a model sits to the ★ (perfect agreement), "
                   "the better; the arcs are equal centred-RMS error.")
    elif graph == "Confusion matrix":
        norm = st.toggle("Show as % of each true class (recall)", value=True, key=f"cm_n::{ws.id}")
        cm = E.confusion(ws, model)
        show_chart(C.confusion_matrix_chart(cm, ws.classes, MODE, norm), key,
                   pd.DataFrame(cm, index=[f"true {c}" for c in ws.classes],
                                columns=[f"pred {c}" for c in ws.classes]))
        st.caption("Rows are the true class, columns the prediction. The % view exposes a "
                   "model that ignores a small class — the counts view hides it.")
    elif graph == "ROC curve":
        curves = E.roc_curves(ws, model)
        show_chart(C.roc_chart(curves, MODE), key,
                   pd.DataFrame([(c[0], c[3]) for c in curves], columns=["curve", "AUC"]))
    elif graph == "Precision–recall curve":
        curves = E.pr_curves(ws, model)
        show_chart(C.pr_chart(curves, MODE), key,
                   pd.DataFrame([(c[0], c[3], c[4]) for c in curves],
                                columns=["class", "average precision", "prevalence"]))
    elif graph == "Calibration curve":
        curves = E.calibration_curves(ws, model)
        show_chart(C.calibration_chart(curves, MODE), key)
        st.caption("Points on the diagonal mean a predicted 70% really happens 70% of the time.")
    elif graph == "Threshold analysis":
        th, pr, rc, f1, best_t = E.threshold_sweep(ws, model)
        show_chart(C.threshold_chart(th, pr, rc, f1, best_t, MODE), key,
                   pd.DataFrame({"threshold": th, "precision": pr, "recall": rc, "F1": f1}))
        st.caption(f"Predictions use a 0.5 cut-off on P({esc(ws.classes[ws.pos])}); the F1-best "
                   f"cut-off on the test set is {best_t:.2f}.")
    elif graph == "Probability distribution":
        groups, xt = E.probability_groups(ws, model)
        show_chart(C.density_lines(groups, MODE, xt), key)
        st.caption("Well-separated humps mean the model tells the classes apart confidently.")
    elif graph == "Per-class metrics":
        rep = E.per_class_report(ws, model)
        m = st.selectbox("Metric", ["F1", "Precision", "Recall"], key=f"pc_m::{ws.id}")
        show_chart(C.per_class_bars(rep, m, MODE), key + m, rep)
    elif graph == "Symbolic equation (PySR)":
        eqs = ws.b["extras"]["pysr"]
        for eq in (eqs if isinstance(eqs, list) else [eqs]):
            st.markdown(f"**{esc(eq.get('lhs', ws.target))} =**")
            if eq.get("latex"):
                st.latex(eq["latex"])
            st.code(eq.get("equation", ""), language=None)
            if eq.get("renamed"):
                st.caption("Renamed for PySR: " + ", ".join(
                    f"`{esc(k)}` = {esc(v)}" for k, v in eq["renamed"].items()))
            if eq.get("front") is not None:
                with st.expander("The full Pareto front (every complexity level)"):
                    st.dataframe(eq["front"], width="stretch")
        st.caption("The equation found by symbolic regression in the notebook. It is also "
                   "in the model list above as 'PySR (symbolic)', so you can predict with it.")


# =============================================================================
#  PAGE
# =============================================================================
sidebar()
ws = ss.get("ws")
st.markdown("# ML Studio")
if ws is not None:
    m = ws.metrics(ws.best, "test")
    score = (f"test R² {C.fmt(m['R2'])}" if ws.task == "regression"
             else f"test ROC-AUC {C.fmt(m['ROC_AUC'])}")
    st.caption(f"**{ws.task.capitalize()}** · predicting **{esc(ws.target)}** from "
               f"{len(ws.features)} variables · best model **{esc(ws.best)}** ({score}) · "
               f"{esc(ws.source)}")
else:
    st.caption("A GUI for the regression and classification workflows.")

ss.setdefault("main_tabs", TAB_PREDICT)
tabs = st.tabs([TAB_PREDICT, TAB_DATA, TAB_MODELS], key="main_tabs", on_change="rerun")
active = ss.get("main_tabs", TAB_PREDICT)
# Lazy tabs: only the visible tab computes. Every widget interaction reruns the
# whole script, so without this each keystroke in the prediction form would
# also rebuild every data and model chart.
with tabs[0]:
    if active == TAB_PREDICT:
        predict_tab()
with tabs[1]:
    if active == TAB_DATA:
        data_tab()
with tabs[2]:
    if active == TAB_MODELS:
        models_tab()
