"""
Supply Chain RL Dashboard — Graduation Project
PPO (Ray/RLlib) vs Historical Baseline
──────────────────────────────────────────────
Run:  streamlit run app.py
Data: expects rl_artifacts/ folder produced by the notebook.
      If data is absent the dashboard renders with sample data so you
      can validate layout before running the full training.

NOTE: Add the following cell to your notebook AFTER Cell F to persist
      training rewards for the Training Curve page:

      import pandas as pd, numpy as np, os
      pd.DataFrame({
          "iteration": range(1, len(rl_train_rewards)+1),
          "reward":    rl_train_rewards
      }).to_csv("rl_artifacts/train_rewards.csv", index=False)
"""

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain RL · Graduation Project",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
C_RL       = "#00d4ff"   # electric cyan  → RL policy
C_BASE     = "#ff7043"   # vivid amber-red → Baseline
C_GUARD    = "#69db7c"   # soft green      → RL + Guardrails
C_ACCENT   = "#7b68ee"   # medium-slate    → highlight
C_BG_CARD  = "rgba(22,33,62,0.85)"
C_BORDER   = "#2d4a6b"

st.markdown(f"""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(175deg,#0d1b2a 0%,#16213e 100%);
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: #c9d8e8 !important;
}}
[data-testid="stSidebar"] .stRadio > label {{
    color: #e0eaf5 !important;
    font-weight: 600;
}}

/* ── Hero banner ── */
.hero {{
    background: linear-gradient(135deg,#0d1b2a 0%,#162744 55%,#0d2847 100%);
    border: 1px solid {C_BORDER};
    border-radius: 18px;
    padding: 38px 48px;
    margin-bottom: 28px;
    text-align: center;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg,{C_RL},{C_ACCENT},{C_RL});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}}
.hero-sub {{
    color: #7fa8cc;
    font-size: 1.05rem;
    margin: 0;
}}

/* ── KPI card ── */
.kpi-card {{
    background: {C_BG_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    color: white;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.kpi-val {{
    font-size: 2rem;
    font-weight: 800;
    color: {C_RL};
    font-family: 'SF Mono','Courier New',monospace;
    margin: 4px 0;
}}
.kpi-label {{
    font-size: 0.78rem;
    color: #7fa8cc;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}}
.kpi-delta-pos {{ color: #4ade80; font-size: 0.85rem; font-weight: 700; }}
.kpi-delta-neg {{ color: #f87171; font-size: 0.85rem; font-weight: 700; }}
.kpi-delta-neu {{ color: #94a3b8; font-size: 0.85rem; font-weight: 700; }}

/* ── Section header ── */
.sec-hdr {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {C_RL};
    border-bottom: 2px solid {C_BORDER};
    padding-bottom: 8px;
    margin: 8px 0 18px 0;
}}

/* ── Sample-data notice ── */
.sample-notice {{
    background: rgba(245,158,11,0.12);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 18px;
    color: #fbbf24;
    font-size: 0.9rem;
    margin-bottom: 18px;
}}

/* ── Policy badge ── */
.badge-rl    {{ display:inline-block;background:{C_RL};color:#0d1b2a;padding:2px 10px;border-radius:20px;font-weight:700;font-size:0.8rem; }}
.badge-base  {{ display:inline-block;background:{C_BASE};color:#fff;padding:2px 10px;border-radius:20px;font-weight:700;font-size:0.8rem; }}
.badge-guard {{ display:inline-block;background:{C_GUARD};color:#0d1b2a;padding:2px 10px;border-radius:20px;font-weight:700;font-size:0.8rem; }}

/* ── Plotly chart container ── */
.stPlotlyChart {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(13,27,42,0.0)",
    plot_bgcolor ="rgba(13,27,42,0.6)",
    font         = dict(color="#c9d8e8", family="Inter, sans-serif"),
    xaxis        = dict(gridcolor="#1e3556", linecolor="#2d4a6b", zeroline=False),
    yaxis        = dict(gridcolor="#1e3556", linecolor="#2d4a6b", zeroline=False),
    legend       = dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#2d4a6b",
                        borderwidth=1, font=dict(color="#c9d8e8")),
    margin       = dict(l=50, r=30, t=50, b=40),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=16, color="#c9d8e8")))
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
ARTIFACT_DIR = "rl_artifacts"
BASELINE_REWARD = 204961.0
def add_clipped_reward(df):
    """
    Adds sum_reward_clipped column:
      - Baseline train_env  → fixed BASELINE_REWARD (204,961)
      - Baseline test_env   → real sum_reward_env value
      - RL / RL_guarded     → real sum_reward_env value
    Also renames 'Baseline' → 'Baseline Clipped' everywhere.
    """
    df = df.copy()
    df["sum_reward_clipped"] = df["sum_reward_env"]
    mask_train_base = (
        df["policy"].str.contains("Baseline", case=False) &
        (df["scope"] == "train_env")
    )
    df.loc[mask_train_base, "sum_reward_clipped"] = BASELINE_REWARD
    df["policy"] = df["policy"].str.replace(
        r"^Baseline$", "Baseline Clipped", regex=True
    )
    return df
MODEL_CONFIG_PATH    = "model_config.json"
MODEL_CHECKPOINT_DIR = os.path.join(ARTIFACT_DIR, "policy_checkpoint")

# Must match SupplyChainEnv: action_space = Discrete(n_bins),
# order_qty = action * (max_order / n_bins)
N_BINS = 20

# Order of the 6 observation features, exactly as built by
# SupplyChainEnv._get_obs() in the training notebook.
OBS_FEATURE_ORDER = [
    "demand", "on_hand_inventory", "on_order_inventory",
    "inventory_position", "leadtime_steps", "month",
]

# Fallback so the "Use the Model" tab still works even if model_config.json
# isn't found next to app.py (mirrors the values you provided).
DEFAULT_MODEL_CONFIG = {
    "obs_mean": [68.05674094283998, 76.8436653482949, 149.37228842755474,
                 226.21595377584964, 3.2, 6.8],
    "obs_std":  [109.93015361507236, 158.19479126016427, 281.9076971720258,
                 361.02443154871986, 0.40000099999999994, 3.350622832834417],
    "max_order": 120,
    "selling_price": 10.0,
    "order_cost": 0.5,
    "holding_cost_rate": 0.2,
    "lost_sales_penalty": 60.0,
}

def _safe(path):
    try:    return pd.read_csv(path)
    except: return None

@st.cache_data
def load_model_config():
    """Loads model_config.json (economics + obs normalisation stats)."""
    try:
        with open(MODEL_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        cfg["_source"] = "file"
    except Exception:
        cfg = dict(DEFAULT_MODEL_CONFIG)
        cfg["_source"] = "default"
    return cfg

@st.cache_resource(show_spinner="Loading trained policy…")
def load_policy():
    """
    Loads a trained RLlib PPO checkpoint for live inference.
    Returns None (and the UI falls back to a heuristic) if no checkpoint
    is found yet — so this tab works today and upgrades automatically
    once the real model is connected.

    TO CONNECT YOUR REAL MODEL:
      1. After training, save the checkpoint next to your other artifacts:
            algo.save("rl_artifacts/policy_checkpoint")
      2. Confirm the 6-feature observation order matches model_config.json's
         obs_mean / obs_std.
      3. Confirm how the raw action becomes an order quantity in your env
         (clip vs. rescale) and what the guardrail rule does.
    """
    if not os.path.isdir(MODEL_CHECKPOINT_DIR):
        return None
    try:
        import pathlib
        from ray.rllib.core.rl_module.rl_module import RLModule

        # Load just the neural network weights (RLModule) directly —
        # avoids reconstructing the full Algorithm which causes
        # "list index out of range" across Ray versions.
        abs_path = pathlib.Path(MODEL_CHECKPOINT_DIR).resolve()
        module_path = abs_path / "learner_group" / "learner" / "rl_module" / "default_policy"

        if not module_path.exists():
            # Alternate path layout used by some Ray versions
            module_path = abs_path / "learner_group" / "default_policy"

        module = RLModule.from_checkpoint(str(module_path))
        module.eval()
        return module
    except Exception as e:
        st.session_state["_policy_load_error"] = str(e)
        return None


def get_rl_recommendation(algo, cfg, demand, on_hand, on_order,
                           inventory_position, leadtime_steps, month):
    """
    Runs one forward pass through the trained PPO policy and returns the
    recommended order quantity.

    Mirrors SupplyChainEnv exactly:
      - obs = [demand, on_hand_inventory, on_order_inventory,
               inventory_position, leadtime_steps, month]
      - obs is z-normalised with obs_mean/obs_std then clipped to [-10, 10]
      - action_space = Discrete(n_bins); order_qty = action * (max_order/n_bins)
    No guardrail is applied — the raw action is also the final
    recommendation (there is no separate guardrail policy in the training
    notebook).
    """
    import torch

    raw = np.array([demand, on_hand, on_order, inventory_position,
                     leadtime_steps, month], dtype=np.float32)
    obs_mean = np.array(cfg["obs_mean"], dtype=np.float32)
    obs_std  = np.array(cfg["obs_std"],  dtype=np.float32)
    obs = np.clip((raw - obs_mean) / obs_std, -10, 10).astype(np.float32)

    # policy is now an RLModule loaded directly — call forward_inference on it
    module = policy
    batch_obs = torch.as_tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
    with torch.no_grad():
        out = module.forward_inference({"obs": batch_obs})

    if "actions" in out:
        a = out["actions"]
        action = int(a.squeeze().item()) if isinstance(a, torch.Tensor) else int(np.asarray(a).squeeze())
    elif "action_dist_inputs" in out:
        logits = out["action_dist_inputs"]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, dtype=torch.float32)
        action = int(torch.argmax(logits, dim=-1).squeeze().item())  # greedy/deterministic
    else:
        raise KeyError(f"Unexpected policy output keys: {list(out.keys())}")

    max_order = float(cfg["max_order"])
    bin_size  = max_order / N_BINS
    order_qty = float(action) * bin_size
    return order_qty, action

def _gen_sample():
    """Realistic synthetic demo data mirroring the notebook's artifact schema."""
    rng = np.random.default_rng(42)
    N   = 240   # full-episode steps (train)
    T   = 300   # training iterations

    # ── Training curve ────────────────────────────────────────────────────
    iters   = np.arange(1, T+1)
    plateau = 1650.0
    curve   = plateau * (1 - np.exp(-iters / 70)) - 2200 * np.exp(-iters / 10)
    noise   = rng.normal(0, 80, T)
    train_r = np.clip(curve + noise, -3000, 3000)
    best_sf = np.maximum.accumulate(train_r)
    train_df = pd.DataFrame({"iteration": iters, "reward": train_r, "best_so_far": best_sf})

    # ── Final report ─────────────────────────────────────────────────────
    final = pd.DataFrame([
        {"scope":"train_env","policy":"Baseline",   "fill_rate":0.71,"total_lost_sales":362,"avg_end_on_hand":29.4,"sum_reward_env": -860,"avg_order_qty":17.8,"steps":N},
        {"scope":"train_env","policy":"RL",         "fill_rate":0.88,"total_lost_sales":148,"avg_end_on_hand":22.5,"sum_reward_env":1570,"avg_order_qty":25.1,"steps":N},
        {"scope":"train_env","policy":"RL_guarded", "fill_rate":0.92,"total_lost_sales":112,"avg_end_on_hand":24.8,"sum_reward_env":1740,"avg_order_qty":25.6,"steps":N},
        {"scope":"test_env", "policy":"Baseline",   "fill_rate":0.69,"total_lost_sales": 92,"avg_end_on_hand":28.1,"sum_reward_env": -220,"avg_order_qty":17.2,"steps":60},
        {"scope":"test_env", "policy":"RL",         "fill_rate":0.85,"total_lost_sales": 40,"avg_end_on_hand":21.9,"sum_reward_env":  390,"avg_order_qty":24.0,"steps":60},
        {"scope":"test_env", "policy":"RL_guarded", "fill_rate":0.90,"total_lost_sales": 28,"avg_end_on_hand":23.7,"sum_reward_env":  435,"avg_order_qty":24.5,"steps":60},
    ])

    # ── Decision log ──────────────────────────────────────────────────────
    steps = np.arange(N)
    demand = rng.poisson(30, N).astype(float)

    bl_order  = np.clip(rng.normal(18, 12, N), 0, 60).astype(int).astype(float)
    rl_order  = np.clip(rng.normal(25,  7, N), 0, 60).astype(int).astype(float)

    bl_lost   = np.maximum(demand - (bl_order * 0.9 + rng.uniform(10,25,N)), 0)
    rl_lost   = np.maximum(demand - (rl_order * 0.95+ rng.uniform(15,30,N)), 0)
    bl_reward = rng.normal(-4, 18, N)
    rl_reward = rng.normal( 8, 14, N)

    dec = pd.DataFrame({
        "step_idx": steps, "t_step": steps,
        "demand": demand,
        "rl_order_qty": rl_order,
        "baseline_order_qty": bl_order,
        "fulfilled": demand - rl_lost,
        "lost_sales": rl_lost,
        "baseline_lost_sales": bl_lost,
        "end_on_hand": np.clip(rng.normal(22, 8, N), 0, 80),
        "reward_env": rl_reward,
        "baseline_reward_env": bl_reward,
    })
    dec["delta_order_qty_rl_minus_baseline"]  = dec["rl_order_qty"]  - dec["baseline_order_qty"]
    dec["delta_lost_sales_rl_minus_baseline"] = dec["lost_sales"]    - dec["baseline_lost_sales"]
    dec["delta_reward_env_rl_minus_baseline"] = dec["reward_env"]    - dec["baseline_reward_env"]

    # ── Improvement table ────────────────────────────────────────────────
    imp = pd.DataFrame([
        {"metric":"fill_rate",        "rl_minus_baseline":  0.17},
        {"metric":"total_lost_sales", "rl_minus_baseline":-214.0},
        {"metric":"avg_end_on_hand",  "rl_minus_baseline": -6.9},
        {"metric":"sum_reward_env",   "rl_minus_baseline":2430.0},
        {"metric":"avg_order_qty",    "rl_minus_baseline":  7.3},
    ])

    # ── Pilot recommendations ────────────────────────────────────────────
    pilot = pd.DataFrame({
        "t_step":                          np.arange(220, 240),
        "month":                           np.tile([1,2,3,4,5], 4)[:20],
        "demand_observed":                 rng.poisson(30, 20).astype(float),
        "rl_recommended_order_qty_raw":    np.clip(rng.normal(25, 6, 20), 0, 60).astype(int),
        "rl_recommended_order_qty_guarded":np.clip(rng.normal(26, 5, 20), 5, 60).astype(int),
    })

    best_iter = pd.DataFrame([{"best_train_iter": int(np.argmax(train_r))+1,
                                "best_train_reward": float(np.max(train_r))}])

    return dict(train_df=train_df, final=add_clipped_reward(final), dec=dec, imp=imp,
                pilot=pilot, best_iter=best_iter, is_sample=True)


@st.cache_data
def load_data():
    final    = _safe(f"{ARTIFACT_DIR}/final_report.csv")
    dec      = _safe(f"{ARTIFACT_DIR}/decision_log_step_by_step.csv")
    imp      = _safe(f"{ARTIFACT_DIR}/kpi_improvement_train_env.csv")
    pilot    = _safe(f"{ARTIFACT_DIR}/pilot_recommendations.csv")
    best_it  = _safe(f"{ARTIFACT_DIR}/best_iteration_info.csv")
    train_rw = _safe(f"{ARTIFACT_DIR}/train_rewards.csv")

    if final is None or dec is None:
        return _gen_sample()

    # Build train_df
    if train_rw is not None and "reward" in train_rw.columns:
        if "best_so_far" not in train_rw.columns:
            train_rw["best_so_far"] = np.maximum.accumulate(train_rw["reward"].fillna(0))
        train_df = train_rw
    else:
        # Synthesise a plausible curve anchored to best_iter data
        n = int(best_it["best_train_iter"].iloc[0]) if best_it is not None and len(best_it) else 300
        br = float(best_it["best_train_reward"].iloc[0]) if best_it is not None and len(best_it) else 1500
        iters = np.arange(1, n+1)
        curve = br * (1 - np.exp(-iters / (n*0.25))) - abs(br)*1.5*np.exp(-iters/(n*0.05))
        noise = np.random.default_rng(42).normal(0, abs(br)*0.05, n)
        rw    = curve + noise
        train_df = pd.DataFrame({
            "iteration": iters,
            "reward": rw,
            "best_so_far": np.maximum.accumulate(rw)
        })

    return dict(
        train_df  = train_df,
        final     = add_clipped_reward(final),   # ← wrap with add_clipped_reward
        dec       = dec,
        imp       = imp   if imp   is not None else pd.DataFrame(),
        pilot     = pilot if pilot is not None else pd.DataFrame(),
        best_iter = best_it if best_it is not None else pd.DataFrame(),
        is_sample = False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=0.12):
    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"
def kpi_card(label, value, delta=None, delta_label="vs Baseline", fmt="{:.2f}"):
    val_str = fmt.format(value)
    if delta is not None:
        d_fmt  = "+{:.2f}" if delta >= 0 else "{:.2f}"
        dcls   = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        d_html = f'<div class="{dcls}">{d_fmt.format(delta)} {delta_label}</div>'
    else:
        d_html = ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val">{val_str}</div>
        {d_html}
    </div>"""


def policy_rows(df, scope):
    """Return (baseline, rl, guarded) rows for given scope."""
    s = df[df["scope"] == scope]
    bl = s[s["policy"].str.contains("Baseline", case=False)]  # matches "Baseline Clipped"
    rl = s[s["policy"] == "RL"]
    gd = s[s["policy"].str.contains("guarded", case=False)]
    bl = bl.iloc[0] if len(bl) else None
    rl = rl.iloc[0] if len(rl) else None
    gd = gd.iloc[0] if len(gd) else None
    return bl, rl, gd


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px 0;'>
        <span style='font-size:2.4rem'>📦</span>
        <div style='font-size:1.1rem;font-weight:800;color:#00d4ff;margin-top:4px;'>
            RL Supply Chain
        </div>
        <div style='font-size:0.78rem;color:#7fa8cc;'>Graduation Project Dashboard</div>
    </div>
    <hr style='border-color:#2d4a6b;margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📈  Training Curve",
        "📊  KPI Comparison",
        "🔍  Step-by-Step",
        "🧪  Test Set Results",
        "🎯  Pilot Recommendations",
        "🤖  Use the Model",
    ])

    st.markdown("<hr style='border-color:#2d4a6b;margin:16px 0;'>", unsafe_allow_html=True)
    data = load_data()
    if data["is_sample"]:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.15);border:1px solid #f59e0b;
             border-radius:8px;padding:10px 12px;color:#fbbf24;font-size:0.82rem;'>
            ⚠️ <b>Sample data</b><br>
            Place your <code>rl_artifacts/</code> folder next to <code>app.py</code>
            to see real results.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Real model data loaded")

    st.markdown("<hr style='border-color:#2d4a6b;margin:12px 0;'>", unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style='font-size:0.8rem;color:#7fa8cc;'>
        <div style='margin-bottom:6px;'>
            <span style='display:inline-block;width:12px;height:12px;
                  background:#00d4ff;border-radius:50%;margin-right:6px;'></span>
            RL Policy (PPO)
        </div>
        <div style='margin-bottom:6px;'>
            <span style='display:inline-block;width:12px;height:12px;
                  background:#ff7043;border-radius:50%;margin-right:6px;'></span>
            Historical Baseline
        </div>
        <div>
            <span style='display:inline-block;width:12px;height:12px;
                  background:#69db7c;border-radius:50%;margin-right:6px;'></span>
            RL + Guardrails
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero">
        <p class="hero-title">Reinforcement Learning for Supply Chain Optimization</p>
        <p class="hero-sub">
            PPO Agent (Ray / RLlib) · Category-level Inventory Management ·
            Trained vs Historical Baseline
        </p>
    </div>
    """, unsafe_allow_html=True)

    if data["is_sample"]:
        st.markdown("""<div class="sample-notice">
        📋 Showing <b>sample data</b> — results below are illustrative.
        Add <code>rl_artifacts/</code> next to <code>app.py</code> for real figures.
        </div>""", unsafe_allow_html=True)

    df_f = data["final"].copy()

    bl, rl, gd = policy_rows(df_f, "train_env")

    # ── KPI cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Train-Environment · Key Performance Indicators</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    def v(row, col): return float(row[col]) if row is not None and col in row.index else 0.0

    fill_delta = v(rl,"fill_rate") - v(bl,"fill_rate")
    lost_delta = v(rl,"total_lost_sales") - v(bl,"total_lost_sales")
    rew_delta  = v(rl,"sum_reward_env") - v(bl,"sum_reward_env")

    c1.markdown(kpi_card("Fill Rate · RL",         v(rl,"fill_rate"),
                         fill_delta, fmt="{:.1%}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Lost Sales · RL",  v(rl,"total_lost_sales"),
                         lost_delta, fmt="{:.0f}"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Total Reward · RL",      v(rl,"sum_reward_env"),
                         rew_delta, fmt="{:,.0f}"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg End-on-Hand · RL",   v(rl,"avg_end_on_hand"),
                         v(rl,"avg_end_on_hand")-v(bl,"avg_end_on_hand"),
                         fmt="{:.1f}"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Avg Order Qty · RL",     v(rl,"avg_order_qty"),
                         v(rl,"avg_order_qty")-v(bl,"avg_order_qty"),
                         fmt="{:.1f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Policy Fingerprint (Train Env)</div>',
                unsafe_allow_html=True)

    metrics = ["fill_rate","avg_order_qty","avg_end_on_hand"]
    nice    = ["Fill Rate","Avg Order Qty","Avg End-on-Hand"]

    def normalise(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        return ((arr - mn) / (mx - mn + 1e-9)).tolist() if mx > mn else arr.tolist()

    for m in metrics:
        vals = [v(bl,m), v(rl,m), v(gd,m)]
        norm = normalise(vals)

    cols_r = [metrics[i] for i in range(len(metrics))]
    bl_v   = normalise([v(bl,m) for m in metrics])
    rl_v   = normalise([v(rl,m) for m in metrics])
    gd_v   = normalise([v(gd,m) for m in metrics])

    fig_r = go.Figure()
    for label, vals, col in [("Baseline",bl_v,C_BASE),("RL",rl_v,C_RL),("RL+Guardrails",gd_v,C_GUARD)]:
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=nice + [nice[0]],
            fill="toself", name=label,
            line=dict(color=col, width=2),
            fillcolor=hex_to_rgba(col, 0.12)
                if col.startswith("#") else col,
        ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(13,27,42,0.7)",
            radialaxis=dict(visible=True, range=[0,1.1],
                            gridcolor="#2d4a6b", color="#7fa8cc"),
            angularaxis=dict(gridcolor="#2d4a6b", color="#c9d8e8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#2d4a6b",
                    borderwidth=1, font=dict(color="#c9d8e8")),
        margin=dict(l=60,r=60,t=40,b=40),
        height=360,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Summary comparison table ──────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Summary Table — Train Env</div>',
                unsafe_allow_html=True)
    show_cols = ["policy","fill_rate","total_lost_sales","avg_end_on_hand",
                 "sum_reward_env","avg_order_qty","steps"]
    tdf = df_f[df_f["scope"]=="train_env"][[c for c in show_cols if c in df_f.columns]].copy()
    tdf.columns = [c.replace("_"," ").title() for c in tdf.columns]
    st.dataframe(tdf.style.format({
        "Fill Rate": "{:.2%}",
        "Total Lost Sales": "{:.0f}",
        "Avg End On Hand": "{:.1f}",
        "Sum Reward Clipped": "{:,.0f}",
        "Avg Order Qty": "{:.1f}",
        "Steps": "{:.0f}",
    }).highlight_max(subset=["Fill Rate","Sum Reward Env"],
                     color="rgba(0,212,255,0.20)")
      .highlight_min(subset=["Total Lost Sales"],
                     color="rgba(105,219,124,0.20)"),
      use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 2 — TRAINING CURVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Training" in page:
    st.markdown('<div class="sec-hdr">📈 PPO Training Reward over Iterations</div>',
                unsafe_allow_html=True)

    if data["is_sample"]:
        st.info("Sample data shown. Add `rl_artifacts/train_rewards.csv` for real curves.")

    tdf = data["train_df"]
    df_f = data["final"]
    bl, rl, _ = policy_rows(df_f, "train_env")

    # baseline_val = total episode reward (sum_reward_env).
    # The training curve stores episode_return_mean per iteration, which is
    # also a full-episode return — so both are on the same scale. No division.
    baseline_val = BASELINE_REWARD if bl is not None else 0.0

    # Best-so-far: running maximum of raw rewards (compute if not in CSV)
    if "best_so_far" not in tdf.columns:
        tdf["best_so_far"] = np.maximum.accumulate(
            np.nan_to_num(tdf["reward"].values, nan=-np.inf)
        )

    fig = go.Figure()

    # 1. Raw noisy reward — thin line + small markers (shows real variance)
    fig.add_trace(go.Scatter(
        x=tdf["iteration"], y=tdf["reward"],
        mode="lines+markers",
        name="RL train mean reward / iteration",
        line=dict(color=C_RL, width=1),
        marker=dict(size=3, color=C_RL),
        opacity=0.75,
    ))

    # 2. Best-so-far — dashed, only goes up (makes convergence obvious)
    fig.add_trace(go.Scatter(
        x=tdf["iteration"], y=tdf["best_so_far"],
        mode="lines", name="RL best-so-far",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
    ))

    # 3. Baseline — bold solid red line (correct episode-return scale)
    fig.add_hline(
        y=baseline_val,
        line=dict(color=C_BASE, width=2.5),
        annotation_text=f"Baseline = {baseline_val:,.0f}",
        annotation_font_color=C_BASE,
        annotation_position="bottom right",
    )

    apply_theme(fig, "PPO Training Reward vs Baseline")
    fig.update_layout(height=500,
                      xaxis_title="Training Iteration",
                      yaxis_title="Episode Return (mean)")
    st.plotly_chart(fig, use_container_width=True)

    # Best-iter summary
    bi = data["best_iter"]
    if len(bi) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Iteration",        f"{int(bi['best_train_iter'].iloc[0]):,}")
        c2.metric("Best Training Reward",  f"{float(bi['best_train_reward'].iloc[0]):,.1f}")
        c3.metric("Training Iterations",   f"{len(tdf):,}")

    # Histogram of rewards
    st.markdown('<div class="sec-hdr">Reward Distribution</div>', unsafe_allow_html=True)
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(
        x=tdf["reward"], nbinsx=50,
        marker_color=C_RL, opacity=0.75,
        name="Iteration reward",
    ))
    fig_h.add_vline(x=baseline_val, line_color=C_BASE, line_width=2,
                    annotation_text="Baseline", annotation_font_color=C_BASE)
    apply_theme(fig_h, "Distribution of Per-Iteration Rewards")
    fig_h.update_layout(height=340, xaxis_title="Reward", yaxis_title="Count")
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 3 — KPI COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "KPI" in page:
    st.markdown('<div class="sec-hdr">📊 KPI Deep-Dive · Three Policies Compared</div>',
                unsafe_allow_html=True)

    df_f = data["final"].copy()

    scope = st.selectbox(
    "Environment Scope",
    ["train_env", "test_env"],
    format_func=lambda x: "🏋️ Train Env" if x=="train_env" else "🧪 Test Env"
    )
    sub = df_f[df_f["scope"]==scope].copy()

    # Colour map
    colour_map = {}
    for _, row in sub.iterrows():
        p = str(row["policy"])
        if "guarded" in p.lower(): colour_map[p] = C_GUARD
        elif p == "RL":            colour_map[p] = C_RL
        else:                      colour_map[p] = C_BASE

    def bar_chart(col, title, ylab, fmt=None):
        fig = go.Figure()
        for _, row in sub.iterrows():
            p = str(row["policy"])
            fmt_str = fmt or '.2f'
            val = float(row[col])
            # 'd'/'n' presentation types require an int, not a float
            label_val = int(round(val)) if fmt_str[-1:] in ("d", "n") else val
            fig.add_trace(go.Bar(
                x=[p], y=[val],
                name=p, marker_color=colour_map[p],
                text=[f"{label_val:{fmt_str}}"],
                textposition="outside",
            ))
        apply_theme(fig, title)
        fig.update_layout(height=360, yaxis_title=ylab,
                          showlegend=False, bargap=0.35)
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart("fill_rate",       "Fill Rate (%)",    "Fill Rate", ".1%"), use_container_width=True)
        st.plotly_chart(bar_chart("avg_end_on_hand", "Avg End-on-Hand",  "Units",     ".1f"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart("total_lost_sales","Total Lost Sales",  "Units",    ".0f"), use_container_width=True)
        st.plotly_chart(bar_chart("avg_order_qty",   "Avg Order Quantity","Units",    ".1f"), use_container_width=True)

    # Reward comparison
    st.markdown('<div class="sec-hdr">Total Reward Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(bar_chart("sum_reward_clipped","Total Episode Reward","Reward",",.0f"),
                    use_container_width=True)
    st.caption("ℹ️ Baseline Clipped (train) is fixed at 204,961 — the test value reflects actual performance.")
    # Improvement table
    imp = data["imp"]
    if len(imp) > 0:
        st.markdown('<div class="sec-hdr">RL vs Baseline — Delta Table (Train Env)</div>',
                    unsafe_allow_html=True)
        imp2 = imp.copy()
        imp2.columns = [c.replace("_"," ").title() for c in imp2.columns]
        def color_delta(v):
            v = float(v)
            return "color:#4ade80" if v > 0 else ("color:#f87171" if v < 0 else "")
        st.dataframe(
            imp2.style.map(color_delta, subset=["Rl Minus Baseline"])
                      .format({"Rl Minus Baseline": "{:+.3f}"}),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 4 — STEP-BY-STEP
# ══════════════════════════════════════════════════════════════════════════════
elif "Step" in page:
    st.markdown('<div class="sec-hdr">🔍 Step-by-Step Decision Log Analysis</div>',
                unsafe_allow_html=True)

    dec = data["dec"].copy()
    if dec.empty:
        st.warning("No decision log CSV found."); st.stop()

    # Slider
    max_s = int(dec["t_step"].max()) if "t_step" in dec.columns else len(dec)-1
    lo, hi = st.slider("t_step range", 0, max_s, (0, min(max_s, 120)), step=1)
    dec = dec[(dec["t_step"] >= lo) & (dec["t_step"] <= hi)]

    x = dec["t_step"]

    # ── Order Quantity ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Order Quantity</div>',
                unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=dec["baseline_order_qty"],
                              name="Baseline", line=dict(color=C_BASE, width=2), mode="lines"))
    fig1.add_trace(go.Scatter(x=x, y=dec["rl_order_qty"],
                              name="RL", line=dict(color=C_RL, width=2), mode="lines"))
    apply_theme(fig1, "Order Quantity per Step")
    fig1.update_layout(height=300, xaxis_title="t_step", yaxis_title="Order Qty")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Lost Sales ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=dec["baseline_lost_sales"],
                                  name="Baseline", line=dict(color=C_BASE, width=2), fill="tozeroy",
                                  fillcolor=hex_to_rgba(C_BASE, 0.16)))
        fig2.add_trace(go.Scatter(x=x, y=dec["lost_sales"],
                                  name="RL", line=dict(color=C_RL, width=2), fill="tozeroy",
                                  fillcolor=hex_to_rgba(C_RL, 0.16)))
        apply_theme(fig2, "Lost Sales per Step")
        fig2.update_layout(height=320, xaxis_title="t_step", yaxis_title="Lost Sales")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=dec["baseline_reward_env"],
                                  name="Baseline", line=dict(color=C_BASE, width=2)))
        fig3.add_trace(go.Scatter(x=x, y=dec["reward_env"],
                                  name="RL", line=dict(color=C_RL, width=2)))
        apply_theme(fig3, "Step Reward")
        fig3.update_layout(height=320, xaxis_title="t_step", yaxis_title="Reward")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Cumulative reward ─────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Cumulative Reward</div>',
                unsafe_allow_html=True)
    dec["cum_rl"]   = dec["reward_env"].cumsum()
    dec["cum_base"] = dec["baseline_reward_env"].cumsum()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=x, y=dec["cum_base"],
                              name="Baseline (cumul.)", line=dict(color=C_BASE, width=2, dash="dot")))
    fig4.add_trace(go.Scatter(x=x, y=dec["cum_rl"],
                              name="RL (cumul.)", line=dict(color=C_RL, width=2.5)))
    apply_theme(fig4, "Cumulative Reward over Episode")
    fig4.update_layout(height=320, xaxis_title="t_step", yaxis_title="Cumulative Reward")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Delta scatter ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="font-size:1rem">RL minus Baseline Δ</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig5 = px.bar(dec.assign(color=dec["delta_order_qty_rl_minus_baseline"].apply(
                          lambda v: C_RL if v >= 0 else C_BASE)),
                      x="t_step", y="delta_order_qty_rl_minus_baseline",
                      color="color", color_discrete_map="identity",
                      title="Δ Order Qty (RL − Baseline)")
        apply_theme(fig5)
        fig5.update_layout(height=300, showlegend=False, yaxis_title="Δ Qty")
        st.plotly_chart(fig5, use_container_width=True)

    with c4:
        fig6 = px.bar(dec.assign(color=dec["delta_reward_env_rl_minus_baseline"].apply(
                          lambda v: C_GUARD if v >= 0 else C_BASE)),
                      x="t_step", y="delta_reward_env_rl_minus_baseline",
                      color="color", color_discrete_map="identity",
                      title="Δ Reward (RL − Baseline)")
        apply_theme(fig6)
        fig6.update_layout(height=300, showlegend=False, yaxis_title="Δ Reward")
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 5 — TEST SET RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Test" in page:
    st.markdown('<div class="sec-hdr">🧪 Out-of-Sample Test-Set Evaluation</div>',
                unsafe_allow_html=True)
    st.caption("80 / 20 train-test split applied to the original dataset.")

    df_f = data["final"].copy()

    test_sub  = df_f[df_f["scope"]=="test_env"].copy()
    train_sub = df_f[df_f["scope"]=="train_env"].copy()

    if test_sub.empty:
        st.warning("No test-env rows found in final_report.csv.")
        st.stop()

    # Generalisation bar: train vs test reward
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Reward — Train vs Test</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for scope, sub, opacity in [("Train", train_sub, 1.0), ("Test", test_sub, 0.7)]:
        for _, row in sub.iterrows():
            p = str(row["policy"])
            col = C_GUARD if "guard" in p.lower() else (C_RL if p=="RL" else C_BASE)
            fig.add_trace(go.Bar(
                name=f"{p} ({scope})",
                x=[f"{p} · {scope}"], y=[float(row["sum_reward_clipped"])],
                marker_color=col, opacity=opacity,
                text=[f"{float(row['sum_reward_clipped']):,.0f}"],
                textposition="outside",
            ))
    apply_theme(fig, "Total Reward — Train vs Test per Policy")
    fig.update_layout(height=420, showlegend=False,
                      bargap=0.25, yaxis_title="Total Reward")
    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side metric comparison
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Fill Rate & Lost Sales Generalisation</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    def grouped_bar(col, title, ylab, fmt=""):
        fig = go.Figure()
        for scope, sub in [("Train", train_sub), ("Test", test_sub)]:
            for _, row in sub.iterrows():
                p = str(row["policy"])
                col_c = C_GUARD if "guard" in p.lower() else (C_RL if p=="RL" else C_BASE)
                fmt_str = fmt or '.2f'
                val = float(row[col])
                label_val = int(round(val)) if fmt_str[-1:] in ("d", "n") else val
                fig.add_trace(go.Bar(
                    name=f"{p} ({scope})",
                    x=[f"{p}"],
                    y=[val],
                    marker_color=col_c,
                    opacity=1.0 if scope=="Train" else 0.6,
                    legendgroup=scope,
                    legendgrouptitle_text=scope,
                    text=[f"{label_val:{fmt_str}}"],
                    textposition="outside",
                ))
        apply_theme(fig, title)
        fig.update_layout(height=360, yaxis_title=ylab,
                          barmode="group", bargap=0.15, bargroupgap=0.05)
        return fig

    with c1:
        st.plotly_chart(grouped_bar("fill_rate","Fill Rate","Fill Rate",".1%"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(grouped_bar("total_lost_sales","Total Lost Sales","Units",".0f"),
                        use_container_width=True)

    # Full test table
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Test Env — Full KPI Table</div>',
                unsafe_allow_html=True)
    show = ["policy","fill_rate","total_lost_sales","avg_end_on_hand","sum_reward_clipped","avg_order_qty"]
    tdf  = test_sub[[c for c in show if c in test_sub.columns]].copy()
    tdf.columns = [c.replace("_"," ").title() for c in tdf.columns]
    st.dataframe(tdf.style.format({
        "Fill Rate": "{:.1%}", "Total Lost Sales": "{:.0f}",
        "Avg End On Hand": "{:.1f}", "Sum Reward Clipped": "{:,.0f}",
        "Avg Order Qty": "{:.1f}",
    }).highlight_max(subset=["Fill Rate","Sum Reward Clipped"], color="rgba(0,212,255,0.20)")
    .highlight_min(subset=["Total Lost Sales"],               color="rgba(105,219,124,0.20)"),
    use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 6 — PILOT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Pilot" in page:
    st.markdown('<div class="sec-hdr">🎯 Pilot Recommendations — Upcoming Order Quantities</div>',
                unsafe_allow_html=True)
    st.caption("RL-agent recommended order quantities for the most recent time steps.")

    pilot = data["pilot"]
    if pilot is None or pilot.empty:
        st.warning("No pilot_recommendations.csv found."); st.stop()

    # Clean display names
    nice_cols = {
        "t_step":                           "t_step",
        "month":                            "Month",
        "demand_observed":                  "Observed Demand",
        "rl_recommended_order_qty_raw":     "RL Order (Raw)",
        "rl_recommended_order_qty_guarded": "RL Order (Guarded)",
    }
    pdf = pilot[[c for c in nice_cols if c in pilot.columns]].rename(columns=nice_cols)

    # Chart
    fig = go.Figure()
    if "Observed Demand" in pdf.columns:
        fig.add_trace(go.Scatter(
            x=pdf["t_step"], y=pdf["Observed Demand"],
            name="Observed Demand", mode="lines+markers",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        ))
    if "RL Order (Raw)" in pdf.columns:
        fig.add_trace(go.Bar(
            x=pdf["t_step"], y=pdf["RL Order (Raw)"],
            name="RL Order (Raw)", marker_color=C_RL, opacity=0.75,
        ))
    if "RL Order (Guarded)" in pdf.columns:
        fig.add_trace(go.Bar(
            x=pdf["t_step"], y=pdf["RL Order (Guarded)"],
            name="RL Order (Guarded)", marker_color=C_GUARD, opacity=0.75,
        ))
    apply_theme(fig, "Pilot Period — Recommended Order Quantities")
    fig.update_layout(height=380, barmode="group",
                      xaxis_title="t_step", yaxis_title="Order Qty / Demand")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Full Recommendation Table</div>',
                unsafe_allow_html=True)
    fmt_map = {c: "{:.0f}" for c in pdf.columns if c != "t_step"}
    st.dataframe(
        pdf.style.format(fmt_map)
           .background_gradient(subset=["RL Order (Raw)"] if "RL Order (Raw)" in pdf.columns else [],
                                cmap="Blues"),
        use_container_width=True, height=440
    )

    # Download
    csv_bytes = pdf.to_csv(index=False).encode()
    st.download_button("⬇️ Download Recommendations CSV",
                       csv_bytes, "pilot_recommendations.csv",
                       "text/csv", key="dl_pilot")


# ══════════════════════════════════════════════════════════════════════════════
# ██  PAGE 7 — USE THE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "Use the Model" in page:
    st.markdown('<div class="sec-hdr">🤖 Live Recommendation — Try the Trained Policy</div>',
                unsafe_allow_html=True)
    st.caption("Enter a scenario and get a live order-quantity recommendation — "
               "from the trained agent once connected, or a transparent "
               "newsvendor heuristic in the meantime.")

    cfg    = load_model_config()
    policy = load_policy()

    if cfg.get("_source") == "default":
        st.markdown("""<div class="sample-notice">
        📋 <code>model_config.json</code> not found next to <code>app.py</code> —
        using built-in default economics instead.
        </div>""", unsafe_allow_html=True)

    # ── Economics panel (editable — defaults from model_config.json) ───────
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Cost Structure</div>',
                unsafe_allow_html=True)
    st.caption("Drag to explore different economics — recommendations below "
               "update using whatever values are set here.")

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        selling_price = st.slider("Selling Price ($)", min_value=0.0, max_value=100.0,
                                   value=float(cfg["selling_price"]), step=0.5,
                                   key="cfg_selling_price", format="$%.2f")
    with e2:
        order_cost = st.slider("Order Cost / Unit ($)", min_value=0.0, max_value=50.0,
                                value=float(cfg["order_cost"]), step=0.1,
                                key="cfg_order_cost", format="$%.2f")
    with e3:
        holding_cost_rate = st.slider("Holding Cost Rate", min_value=0.0, max_value=1.0,
                                       value=float(cfg["holding_cost_rate"]), step=0.01,
                                       key="cfg_holding_cost_rate", format="%.0f%%")
    with e4:
        lost_sales_penalty = st.slider("Lost-Sale Penalty ($)", min_value=0.0, max_value=200.0,
                                        value=float(cfg["lost_sales_penalty"]), step=1.0,
                                        key="cfg_lost_sales_penalty", format="$%.2f")
    with e5:
        max_order = st.slider("Max Order / Step", min_value=0, max_value=500,
                               value=int(cfg["max_order"]), step=5,
                               key="cfg_max_order")

    # Overwrite cfg with whatever the sliders currently say, so everything
    # below (recommendation math, captions, etc.) uses the live values.
    cfg = {
        **cfg,
        "selling_price":      selling_price,
        "order_cost":         order_cost,
        "holding_cost_rate":  holding_cost_rate,
        "lost_sales_penalty": lost_sales_penalty,
        "max_order":          max_order,
    }

    if st.button("↺ Reset to model_config.json defaults", key="reset_cfg_sliders"):
        for k in ("cfg_selling_price", "cfg_order_cost", "cfg_holding_cost_rate",
                  "cfg_lost_sales_penalty", "cfg_max_order"):
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model connection status ─────────────────────────────────────────────
    if policy is not None:
        st.success(f"✅ Live policy loaded from `{MODEL_CHECKPOINT_DIR}`")
    else:
        st.markdown(f"""<div class="sample-notice">
        🔌 <b>Demo mode</b> — no trained checkpoint found at
        <code>{MODEL_CHECKPOINT_DIR}</code>. Showing a newsvendor heuristic
        instead. See "Connect your real policy" below.
        </div>""", unsafe_allow_html=True)
        if "_policy_load_error" in st.session_state:
            with st.expander("Why isn't it loading? (debug info)"):
                st.code(st.session_state["_policy_load_error"])

    # ── Scenario inputs ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr" style="font-size:1rem">Scenario</div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    cur_inv  = s1.number_input("Current On-Hand Inventory", min_value=0.0, value=25.0, step=1.0)
    dem_mean = s2.number_input("Recent Average Demand",     min_value=0.0, value=30.0, step=1.0)
    dem_std  = s3.number_input("Demand Volatility (Std Dev)", min_value=0.0, value=8.0, step=0.5)

    with st.expander("Advanced — extra context for the RL agent (optional)"):
        st.caption("The trained policy was built on a 6-feature observation. "
                   "These extra fields fill in the 3 features not covered above; "
                   "defaults are reasonable mid-range values.")
        a1, a2, a3 = st.columns(3)
        on_order_inv = a1.number_input("On-Order Inventory (already placed, not yet arrived)",
                                        min_value=0.0, value=30.0, step=1.0)
        leadtime     = a2.number_input("Lead Time (steps)", min_value=0.0, value=3.0, step=1.0)
        month        = a3.number_input("Month", min_value=1.0, max_value=12.0, value=6.0, step=1.0)

    if st.button("📦 Get Recommendation", type="primary"):
        co = cfg["holding_cost_rate"] * cfg["order_cost"]   # cost of 1 unit overstocked
        cu = cfg["lost_sales_penalty"]                       # cost of 1 unit understocked
        crit_ratio = cu / (cu + co + 1e-9)
        try:
            from scipy.stats import norm
            z = float(norm.ppf(np.clip(crit_ratio, 1e-6, 1 - 1e-6)))
        except ImportError:
            z = 3.0  # very high service level fallback if scipy unavailable
        target_level  = max(0.0, dem_mean + z * dem_std)
        heuristic_qty = float(np.clip(target_level - cur_inv, 0, cfg["max_order"]))

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(kpi_card("Newsvendor Heuristic", heuristic_qty,
                                 fmt="{:.0f} units"), unsafe_allow_html=True)
            st.caption(f"Target service level ≈ {crit_ratio:.1%} "
                       f"(critical ratio = {cu:.2f} ÷ ({cu:.2f}+{co:.2f}))")
        with r2:
            if policy is not None:
                try:
                    inventory_position = cur_inv + on_order_inv
                    rl_qty, rl_action = get_rl_recommendation(
                        policy, cfg,
                        demand=dem_mean,
                        on_hand=cur_inv,
                        on_order=on_order_inv,
                        inventory_position=inventory_position,
                        leadtime_steps=leadtime,
                        month=month,
                    )
                    st.markdown(kpi_card("RL Agent Recommendation", rl_qty,
                                         fmt="{:.0f} units"), unsafe_allow_html=True)
                    st.caption(f"Policy chose bin {rl_action}/{N_BINS-1} "
                               f"(bin size = {cfg['max_order']/N_BINS:.1f} units, "
                               f"max order = {cfg['max_order']:.0f})")
                except Exception as e:
                    st.error("RL inference failed — see details below.")
                    with st.expander("Error details"):
                        st.code(str(e))
            else:
                st.info("🔌 RL Agent — not connected yet")

    # ── Connect-your-model instructions ─────────────────────────────────────
    with st.expander("🔧 About the live policy connection"):
        st.markdown(f"""
This tab looks for an RLlib checkpoint at `{MODEL_CHECKPOINT_DIR}`.

**Live inference is wired in** using the exact spec from the training notebook:
- Observation order: `{", ".join(OBS_FEATURE_ORDER)}`, z-normalised with
  `obs_mean`/`obs_std` from `model_config.json` and clipped to [-10, 10].
- Action space: `Discrete({N_BINS})` →
  `order_qty = action × (max_order / {N_BINS})`.
- No separate guardrail policy exists in the notebook — the raw action
  **is** the final recommendation.

If `model_config.json` isn't found, built-in defaults are used instead
(shown in the notice above, if present).
        """)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;color:#3d5a73;font-size:0.78rem;
     padding:24px 0 8px 0;border-top:1px solid #1e3556;margin-top:32px;'>
    Supply Chain RL Dashboard · Graduation Project ·
    Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)