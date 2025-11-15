import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
import streamlit as st

st.set_page_config(layout="wide")

st.title("Time Series Pattern Finder – Stumpy")

# --------------------------------------------------------------------
# Sidebar configuration
# --------------------------------------------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV time series", type=["csv"])

# Make synthetic demo deterministic
np.random.seed(0)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    ts = df.iloc[:, 0].astype(float).values
    st.sidebar.success("Using uploaded CSV (first column as time series)")
else:
    st.sidebar.info("Using synthetic sine wave demo")
    ts = np.sin(np.linspace(0, 20, 3000)) * (1 + 0.3 * np.random.randn(3000))

# Window size
m = int(st.sidebar.slider("Window Size (m)", 30, 500, 120))

# Number of motifs
k_motifs = st.sidebar.slider("Number of motifs (top-K)", 1, 10, 3)

# NEW: Exclusion radius multiplier
exclusion_mult = st.sidebar.slider(
    "Diversity: Motif Exclusion Radius Multiplier",
    1.0,
    10.0,
    3.0,
    help="Higher values force motifs to come from more distant, distinct areas of the time series.",
)

# Playback settings
play_speed = st.sidebar.slider("Play speed (sec per frame)", 0.01, 1.0, 0.15)
auto_play = st.sidebar.checkbox("Auto-play subsequences", value=False)

# Convert to list for cache stability
ts_list = ts.tolist()

# --------------------------------------------------------------------
# Cached matrix profile computation
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def compute_mp_once(ts_list, m):
    ts_arr = np.array(ts_list, dtype=float)
    mp = stumpy.stump(ts_arr, m)
    return mp

# --------------------------------------------------------------------
# Cached motif computation with exclusion radius
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_top_k_motifs(P_list, I_list, m, k, exclusion_mult):

    P = np.array(P_list, dtype=float).copy()
    I = np.array(I_list, dtype=int)

    motif_starts = []
    neighbor_sets = []

    exclusion_radius = int(m * exclusion_mult)

    for _ in range(k):
        motif_idx = int(np.argmin(P))
        nearest = int(I[motif_idx])

        motif_starts.append(motif_idx)
        neighbor_sets.append(np.array([nearest]))

        # Mask motif window ± exclusion_radius
        start = max(0, motif_idx - exclusion_radius)
        end = min(len(P), motif_idx + m + exclusion_radius)
        P[start:end] = np.inf

        # Mask nearest neighbor window ± exclusion_radius
        start_n = max(0, nearest - exclusion_radius)
        end_n = min(len(P), nearest + m + exclusion_radius)
        P[start_n:end_n] = np.inf

    return motif_starts, neighbor_sets


st.write("Computing matrix profile and motifs (cached where possible)…")

# Compute MP once
mp = compute_mp_once(ts_list, m)
P = mp[:, 0]
I = mp[:, 1].astype(int)

# Serialize for caching
P_list = P.tolist()
I_list = I.tolist()

motif_indices, neighbor_sets = compute_top_k_motifs(
    P_list, I_list, m, k_motifs, exclusion_mult
)

# --------------------------------------------------------------------
# Motif selection
# --------------------------------------------------------------------
motif_labels = [
    f"Motif #{i+1} (start idx = {motif_indices[i]})"
    for i in range(len(motif_indices))
]

selected_label = st.selectbox("Select motif set", motif_labels)
selected_idx = motif_labels.index(selected_label)

current_motif_start = int(motif_indices[selected_idx])
current_neighbors = neighbor_sets[selected_idx]

# --------------------------------------------------------------------
# Plotting function
# --------------------------------------------------------------------
plot_placeholder = st.empty()

subseq = st.slider(
    "Subsequence index",
    min_value=0,
    max_value=len(ts) - m - 1,
    value=current_motif_start,
)

def plot_frame(sub_idx: int, dynamic_neighbors: bool = False):
    fig, axs = plt.subplots(3, 1, figsize=(12, 6))

    # 1) Raw time series
    axs[0].plot(ts, color="black")
    axs[0].axvspan(sub_idx, sub_idx + m, color="green", alpha=0.35,
                   label="Current subsequence")

    if dynamic_neighbors:
        nn = int(mp[sub_idx, 1])
        axs[0].axvspan(nn, nn + m, color="gray", alpha=0.35, label="Nearest neighbor")
    else:
        nn = int(current_neighbors[0])
        axs[0].axvspan(nn, nn + m, color="gray", alpha=0.35, label="Motif neighbor")

    axs[0].legend(loc="upper right")
    axs[0].set_title("Raw Time Series")

    # 2) Matrix profile
    axs[1].plot(P, color="black")
    axs[1].axvline(sub_idx, color="green", linestyle="--")
    axs[1].set_title("Matrix Profile")

    # 3) Motif overlay
    current_seq = ts[sub_idx : sub_idx + m]
    axs[2].plot(current_seq, color="green", label="Current subsequence")

    if dynamic_neighbors:
        nn = int(mp[sub_idx, 1])
        axs[2].plot(ts[nn : nn + m], color="gray", label="Nearest neighbor")
    else:
        nn = int(current_neighbors[0])
        axs[2].plot(ts[nn : nn + m], color="gray", label="Motif neighbor")

    axs[2].legend(loc="upper right")
    axs[2].set_title("Motif / Neighbor Overlay")

    fig.tight_layout()
    return fig

# --------------------------------------------------------------------
# Animation / static output
# --------------------------------------------------------------------
if auto_play:
    for i in range(subseq, len(ts) - m):
        fig = plot_frame(i, dynamic_neighbors=True)
        plot_placeholder.pyplot(fig)
        time.sleep(play_speed)
else:
    fig = plot_frame(subseq, dynamic_neighbors=False)
    plot_placeholder.pyplot(fig)
