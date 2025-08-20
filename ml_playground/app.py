# ml_playground/app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from core import (
    list_datasets, load_dataset, load_csv, split_features_target,
    list_models, get_param_space, build_model,
    list_clusterers, get_cluster_param_space, build_clusterer,
    train_and_eval, cluster_and_eval, pca_project
)

st.set_page_config(page_title="ML Playground", layout="wide")

# ---------- Sidebar: Task ----------
st.sidebar.header("Task")
task = st.sidebar.radio("Wähle Aufgabe", ["Classification", "Clustering"])

# ---------- Sidebar: Data ----------
st.sidebar.header("Datenquelle")
data_src = st.sidebar.radio("Quelle", ["Built-in dataset", "Upload CSV"])

X, y, ds_name, has_labels = None, None, None, False

if data_src == "Built-in dataset":
    name = st.sidebar.selectbox("Dataset", list_datasets())
    X, y = load_dataset(name)
    ds_name = name
    has_labels = y is not None
else:
    file = st.sidebar.file_uploader("CSV auswählen", type=["csv"])
    if file is not None:
        has_target = st.sidebar.checkbox("Hat Target-Spalte?")
        if has_target:
            df = pd.read_csv(file)
            target_col = st.sidebar.selectbox("Target", ["<choose>"] + list(df.columns))
            if target_col != "<choose>":
                X, y = split_features_target(df, target_col)
                has_labels = True
                ds_name = "uploaded.csv"
        else:
            X, y = load_csv(file, target=None)  # nur numerische Features
            ds_name = "uploaded.csv"
            has_labels = False

# ---------- Sidebar: Model / Params ----------
if task == "Classification":
    st.sidebar.header("Classifier")
    model_name = st.sidebar.selectbox("Modell", list_models())
    space = get_param_space(model_name)
else:
    st.sidebar.header("Clusterer")
    model_name = st.sidebar.selectbox("Clusterer", list_clusterers())
    space = get_cluster_param_space(model_name)

# dynamische Controls aus Param-Raum
def render_controls(param_space: dict):
    params = {}
    for pname, spec in param_space.items():
        ptype = spec.get("type", "int")
        label = spec.get("label", pname)
        if ptype == "int":
            params[pname] = st.sidebar.slider(
                label, int(spec["min"]), int(spec["max"]),
                int(spec.get("default", spec["min"])),
                int(spec.get("step", 1))
            )
        elif ptype == "float":
            params[pname] = st.sidebar.slider(
                label, float(spec["min"]), float(spec["max"]),
                float(spec.get("default", spec["min"]))
            )
        elif ptype == "select":
            params[pname] = st.sidebar.selectbox(label, spec["choices"], index=spec.get("index", 0))
        else:
            params[pname] = spec.get("default")
    return params

params = render_controls(space) if space else {}
# ---------- Sidebar: Projection ----------
st.sidebar.header("Projection")
pca_dims = st.sidebar.radio("PCA", [2, 3], index=0, horizontal=True)
marker_size = st.sidebar.slider("Marker size", 4, 18, 7)
marker_opacity = st.sidebar.slider("Marker opacity", 0.2, 1.0, 0.85)
plot_height = st.sidebar.slider("Plot height (px)", 400, 900, 700)

# ---------- Main ----------
st.title("ML Playground")
if X is None:
    st.info("Bitte Datenquelle wählen / CSV laden.")
    st.stop()

st.markdown(
    f"**Dataset:** `{ds_name}`  •  **Shape:** `{X.shape}`  "
    + (f"•  **#Klassen:** `{len(np.unique(y))}`" if (has_labels and task == "Classification") else "")
)

# --- Task run ---
if task == "Classification":
    if not has_labels:
        st.error("Für Classification wird eine Zielspalte (Labels) benötigt.")
        st.stop()

    model = build_model(model_name, params)
    out = train_and_eval(X, y, model)
    st.subheader("Ergebnis (Test)")
    st.write(f"**Modell:** {model_name}  •  **Accuracy:** {out['accuracy']:.4f}")

elif task == "Clustering":
    clusterer = build_clusterer(model_name, params)
    out = cluster_and_eval(X, clusterer, y_true=y if has_labels else None)

    st.subheader("Ergebnis (Clustering)")
    met_lines = [
        f"**#Cluster (≠ -1):** {out['n_clusters']}",
        f"**Silhouette:** {out['silhouette']:.3f}" if out['silhouette'] is not None else "",
        f"**Calinski-Harabasz:** {out['calinski_harabasz']:.1f}" if out['calinski_harabasz'] is not None else "",
        f"**Davies-Bouldin:** {out['davies_bouldin']:.3f}" if out['davies_bouldin'] is not None else "",
    ]
    if out.get("ari") is not None:
        met_lines.append(f"**ARI vs. true:** {out['ari']:.3f}")
    if out.get("nmi") is not None:
        met_lines.append(f"**NMI vs. true:** {out['nmi']:.3f}")
    st.markdown("  •  ".join([m for m in met_lines if m]))

# --- PCA Plot ---
Z = pca_project(X, n_components=pca_dims)  # immer skaliert + PCA
color = None
title = f"PCA Projection ({pca_dims}D) – {ds_name}"

if task == "Classification" and has_labels:
    color = pd.Series(y).astype(str)
elif task == "Clustering" and out.get("labels") is not None:
    color = pd.Series(out["labels"]).astype(str)

if pca_dims == 2:
    fig = px.scatter(
        x=Z[:, 0], y=Z[:, 1], color=color,
        labels={"x": "PC 1", "y": "PC 2", "color": "Label/Cluster"},
        title=title, opacity=marker_opacity
    )
    fig.update_traces(marker=dict(size=marker_size))
else:
    fig = px.scatter_3d(
        x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], color=color,
        labels={"x": "PC 1", "y": "PC 2", "z": "PC 3", "color": "Label/Cluster"},
        title=title, opacity=marker_opacity
    )
    fig.update_traces(marker=dict(size=marker_size))

fig.update_layout(height=plot_height, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
