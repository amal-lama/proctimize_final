# === NEW PAGE: Multicollinearity Handler =====================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# OLD / REUSED STYLE SNIPPETS (from your EDA page)
# --------------------------------------------------------------------
st.set_page_config(page_title="ProcTimize - Multicollinearity", layout="wide")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #001E96 !important;
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #1ABC9C;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Multicollinearity Handling")

# --------------------------------------------------------------------
# HELPER FUNCTIONS  (NEW CODE)
# --------------------------------------------------------------------
def get_key_columns(df: pd.DataFrame):
    """
    Try to pull geo/date/ZIP/DMA/target from session_state.
    If missing, let user choose from dropdowns.
    """
    st.subheader("Step 0: Confirm Key Columns")

    all_cols = df.columns.tolist()

    # Read from session_state if available
    geo_default = st.session_state.get("geo_column", None)
    date_default = st.session_state.get("date_column", None)
    zip_default = st.session_state.get("ZIP_column", None)
    dma_default = st.session_state.get("DMA_column", None)
    dep_default = st.session_state.get("dependent_variable", None)

    col1, col2, col3 = st.columns(3)
    with col1:
        geo_column = st.selectbox(
            "Modeling Granularity Column (HCP / NPI / Physician ID)",
            [None] + all_cols,
            index=(all_cols.index(geo_default) + 1) if geo_default in all_cols else 0,
            key="mc_geo_column"
        )
    with col2:
        date_column = st.selectbox(
            "Date Column",
            [None] + all_cols,
            index=(all_cols.index(date_default) + 1) if date_default in all_cols else 0,
            key="mc_date_column"
        )
    with col3:
        dependent_variable = st.selectbox(
            "Dependent Variable (KPI / Sales)",
            [None] + all_cols,
            index=(all_cols.index(dep_default) + 1) if dep_default in all_cols else 0,
            key="mc_dep_column"
        )

    col4, col5 = st.columns(2)
    with col4:
        zip_column = st.selectbox(
            "ZIP Column (optional)",
            [None] + all_cols,
            index=(all_cols.index(zip_default) + 1) if zip_default in all_cols else 0,
            key="mc_zip_column"
        )
    with col5:
        dma_column = st.selectbox(
            "DMA Column (optional)",
            [None] + all_cols,
            index=(all_cols.index(dma_default) + 1) if dma_default in all_cols else 0,
            key="mc_dma_column"
        )

    # Persist back to session_state for other pages to reuse
    if geo_column is not None:
        st.session_state["geo_column"] = geo_column
    if date_column is not None:
        st.session_state["date_column"] = date_column
    if zip_column is not None:
        st.session_state["ZIP_column"] = zip_column
    if dma_column is not None:
        st.session_state["DMA_column"] = dma_column
    if dependent_variable is not None:
        st.session_state["dependent_variable"] = dependent_variable

    return geo_column, date_column, zip_column, dma_column, dependent_variable


def get_candidate_features(df, geo_column, date_column, zip_column, dma_column, dependent_variable):
    """
    Return the list of numeric feature columns to check for multicollinearity.
    Excludes geo/date/ZIP/DMA/target.
    """
    non_feature_cols = {
        col for col in [geo_column, date_column, zip_column, dma_column, dependent_variable]
        if col is not None
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in non_feature_cols]

    return feature_cols, list(non_feature_cols)


def compute_corr_pairs(df, feature_cols, threshold):
    """
    Compute highly correlated pairs (absolute correlation >= threshold).
    Returns sorted list of tuples: (feature1, feature2, corr_value).
    """
    corr_matrix = df[feature_cols].corr().abs()
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1, f2 = feature_cols[i], feature_cols[j]
            corr_val = corr_matrix.loc[f1, f2]
            if corr_val >= threshold:
                pairs.append((f1, f2, corr_val))

    # Sort by descending corr_value
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs, corr_matrix


def remove_correlated_features(df, feature_cols, dependent_variable, threshold):
    """
    Removal approach:
    - For each pair with |corr| >= threshold, drop the less important feature.
    - Importance defined as |corr(feature, dependent)| if dependent is provided.
      Otherwise, fallback to dropping the alphabetically later one.
    """
    _, corr_matrix = compute_corr_pairs(df, feature_cols, threshold)

    to_drop = set()
    kept = set(feature_cols)

    if dependent_variable is not None and dependent_variable in df.columns:
        target_corr = df[feature_cols + [dependent_variable]].corr()[dependent_variable].abs()
    else:
        target_corr = None

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1, f2 = feature_cols[i], feature_cols[j]
            if f1 in to_drop or f2 in to_drop:
                continue

            corr_val = corr_matrix.loc[f1, f2]
            if corr_val >= threshold:
                # Decide which one to drop
                if target_corr is not None:
                    # Drop the one with LOWER |corr(feature, target)|
                    c1 = target_corr.get(f1, 0)
                    c2 = target_corr.get(f2, 0)
                    if c1 >= c2:
                        drop_feature = f2
                    else:
                        drop_feature = f1
                else:
                    # Fallback: drop alphabetically later one
                    drop_feature = max(f1, f2)

                to_drop.add(drop_feature)
                if drop_feature in kept:
                    kept.remove(drop_feature)

    df_reduced = df.copy()
    df_reduced = df_reduced.drop(columns=list(to_drop))

    return df_reduced, list(kept), list(to_drop)


def find_corr_clusters(df, feature_cols, threshold):
    """
    Build clusters of correlated features using graph connected components.
    Edge between features if |corr| >= threshold.
    """
    corr_matrix = df[feature_cols].corr().abs()
    n = len(feature_cols)
    adj = {f: set() for f in feature_cols}

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = feature_cols[i], feature_cols[j]
            if corr_matrix.loc[f1, f2] >= threshold:
                adj[f1].add(f2)
                adj[f2].add(f1)

    visited = set()
    clusters = []

    for f in feature_cols:
        if f not in visited and adj[f]:
            stack = [f]
            cluster = set()
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    cluster.add(node)
                    stack.extend(list(adj[node] - visited))
            if len(cluster) > 1:
                clusters.append(sorted(list(cluster)))

    return clusters


# def combine_clusters(df, feature_cols, clusters, method="sum", drop_original=True):
#     """
#     Combination approach:
#     - For each cluster, create a new combined column (sum or mean).
#     - Optionally drop original columns.
#     """
#     df_combined = df.copy()
#     new_cols_info = []

#     for idx, cluster in enumerate(clusters, start=1):
#         if method == "mean":
#             combined_series = df_combined[cluster].mean(axis=1)
#             method_str = "mean"
#         else:
#             combined_series = df_combined[cluster].sum(axis=1)
#             method_str = "sum"

#         new_col_name = f"COMBO_{idx}"
#         df_combined[new_col_name] = combined_series
#         new_cols_info.append({"combo_name": new_col_name,
#                               "features": cluster,
#                               "method": method_str})

#         if drop_original:
#             df_combined = df_combined.drop(columns=cluster)

#     return df_combined, pd.DataFrame(new_cols_info)

def combine_clusters(df, feature_cols, clusters, new_names, method="sum", drop_original=True):
    """
    Combination approach with user-defined names:
    - new_names: list of names for each cluster
    """
    df_combined = df.copy()
    new_cols_info = []

    if len(new_names) != len(clusters):
        raise ValueError("Number of new names must match number of clusters.")

    for idx, cluster in enumerate(clusters):
        user_col_name = new_names[idx]

        if method == "mean":
            combined_series = df_combined[cluster].mean(axis=1)
            method_str = "mean"
        else:
            combined_series = df_combined[cluster].sum(axis=1)
            method_str = "sum"

        df_combined[user_col_name] = combined_series
        new_cols_info.append({
            "combo_name": user_col_name,
            "features": cluster,
            "method": method_str
        })

        if drop_original:
            df_combined = df_combined.drop(columns=cluster)

    return df_combined, pd.DataFrame(new_cols_info)



def apply_pca(df, feature_cols, variance_threshold=0.9):
    """
    PCA approach with contribution analysis:
    - Standardize feature columns
    - Fit PCA and keep components until variance_threshold is reached
    - Compute per-channel contribution to each PC
    - Extract top 3 contributing channels per PC
    """

    # ===== 1. Standardize features =====
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===== 2. Fit full PCA =====
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # ===== 3. Choose number of components =====
    n_components = int(np.searchsorted(cumulative_var, variance_threshold) + 1)
    n_components = max(1, min(n_components, len(feature_cols)))

    # ===== 4. Extract selected PCs =====
    pcs = X_pca[:, :n_components]
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pcs_df = pd.DataFrame(pcs, columns=pc_cols, index=df.index)

    # ===== 5. Rebuild final dataframe =====
    non_feature_cols = [c for c in df.columns if c not in feature_cols]
    df_pca = pd.concat([df[non_feature_cols], pcs_df], axis=1)

    # ===== 6. Explained variance table =====
    explained_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained_var))],
        "Explained_Variance_Ratio": explained_var,
        "Cumulative_Variance": cumulative_var
    })

    # ===== 7. Compute PCA LOADINGS =====
    loadings = pca.components_[:n_components]  # shape: (k, num_features)

    # ===== 8. Compute contribution (%) of each channel to each PC =====
    squared_loadings = loadings ** 2
    contrib_pct = squared_loadings / squared_loadings.sum(axis=1, keepdims=True)

    contrib_df = pd.DataFrame(
        contrib_pct,
        columns=feature_cols,
        index=[f"PC{i+1}" for i in range(n_components)]
    )

    # ===== 9. Extract TOP 3 contributing channels for each PC =====
    top3_dict = {}
    for pc in contrib_df.index:
        top3 = contrib_df.loc[pc].sort_values(ascending=False).head(3)
        top3_dict[pc] = pd.DataFrame({
            "Channel": top3.index,
            "Contribution (%)": (top3.values * 100).round(2)
        })

    # RETURN EVERYTHING
    return df_pca, explained_df, n_components, contrib_df, top3_dict





# --------------------------------------------------------------------
# MAIN APP LOGIC (NEW)
# --------------------------------------------------------------------

uploaded_df = None

uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)

if uploaded_df is not None:
    st.success("Using uploaded file.")
    joined_df = uploaded_df

elif "joined_output_df" in st.session_state and not st.session_state["joined_output_df"].is_empty():
    st.success("Using dataset from previous page (session state).")
    joined_df = st.session_state["joined_output_df"].to_pandas()

else:
    st.warning("⚠️ No dataset found. Upload a file OR complete the join step first.")
    st.stop()
# if "joined_output_df" not in st.session_state or st.session_state["joined_output_df"].is_empty():
#     st.warning("⚠️ No joined dataset found. Please complete the file join step first.")
#     st.stop()

# # Convert Polars DF from previous page to Pandas
# joined_pl = st.session_state["joined_output_df"]
# joined_df = joined_pl.to_pandas()

st.markdown("### Preview of Joined Dataset")
st.dataframe(joined_df.head(50))

geo_column, date_column, zip_column, dma_column, dependent_variable = get_key_columns(joined_df)

feature_cols, non_feature_cols = get_candidate_features(
    joined_df, geo_column, date_column, zip_column, dma_column, dependent_variable
)

if not feature_cols:
    st.error("No numeric feature columns found for multicollinearity analysis.")
    st.stop()

st.write(f"**Number of candidate features for multicollinearity check:** {len(feature_cols)}")
st.write(f"**Candidate features:** {feature_cols}")

# --------------------------------------------------------------------
# 1. Correlation Overview (NEW)
# --------------------------------------------------------------------
st.subheader("Step 1: Correlation Overview")

corr_threshold_view = st.slider(
    "Correlation threshold to *highlight* strong relationships (for overview only)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
    key="overview_threshold"
)
#------
corr = joined_df[feature_cols].corr().fillna(0)

fig = px.imshow(
    corr,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
)

fig.update_traces(
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    hovertemplate="Corr: %{z:.2f}<extra></extra>"
)

fig.update_layout(
    width=700,
    height=700,
    xaxis_title="Predictors",
    yaxis_title="Predictors",
    xaxis=dict(tickangle=45),
)

st.plotly_chart(fig, use_container_width=True)

#------

# corr_matrix_full = joined_df[feature_cols].corr()

# fig_corr = px.imshow(
#     corr_matrix_full,
#     text_auto=False,
#     color_continuous_scale="RdBu",
#     zmin=-1,
#     zmax=1,
#     title="Correlation Matrix of Candidate Features"
# )
# fig_corr.update_layout(width=700, height=700)
# st.plotly_chart(fig_corr, use_container_width=True)

high_pairs, _ = compute_corr_pairs(joined_df, feature_cols, corr_threshold_view)
if high_pairs:
    st.markdown(f"**Pairs with |corr| ≥ {corr_threshold_view}:**")
    st.dataframe(
        pd.DataFrame(high_pairs, columns=["Feature 1", "Feature 2", "Abs Corr"])
    )
else:
    st.info("No feature pairs exceed the selected overview threshold.")

# --------------------------------------------------------------------
# 2. Multicollinearity Treatment Options (NEW)
# --------------------------------------------------------------------
st.subheader("Step 2: Select a strategy")

tab_removal, tab_combination, tab_pca = st.tabs(["Removal Approach", "Combination Approach", "PCA Approach"])


# -------------------- Removal Approach Tab -----------------------------------
with tab_removal:
    st.markdown("#### Removal Approach (Drop one of the highly correlated channels)")

    removal_threshold = st.slider( 
        "Correlation threshold for **removal** (dynamic)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        key="removal_threshold"
    )

    st.write("Channels with |corr| above this threshold will be considered for removal.")

    pairs_removal, _ = compute_corr_pairs(joined_df, feature_cols, removal_threshold)
    if pairs_removal:
        st.markdown(f"**Pairs flagged for removal (|corr| ≥ {removal_threshold}):**")
        st.dataframe(pd.DataFrame(pairs_removal, columns=["Feature 1", "Feature 2", "Abs Corr"]))
    else:
        st.info("No pairs exceed the current removal threshold.")

    if st.button("Apply Removal", key="btn_apply_removal"):
        df_reduced, kept_cols, dropped_cols = remove_correlated_features(
            joined_df, feature_cols, dependent_variable, removal_threshold
        )

        st.success(
            f"Removal complete. Dropped {len(dropped_cols)} features. "
            f"{len(kept_cols)} features remain."
        )
        st.markdown("**Dropped Features:**")
        st.write(dropped_cols)
        st.markdown("**Kept Feature Columns:**")
        st.write(kept_cols)

        st.session_state["joined_output_df"] = df_reduced
        st.markdown("**Sample of updated dataset (top 20 rows):**")
        st.dataframe(df_reduced.head(20))


# -------------------- Combination Approach Tab --------------------------------
with tab_combination:
    st.markdown("#### Combination Approach (Combine highly correlated channels)")

    combo_threshold = st.slider(

        "Correlation threshold for **combination** (dynamic)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        key="combination_threshold"
    )

    method = st.selectbox(
        "Aggregation method for each correlated cluster",
        ["sum", "mean"],
        index=0
    )

    drop_original = st.checkbox(
        "Drop original correlated features after creating combined columns",
        value=True
    )
    

    clusters = find_corr_clusters(joined_df, feature_cols, combo_threshold)

    clusters = find_corr_clusters(joined_df, feature_cols, combo_threshold)

    if clusters:
        st.markdown(
            f"**Identified {len(clusters)} correlated clusters at threshold {combo_threshold}:**"
        )

        user_combo_names = []

        for i, cluster in enumerate(clusters, start=1):
            st.write(f"Cluster {i}: {cluster}")

            default_name = f"COMBO_{i}"
            combo_name = st.text_input(
                f"Enter new column name for Cluster {i}",
                value=default_name,
                key=f"combo_name_cluster_{i}"
            )
            user_combo_names.append(combo_name)

    else:
        st.info("No correlated clusters found for the given threshold.")
        user_combo_names = []

    # Only show apply button if we actually have clusters
    if clusters and st.button("Apply Combination", key="btn_apply_combination"):
        df_combined, combos_info = combine_clusters(
            joined_df,
            feature_cols,
            clusters,
            new_names=user_combo_names,   # <-- pass user names here
            method=method,
            drop_original=drop_original
        )

        st.success(
            "Combination complete. New combined columns created: "
            + ", ".join(combos_info["combo_name"].tolist())
        )

        st.markdown("**Combination Mapping:**")
        st.dataframe(combos_info)

        # (Optional) store back to session_state for later steps
        st.session_state["joined_output_df"] = df_combined
        st.markdown("**Sample of updated dataset (top 20 rows):**")
        st.dataframe(df_combined.head(20))


#-------------------- PCA Approach Tab ----------------------------------------
with tab_pca:
    st.markdown("#### PCA Approach (Convert correlated channels into orthogonal components)")

    variance_threshold = st.slider(
        "Target cumulative explained variance for PCA",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
        key="pca_variance_threshold"
    )

    st.write(
        "PCA will choose the minimum number of components that explain at least "
        f"{variance_threshold:.2f} of the variance in the selected features."
    )

    if st.button("Run PCA", key="btn_run_pca"):
        df_pca, explained_df, n_components, contrib_df, top3= apply_pca(
            joined_df, feature_cols, variance_threshold=variance_threshold
        )

        st.success(f"PCA complete. Selected **{n_components}** components to reach the variance target.")

        st.markdown("**Explained Variance by Component:**")
        st.dataframe(explained_df)

        fig_var = px.bar(
            explained_df,
            x="Component",
            y="Explained_Variance_Ratio",
            title="Explained Variance Ratio per Component"
        )
        st.plotly_chart(fig_var, use_container_width=True)

        fig_cum = px.line(
            explained_df,
            x="Component",
            y="Cumulative_Variance",
            markers=True,
            title="Cumulative Explained Variance"
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.session_state["df_pca_multicollinearity"] = df_pca

        st.markdown("**Sample of PCA-transformed dataset (top 20 rows):**")
        st.dataframe(df_pca.head(20))

        st.subheader("Contribution of Original Channels to Each Principal Component")
        st.dataframe(contrib_df.style.format("{:.2%}"))

        # === TOP 3 Contributors ===
        st.subheader("Top 3 Contributing Channels per Component")

        st.dataframe(contrib_df.style.format("{:.2%}"))
    
        st.subheader("Top 3 Contributors per PC")
        for pc, table in top3.items():
            st.markdown(f"### {pc}")
            st.dataframe(table)

