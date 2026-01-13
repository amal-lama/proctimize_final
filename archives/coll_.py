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

    if clusters:
        st.markdown(f"**Identified {len(clusters)} correlated clusters at threshold {combo_threshold}:**")
        for i, cluster in enumerate(clusters, start=1):
            st.write(f"Cluster {i}: {cluster}")
    else:
        st.info("No correlated clusters found for the given threshold.")

    if clusters and st.button("Apply Combination", key="btn_apply_combination"):
        df_combined, combos_info = combine_clusters(
            joined_df, feature_cols, clusters,
            method=method,
            drop_original=drop_original
        )

        st.success("Combination complete. New COMBO columns created.")
        st.markdown("**Combination Mapping:**")
        st.dataframe(combos_info)

        st.session_state["df_combined_multicollinearity"] = df_combined

        st.markdown("**Sample of updated dataset (top 20 rows):**")
        st.dataframe(df_combined.head(20))

-------------------- PCA Approach Tab ----------------------------------------
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
        df_pca, explained_df, n_components = apply_pca(
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



#----------------------------------------


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
