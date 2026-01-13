# tab_mapping.py
import io
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from helper import detect_date_columns_by_sampling, to_polars_date_safe

def render_mapping():
    tab = st.container()
    with tab:
        tab1, tab2 = st.tabs(["Data Mapping", "Data Profiling Placeholder"])
    # Only fill tab1 here; tab2 will be owned by profiling module
    with tab1:
        uploaded_files = st.session_state.get("uploaded_files")
        if uploaded_files:
            st.session_state["current_step"] = 1
            if len(uploaded_files) == 1:
                _single_file_mapping(uploaded_files[0])
            else:
                _multi_file_mapping(uploaded_files)

def _single_file_mapping(file):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Standardize columns for a single file")
        df = pd.read_csv(file)
        st.write(f"Currently editing: `{file.name}`")
        st.dataframe(df[:3], hide_index=True)
        selected_cols = st.multiselect(
            f"Select columns from `{file.name}`:",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key="select_cols_0"
        )
        df = df[selected_cols]
        rename_columns = pd.DataFrame({"Rename columns if necessary": df.columns})
        edited_renamed_df = st.data_editor(
            rename_columns,
            column_config={"Rename columns if necessary": st.column_config.TextColumn()},
            num_rows="fixed",
            use_container_width=True,
            key="rename_editor_0",
            hide_index=True
        )
        rename_dict = dict(zip(df.columns, edited_renamed_df["Rename columns if necessary"]))
        df = df.rename(columns=rename_dict)

        detected_dates = detect_date_columns_by_sampling(df)
        date_cols = st.multiselect("Select date columns", options=df.columns, default=detected_dates)
        for date_col in date_cols:
            date_format = st.selectbox(
                f"Select current date format for `{date_col}`:",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                index=None,
                key=f"date_format_0_{date_col}"
            )
            if date_format:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                    st.success(f"Date column `{date_col}` standardized!")
                except Exception as e:
                    st.error(f"Failed to parse `{date_col}`: {e}")

        npi_col = "account_npi"
        if npi_col in df.columns:
            s = df[npi_col].astype(str).str.strip()
            s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
            s = s.str.replace(r"\.0$", "", regex=True)
            mask_valid = s.str.isnumeric() & s.str.len().eq(10) & s.str.startswith(("1", "2"))
            df["npi_clean"] = s.where(mask_valid)
            df["npi_valid"] = df["npi_clean"].notna()

        str_cols = df.select_dtypes(include=["object", "string"]).columns
        df[str_cols] = df[str_cols].fillna("")
        st.session_state["df_final"] = df
        st.session_state["current_step"] = 3

    with col2:
        st.markdown("### Final Preview")
        st.dataframe(df, hide_index=True)

def _multi_file_mapping(uploaded_files):
    st.markdown("### Standardize columns for multiple files")
    renamed_columns_list = []
    for i, file in enumerate(uploaded_files):
        col1, col2 = st.columns([2, 3])
        with col1:
            with st.expander(f"Currently editing: `{file.name}`"):
                df = pd.read_csv(file)
                st.dataframe(df[:3], hide_index=True)
                selected_cols = st.multiselect(
                    f"Select columns from `{file.name}`:",
                    df.columns.tolist(),
                    default=df.columns.tolist(),
                    key=f"select_cols_{i}"
                )
                df = df[selected_cols]
                rename_columns = pd.DataFrame({"Rename columns if necessary": df.columns})
                edited_renamed_df = st.data_editor(
                    rename_columns,
                    column_config={"Rename columns if necessary": st.column_config.TextColumn()},
                    num_rows="fixed",
                    use_container_width=True,
                    key=f"rename_editor_{i}",
                    hide_index=True
                )
                rename_dict = dict(zip(df.columns, edited_renamed_df["Rename columns if necessary"]))
                df = df.rename(columns=rename_dict)
                renamed_columns_list.append(set(df.columns))

                detected_dates = detect_date_columns_by_sampling(df)
                date_cols = st.multiselect(
                    "Select date columns", options=df.columns,
                    default=detected_dates, key=f"date_col_{i}"
                )
                for date_col in date_cols:
                    date_format = st.selectbox(
                        f"Select format for `{date_col}`:",
                        ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                        index=None, key=f"date_format_{i}_{date_col}"
                    )
                    if date_format:
                        try:
                            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                            st.success(f"Standardized `{date_col}`!")
                        except Exception as e:
                            st.error(f"Could not parse `{date_col}`: {e}")

                npi_col = "account_npi"
                if npi_col in df.columns:
                    s = df[npi_col].astype(str).str.strip()
                    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
                    s = s.str.replace(r"\.0$", "", regex=True)
                    mask_valid = s.str.isnumeric() & s.str.len().eq(10) & s.str.startswith(("1", "2"))
                    df[npi_col] = s.where(mask_valid)
                str_cols = df.select_dtypes(include=["object", "string"]).columns.drop([npi_col], errors="ignore")
                df[str_cols] = df[str_cols].fillna("")

                if "df_final_list" not in st.session_state:
                    st.session_state["df_final_list"] = []
                if len(st.session_state["df_final_list"]) <= i:
                    st.session_state["df_final_list"].append(df)
                else:
                    st.session_state["df_final_list"][i] = df

        st.session_state["current_step"] = 2

        with col2:
            st.markdown(f"Data Frame Preview: `{file.name}`")
            st.dataframe(df, hide_index=True)

    with st.expander("Merge Files"):
        st.subheader("Select Merge Strategy")
        common_columns = set.intersection(*renamed_columns_list) if renamed_columns_list else set()
        merge_strategy = st.radio("Merge type:", ["Vertical Stack", "Horizontal Join"])
        if merge_strategy == "Horizontal Join":
            join_keys = st.multiselect("Select join key(s):", list(common_columns))
            join_type = st.selectbox("Join type:", ["inner", "left", "right", "outer"])
        else:
            join_keys = join_type = None

        if st.button("Merge"):
            if merge_strategy in ["vertical", "Vertical Stack"]:
                df_final = pd.concat(st.session_state["df_final_list"], ignore_index=True, sort=False)
            elif merge_strategy in ["horizontal", "Horizontal Join"]:
                if not join_keys:
                    st.warning("Join key must be provided for horizontal joins.")
                    st.stop()
                df_final = st.session_state["df_final_list"][0]
                for df in st.session_state["df_final_list"][1:]:
                    df_final = pd.merge(df_final, df, on=join_keys, how=join_type)
            else:
                st.error("Invalid merge strategy. Choose 'vertical' or 'horizontal'.")
                st.stop()

            if join_keys:
                for col_name in join_keys:
                    if col_name in df_final.columns:
                        df_final = df_final[df_final[col_name].notnull()]

            for col in df_final.select_dtypes(include="object").columns:
                if "date" in col.lower():
                    try:
                        df_final[col] = pd.to_datetime(df_final[col], errors='coerce')
                    except Exception:
                        pass

            df_final_pl = to_polars_date_safe(df_final)
            st.session_state["df_final"] = df_final_pl
            st.session_state["merge_done"] = True

            st.markdown("### Final Data Frame")
            st.dataframe(df_final_pl)
            try:
                merged_csv = df_final_pl.to_pandas().to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download merged CSV",
                    data=merged_csv,
                    file_name="merged_final.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(f"Could not prepare file for download: {e}")
        st.session_state["current_step"] = 3
