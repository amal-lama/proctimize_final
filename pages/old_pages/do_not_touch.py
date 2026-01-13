import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import re
import io

st.set_page_config(page_title="ProcTimize", layout="wide")
st.title("DATA INGESTION")

from datetime import datetime, timedelta, date

RE_NPI = re.compile(r"^[12]\d{9}$")

def luhn_valid_npi(npi: str) -> bool:
    if not RE_NPI.fullmatch(npi):
        return False
    base, chk = npi[:9], ord(npi[9]) - 48
    s = 24  # CMS '80840' tweak
    for i in range(9):
        d = ord(base[8 - i]) - 48
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return ((10 - (s % 10)) % 10) == chk

# Polars-based date detection by sampling (optional helper)
def detect_date_columns_by_sampling(
    df: pl.DataFrame,
    sample_size: int = 100,
    threshold: float = 0.8,
    formats: list[str] | None = None
) -> list[str]:
    if formats is None:
        formats = ["%d/%m/%Y", "%Y/%m/%d", "%Y/%d/%m", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y-%d-%m", "%Y-%m-%d"]
    cols: list[str] = []
    n = min(sample_size, df.height)
    samp = df.sample(n=n, with_replacement=False) if df.height > 0 else df
    for c, dt in df.schema.items():
        if dt != pl.Utf8:
            continue
        best = 0.0
        for fmt in formats:
            parsed = samp.select(
                pl.col(c).str.strptime(pl.Date, format=fmt, strict=False, exact=True).alias("_p")
            ).get_column("_p")
            ratio = parsed.is_not_null().sum() / max(1, len(parsed))
            if ratio > best:
                best = ratio
        if best >= threshold:
            cols.append(c)
    return cols

# Optional: explicit date parsing utility to ensure typed dates downstream
def parse_dates(df: pl.DataFrame, selected: dict[str, str]) -> pl.DataFrame:
    exprs = []
    for col, fmt in selected.items():
        exprs.append(
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, format=fmt, strict=True, cache=True).alias(col)
        )
    return df.with_columns(exprs)

def reset_full_workflow():
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in [
        "df_final", "df_final_list", "merge_done",
        "df_transformed", "transform_complete",
        "numerical_config_dict", "categorical_config_dict",
    ]:
        st.session_state.pop(k, None)
    st.session_state.pop("uploaded_files", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state["current_step"] = 0
    for k in list(st.session_state.keys()):
        if k.startswith((
            "select_cols_", "rename_editor_", "date_format_", "date_col_",
            "date_cols_", "npi_", "groupby_", "analysis_", "granularity_",
            "download_", "kpi_", "modify_granularity_", "normalize_"
        )):
            st.session_state.pop(k, None)


# ------------------------------------------------------------------------------
def last_working_day(year: int, month: int, work_days: int = 5) -> date:
    # Compute last day of month then roll back to last working day (Mon=0..Sun=6)
    if month == 12:
        month = 0
        year += 1
    last_day = date(year, month + 1, 1) - timedelta(days=1)
    if work_days == 5:
        while last_day.weekday() > 4:
            last_day -= timedelta(days=1)
    return last_day

def last_week_apportion(
    df: pl.DataFrame,
    date_col: str,
    kpi_cols: list[str],
    work_days: int = 5
) -> pl.DataFrame:
    """
    Proportionately allocates KPIs of the last (cross-boundary) week in a month to the next month.
    - For rows where the week crosses month-end, subtract carryover from current row and add a new row
      on the first day of the next month with the carried KPI values.
    Assumptions:
    - date_col is pl.Date or pl.Datetime (if datetime, time-of-day is preserved where relevant).
    """
    # Ensure date column is Date (drop time) for month logic
    df = df.with_columns([
        pl.col(date_col).dt.date().alias("_date_only")
    ])

    # Compute month key and last working day per month
    df = df.with_columns([
        pl.col("_date_only").dt.strftime("%Y-%m").alias("_ym")
    ])

    # Build lookup of last working day for each year-month present
    months = (
        df.select(pl.col("_date_only").dt.year().alias("_yr"),
                  pl.col("_date_only").dt.month().alias("_mo"))
          .unique()
          .rows()
    )
    lwd_map = {}
    for (yr, mo) in months:
        lwd_map[f"{yr:04d}-{mo:02d}"] = last_working_day(yr, mo, work_days)

    df = df.with_columns([
        pl.col("_ym").map_elements(lambda s: lwd_map.get(s), return_dtype=pl.Date).alias("_last_working_date")
    ])

    # Days from current date to last working date inclusive
    df = df.with_columns([
        (pl.col("_last_working_date").cast(pl.Date) - pl.col("_date_only").cast(pl.Date)).dt.total_days().cast(pl.Int32).alias("_diff_days"),
        (pl.lit(1) + (pl.col("_last_working_date").cast(pl.Date) - pl.col("_date_only").cast(pl.Date)).dt.total_days()).cast(pl.Int32).alias("_day_diff_inclusive")
    ])

    # Carryover only when the week is not full within the month (i.e., last partial week)
    # day_diff_inclusive < work_days -> carry a proportional part into next month
    df = df.with_columns([
        pl.when(pl.col("_day_diff_inclusive") < work_days)
          .then((work_days - pl.col("_day_diff_inclusive")) / work_days)
          .otherwise(0.0)
          .alias("_carry_frac")
    ])

    # Create adjusted KPI columns to subtract from current and add to next month
    adjusted_cols = [f"_adj_{c}" for c in kpi_cols]
    df = df.with_columns([
        (pl.col(c).cast(pl.Float64) * pl.col("_carry_frac")).alias(f"_adj_{c}") for c in kpi_cols
    ])

    # Subtract adjusted portion from current rows
    df_current = df.with_columns([
        (pl.col(c).cast(pl.Float64) - pl.col(f"_adj_{c}")).alias(c) for c in kpi_cols
    ])

    # Prepare new rows for next month (first day)
    # Next month first day = month_end.rollforward + 1 day, or month_start of next month
    df_new = (
        df.filter(
            pl.any_horizontal([pl.col(ac) > 0 for ac in adjusted_cols])
        )
        .with_columns([
            # First day of next month
            pl.col("_date_only").dt.month_end().dt.offset_by("1d").alias("_next_month_first")
        ])
        .with_columns([
            # Set date_col to next month first day
            pl.when(True).then(pl.col("_next_month_first")).alias(date_col)
        ])
        .with_columns([
            # New KPI values are the adjusted portion
            pl.col(f"_adj_{c}").alias(c) for c in kpi_cols
        ])
    )

    # Select original columns for current rows
    current_cols = [c for c in df.columns if not c.startswith("_")]
    df_current = df_current.select(current_cols)

    # Select same columns for new rows (ensure presence)
    df_new = df_new.select(current_cols)

    # Combine
    out = pl.concat([df_current, df_new], how="vertical")
    return out

# --- CONFIGURATION ---
def kpi_table(df: pl.DataFrame):
    numerical_config_dict: dict[str, str] = {}
    categorical_config_dict: dict[str, str] = {}
    try:
        st.success("File loaded successfully!")
        st.write("### Preview of Data")
        # Streamlit currently expects pandas/pyarrow for rich editing; convert small head only
        st.dataframe(df.head(10).to_pandas(), use_container_width=True)

        # Polars dtypes for filtering
        numeric_dtypes = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
        category_dtypes = [pl.Utf8, pl.Categorical]

        # Column lists by dtype
        dtypes = df.dtypes  # list[pl.DataType]
        cols = df.columns
        num_cols = [c for c, t in zip(cols, dtypes) if any(t == dt for dt in numeric_dtypes)]
        cat_cols = [c for c, t in zip(cols, dtypes) if any(t == dt for dt in category_dtypes)]

        st.subheader("🔢 Numerical Columns")
        selected_num_cols = st.multiselect("Select numerical columns:", num_cols)

        edited_num_df = None
        if selected_num_cols:
            # Build a tiny pandas table for st.data_editor
            num_ops_df = pl.DataFrame(
                {"Column": selected_num_cols, "Operation": [""] * len(selected_num_cols)}
            ).to_pandas()
            operation_options = ["average", "sum", "product", "min", "max"]
            edited_num_df = st.data_editor(
                num_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=operation_options)
                },
                num_rows="fixed",
                key="numerical_editor"
            )

        st.subheader("🔠 Categorical Columns")
        selected_cat_cols = st.multiselect("Select categorical columns:", cat_cols)

        edited_cat_df = None
        if selected_cat_cols:
            cat_ops_df = pl.DataFrame(
                {"Column": selected_cat_cols, "Operation": [""] * len(selected_cat_cols)}
            ).to_pandas()
            cat_operation_options = ["count", "distinct count"]
            edited_cat_df = st.data_editor(
                cat_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=cat_operation_options)
                },
                num_rows="fixed",
                key="categorical_editor"
            )

        if selected_num_cols or selected_cat_cols:
            if st.button("💾 Save All Configurations"):
                if edited_num_df is not None and not edited_num_df.empty:
                    numerical_config_dict = edited_num_df.set_index("Column")["Operation"].to_dict()
                if edited_cat_df is not None and not edited_cat_df.empty:
                    categorical_config_dict = edited_cat_df.set_index("Column")["Operation"].to_dict()
                st.success("✅ Configuration saved!")
                return numerical_config_dict, categorical_config_dict

    except Exception as e:
        st.error(f"Error processing KPI Table: {e}")
st.markdown(
    """
    <style>

     /* Sidebar with deep blue gradient */
    section[data-testid="stSidebar"] {
        background-color: #001E96 !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #001E96 !important;
        filter: brightness(1.1);
    }
    </style>
    """, unsafe_allow_html=True)


# Define steps and current step (0-based index)
steps = ["Upload", "Standardize", "Merge", "Filter", "Pivot", "Save"]


if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

# Calculate progress percentage
progress_percent = int((st.session_state["current_step"] / (len(steps) - 1)) * 100)

# background: linear-gradient(to right, #4f91ff, #007aff);
# Display the progress bar using HTML and CSS
st.markdown(f"""
<style>
.progress-container {{
    width: 100%;
    margin-top: 20px;
}}

.progress-bar {{
    height: 10px;
    background: linear-gradient(to right, #001E96, #007aff);
    border-radius: 10px;
    width: {progress_percent}%;
    transition: width 0.4s ease-in-out;
}}

.track {{
    background-color: #e0e0e0;
    height: 10px;
    width: 100%;
    border-radius: 10px;
    margin-bottom: 10px;
}}

.step-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    color: #333;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
</style>

<div class="progress-container">
    <div class="track">
        <div class="progress-bar"></div>
    </div>
    <div class="step-labels">
        {''.join([f"<span>{step}</span>" for step in steps])}
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()
c_reset, _ = st.columns([1, 3])
with c_reset:
    if st.button("Reset workflow (Mapping + Profiling)"):
        reset_full_workflow()
        st.success("Workflow reset. Start again from Upload.")
        st.rerun()

col1, col2  = st.columns(2)

with col1:
    st.markdown("Add description of the functionality of this page later, this is currently being edited by Abdul")
with col2:

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    @st.cache_data(show_spinner=False)
    def _read_csv_polars_from_upload(name: str, size: int, file_bytes: bytes) -> pl.DataFrame:
        # Keyed by (name, size) so re-uploads with same content reuse cache
        bio = io.BytesIO(file_bytes)
        bio.seek(0)
        return pl.read_csv(
            bio,
            try_parse_dates=False,
            infer_schema_length=10000,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )

    with st.expander("Upload CSV Files"):
        uploaded_files = st.file_uploader(
            "Select CSV file(s) to upload:",
            type="csv",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}"
        )
        st.session_state["uploaded_files"] = uploaded_files

        # Immediately read and store Polars DataFrames if files are provided
        if uploaded_files:
            dfs: list[pl.DataFrame] = []
            for f in uploaded_files:
                # f is an UploadedFile with attributes: name, size, getvalue()
                df_i = _read_csv_polars_from_upload(f.name, f.size, f.getvalue())
                dfs.append(df_i)
            st.session_state["df_final_list"] = dfs
            # Optional: also keep names for UI
            st.session_state["uploaded_names"] = [f.name for f in uploaded_files]

    with st.expander("Connect to Azure Blob Storage"):
        st.markdown(
            "This section will allow you to securely connect to an Azure Blob Storage account and load files directly."
        )

        st.subheader("Azure Credentials")
        blob_account_name = st.text_input(
            "Azure Storage Account Name",
            disabled=True,
            placeholder="e.g., mystorageaccount"
        )
        blob_container_name = st.text_input(
            "Container Name",
            disabled=True,
            placeholder="e.g., marketing-data"
        )
        blob_sas_token = st.text_input(
            "SAS Token",
            type="password",
            disabled=True,
            placeholder="Your secure token"
        )

        st.subheader("File Selection")
        st.selectbox(
            "Browse files in container",
            options=["-- Coming soon --"],
            disabled=True
        )
        st.button("Refresh File List", disabled=True)

        st.markdown("---")
        st.info("You will be able to load files directly from your Azure Blob Storage container using the credentials above.")

        # TO IMPLEMENT LATER (ensure Polars output):
        # from azure.storage.blob import ContainerClient
        # container_url = f"https://{blob_account_name}.blob.core.windows.net/{blob_container_name}?{blob_sas_token}"
        # container_client = ContainerClient.from_container_url(container_url)
        # blobs = [blob.name for blob in container_client.list_blobs()]
        # blob_client = container_client.get_blob_client(blob_name)
        # stream = blob_client.download_blob().readall()  # bytes
        # df_final = pl.read_csv(io.BytesIO(stream), try_parse_dates=False)

    # Keep a canonical placeholder; actual frames live in session state
    df_final = None

tab1, tab2 = st.tabs(["Data Mapping", "Data Profiling"])


# --- DATA MAPPING PAGE ---
with tab1:
    uploaded_files = st.session_state.get("uploaded_files")
    if uploaded_files:
        st.session_state["current_step"] = 1
        st.write(st.session_state["current_step"])

        # Helper: read file to Polars (use cached loader from earlier snippet if available)
        def _read_one_polars(file) -> pl.DataFrame:
            return pl.read_csv(
                file,
                try_parse_dates=False,
                infer_schema_length=10000,
                null_values=["", "nan", "NaN", "NULL", "None"],
            )

        if len(uploaded_files) == 1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Standardize columns for a single file (Polars)")
                file = uploaded_files[0]
                df = _read_one_polars(file)

                st.write(f"**Currently editing:** `{file.name}`")
                st.dataframe(df.head(3).to_pandas(), hide_index=True)

                # Step 1: Column Selection
                selected_cols = st.multiselect(
                    f"Select columns from `{file.name}`:",
                    df.columns,
                    default=df.columns,
                    key="select_cols_0"
                )
                df = df.select(selected_cols)

                # Step 2a: Column Renaming (edit small table via pandas)
                rename_columns = pl.DataFrame({"Rename columns if necessary": df.columns}).to_pandas()
                column_config = {
                    "Rename columns if necessary": st.column_config.TextColumn()
                }
                edited_renamed_df = st.data_editor(
                    rename_columns,
                    column_config=column_config,
                    num_rows="fixed",
                    use_container_width=True,
                    key="rename_editor_0",
                    hide_index=True
                )
                rename_dict = dict(zip(df.columns, edited_renamed_df["Rename columns if necessary"]))
                df = df.rename(rename_dict)  # Polars rename
                # Suggestion: Validate duplicate names here

                # Step 2b: Date Standardization (Polars detection + parsing)
                detected_dates = detect_date_columns_by_sampling(df)
                date_cols = st.multiselect("Select date columns", options=df.columns, default=detected_dates)
                selected_date_formats = {}
                for date_col in date_cols:
                    date_format = st.selectbox(
                        f"Select current date format for `{date_col}`:",
                        ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                        index=None,
                        key=f"date_format_0_{date_col}"
                    )
                    if date_format:
                        selected_date_formats[date_col] = date_format
                if selected_date_formats:
                    try:
                        df = parse_dates(df, selected_date_formats)
                        st.success("Date columns standardized!")
                    except Exception as e:
                        st.error(f"Failed to parse selected date columns: {e}")

                # NPI cleaning (Polars)
                npi_col = "account_npi"  # adjust if renamed above
                if npi_col in df.columns:
                    df = (
                        df.with_columns(
                            pl.col(npi_col).cast(pl.Utf8).str.strip_chars()
                              .str.replace(r"[^0-9]", "", literal=True).alias("npi_digits")
                        )
                        .with_columns(
                            (pl.col("npi_digits").str.len_chars() == 10)
                            & (pl.col("npi_digits").str.slice(0, 1).is_in(["1", "2"]))
                            .alias("npi_format_valid")
                        )
                    )
                    # Suggestion: add Luhn validity via annotate_luhn and keep canonical npi_digits column

                # Final clean-up: avoid blanket empty-string fills; keep nulls
                # If needed, only trim whitespace
                # df = df.with_columns([pl.col(c).str.strip() for c in df.columns if df.schema[c] == pl.Utf8])

                st.session_state["df_final"] = df
                st.session_state["current_step"] = 3

            with col2:
                st.markdown("### Final Preview")
                st.dataframe(df.head(50).to_pandas(), hide_index=True)

        else:
            st.markdown("### Standardize columns for multiple files")
            renamed_columns_list = []
            df_list: list[pl.DataFrame] = []

            for i, file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 3])
                with col1:
                    with st.expander(f"**Currently editing:** `{file.name}`"):
                        df_i = _read_one_polars(file)
                        st.dataframe(df_i.head(3).to_pandas(), hide_index=True)

                        selected_cols = st.multiselect(
                            f"Select columns from `{file.name}`:",
                            df_i.columns,
                            default=df_i.columns,
                            key=f"select_cols_{i}"
                        )
                        df_i = df_i.select(selected_cols)

                        rename_columns = pl.DataFrame({"Rename columns if necessary": df_i.columns}).to_pandas()
                        column_config = {"Rename columns if necessary": st.column_config.TextColumn()}
                        edited_renamed_df = st.data_editor(
                            rename_columns,
                            column_config=column_config,
                            num_rows="fixed",
                            use_container_width=True,
                            key=f"rename_editor_{i}",
                            hide_index=True
                        )
                        rename_dict = dict(zip(df_i.columns, edited_renamed_df["Rename columns if necessary"]))
                        df_i = df_i.rename(rename_dict)
                        renamed_columns_list.append(set(df_i.columns))

                        detected_dates = detect_date_columns_by_sampling(df_i)
                        date_cols = st.multiselect(
                            "Select date columns",
                            options=df_i.columns,
                            default=detected_dates,
                            key=f"date_col_{i}"
                        )
                        selected_date_formats_i = {}
                        for date_col in date_cols:
                            date_format = st.selectbox(
                                f"Select format for `{date_col}`:",
                                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                                index=None,
                                key=f"date_format_{i}_{date_col}"
                            )
                            if date_format:
                                selected_date_formats_i[date_col] = date_format
                        if selected_date_formats_i:
                            try:
                                df_i = parse_dates(df_i, selected_date_formats_i)
                                st.success("Standardized selected date columns!")
                            except Exception as e:
                                st.error(f"Could not parse selected date columns: {e}")

                        npi_col = "account_npi"
                        if npi_col in df_i.columns:
                            df_i = (
                                df_i.with_columns(
                                    pl.col(npi_col).cast(pl.Utf8).str.strip_chars()
                                      .str.replace(r"[^0-9]", "", literal=True).alias(npi_col)
                                )
                                .with_columns(
                                    (pl.col(npi_col).str.len_chars() == 10)
                                    & (pl.col(npi_col).str.slice(0, 1).is_in(["1", "2"]))
                                    .alias("npi_format_valid")
                                )
                            )

                        # Avoid filling nulls with ""; keep nulls semantic
                        # df_i = df_i  # optional string trim as in single-file

                        if len(df_list) <= i:
                            df_list.append(df_i)
                        else:
                            df_list[i] = df_i

                    st.session_state["current_step"] = 2

                with col2:
                    st.markdown(f"**Data Frame Preview**: `{file.name}`")
                    st.dataframe(df_i.head(50).to_pandas(), hide_index=True)

            # Persist list for merge step
            st.session_state["df_final_list"] = df_list

            with st.expander("Merge Files"):
                st.subheader("Select Merge Strategy")

                common_columns = set.intersection(*renamed_columns_list) if renamed_columns_list else set()
                merge_strategy = st.radio("Merge type:", ["Vertical Stack", "Horizontal Join"])

                if merge_strategy == "Horizontal Join":
                    join_keys = st.multiselect("Select join key(s):", list(common_columns))
                    join_type = st.selectbox("Join type:", ["inner", "left", "right", "outer"])
                else:
                    join_keys = None
                    join_type = None

                if st.button("Merge"):
                    if merge_strategy in ["vertical", "Vertical Stack"]:
                        # Optionally align schemas: pl.concat with how="vertical_relaxed" tolerates mismatches
                        df_final = pl.concat(st.session_state["df_final_list"], how="vertical_relaxed")

                    elif merge_strategy in ["horizontal", "Horizontal Join"]:
                        if not join_keys:
                            st.warning("Join key must be provided for horizontal joins.")
                            st.stop()
                        frames = st.session_state["df_final_list"]
                        df_final = frames[0]
                        # Ensure key dtypes are consistent across frames
                        key_dtypes = {k: df_final.schema[k] for k in join_keys}
                        for k, dt in key_dtypes.items():
                            df_final = df_final.with_columns(pl.col(k).cast(dt))
                        for df_next in frames[1:]:
                            for k, dt in key_dtypes.items():
                                df_next = df_next.with_columns(pl.col(k).cast(dt))
                            df_final = df_final.join(df_next, on=join_keys, how=join_type.lower())

                        # Drop rows with null join keys if desired
                        # for k in join_keys:
                        #     df_final = df_final.filter(pl.col(k).is_not_null())

                    else:
                        st.error("Invalid merge strategy. Choose 'vertical' or 'horizontal'.")
                        st.stop()

                    # Optional post-processing: do not auto-parse dates by name; rely on earlier parsing
                    st.session_state["df_final"] = df_final
                    st.session_state["merge_done"] = True

                    st.markdown("### Final Data Frame")
                    st.dataframe(df_final.head(100).to_pandas())

                    try:
                        csv_bytes = df_final.write_csv().encode("utf-8")
                        st.download_button(
                            "Download merged CSV",
                            data=csv_bytes,
                            file_name="merged_final.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.warning(f"Could not prepare file for download: {e}")

                st.session_state["current_step"] = 3

# --- DATA PROFILING PAGE ---
with tab2:
    st.markdown("### Data Profiling")

    # Require a completed merge
    if not st.session_state.get("merge_done") or "df_final" not in st.session_state:
        st.info("Merge files first (Vertical Stack or Horizontal Join), then return to profiling.")
        st.stop()

    # Get merged dataframe (Polars)
    df_any = st.session_state["df_final"]
    if not isinstance(df_any, pl.DataFrame):
        # If earlier code accidentally put pandas here, convert once:
        df_any = pl.from_pandas(df_any)
        st.session_state["df_final"] = df_any
    dfp: pl.DataFrame = df_any

    # --- Overview ---
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{dfp.height:,}")
    c2.metric("Columns", f"{dfp.width:,}")
    # Estimate memory size in MB if available
    try:
        mem_mb = dfp.estimated_size("mb")
        c3.metric("Memory", f"{mem_mb:.2f} MB")
    except Exception:
        c3.metric("Memory", "n/a")
    st.dataframe(dfp.head(10).to_pandas(), use_container_width=True)

    # --- Schema / dtypes ---
    st.subheader("Schema")
    schema_df = pl.DataFrame({"column": list(dfp.schema.keys()), "dtype": [str(t) for t in dfp.schema.values()]})
    st.dataframe(schema_df.to_pandas(), use_container_width=True)

    # --- Missingness ---
    st.subheader("No HCP, Date Row removal")
    # Candidate detection
    cols = dfp.columns
    npi_candidates = [c for c in cols if "npi" in c.lower()]
    # For date candidates, include columns typed as Date/Datetime
    date_candidates = [c for c, t in dfp.schema.items() if t in (pl.Date, pl.Datetime)] + \
                      [c for c in cols if "date" in c.lower()]

    npi_col_rm = st.selectbox(
        "NPI column for removal rule:",
        options=cols,
        index=(cols.index(npi_candidates[0]) if npi_candidates else 0)
    )
    date_col_rm = st.selectbox(
        "Date column for removal rule:",
        options=cols,
        index=(cols.index(date_candidates[0]) if date_candidates else 0)
    )

    if st.button("Remove rows where BOTH NPI and Date are missing"):
        # Produce a boolean mask in Polars: NPI is null/empty AND date is null
        # Normalize NPI empties
        dfp2 = dfp.with_columns([
            pl.col(npi_col_rm).cast(pl.Utf8).alias("_npi_str"),
        ]).with_columns([
            pl.when(pl.col("_npi_str").is_null() | (pl.col("_npi_str").str.strip_chars() == ""))
              .then(True).otherwise(False).alias("_npi_na")
        ])

        # Coerce date to a Date/Datetime temporarily as needed, then check null
        if dfp2.schema[date_col_rm] not in (pl.Date, pl.Datetime):
            dfp2 = dfp2.with_columns([
                pl.col(date_col_rm).cast(pl.Utf8)
                  .str.to_datetime(strict=False, infer_format=True)
                  .alias("_date_tmp")
            ])
            date_is_na = pl.col("_date_tmp").is_null()
            date_col_check = "_date_tmp"
        else:
            date_is_na = pl.col(date_col_rm).is_null()
            date_col_check = date_col_rm

        dfp2 = dfp2.with_columns([
            date_is_na.alias("_date_na")
        ])

        # Drop rows where both missing
        dfp2 = dfp2.with_columns([
            (pl.col("_npi_na") & pl.col("_date_na")).alias("_drop_row")
        ])
        dropped = int(dfp2.select(pl.col("_drop_row").sum()).item())
        df_clean = dfp2.filter(~pl.col("_drop_row")).drop(["_npi_str", "_npi_na", "_date_na", "_drop_row"] + (["_date_tmp"] if date_col_check == "_date_tmp" else []))

        st.session_state["df_final"] = df_clean
        st.success(f"Dropped {dropped:,} rows where BOTH `{npi_col_rm}` and `{date_col_rm}` were missing.")
        st.dataframe(df_clean.head(10).to_pandas(), use_container_width=True)

    # --- NPI Format check ---
    st.subheader("NPI Format check")
    npi_col = st.selectbox(
        "Select the column to check for NPI format:",
        options=dfp.columns,
        help="Choose the column containing NPI numbers"
    )

    # Clean to digits and validate format in Polars
    df_npi = (
        dfp.select([
            pl.col(npi_col).cast(pl.Utf8).str.strip_chars().str.replace(r"[^0-9]", "", literal=True).alias("_npi_digits")
        ])
        .with_columns([
            (pl.col("_npi_digits").str.len_chars() == 10)
            & (pl.col("_npi_digits").str.slice(0, 1).is_in(["1", "2"]))
            .alias("_npi_format_valid")
        ])
    )

    n_valid = int(df_npi.select(pl.col("_npi_format_valid").sum()).item())
    n_total = df_npi.height
    n_invalid = n_total - n_valid

    st.write(f"Format-valid NPIs (10 digits, starts 1/2): {n_valid:,}")
    st.write(f"Format-invalid NPIs: {n_invalid:,}")

    if n_invalid > 0:
        # Sample invalid NPIs
        invalid_sample = (
            df_npi.filter(~pl.col("_npi_format_valid"))
                 .select("_npi_digits")
                 .unique()
                 .head(10)
                 .to_pandas()
        )
        st.write("Sample format-invalid NPIs:")
        st.dataframe(invalid_sample)



                    