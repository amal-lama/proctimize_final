import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import requests
import io
import json
import time
from datetime import datetime,timedelta,date
from typing import List, Dict
import re


RE_NPI = re.compile(r"^[12]\d{9}$")


def luhn_valid_npi(npi: str) -> bool:
    if not npi or not RE_NPI.fullmatch(npi):
        return False
    base, chk = npi[:9], ord(npi[9]) - 48
    s = 24
    for i in range(9):
        d = ord(base[8 - i]) - 48
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return ((10 - (s % 10)) % 10) == chk

def detect_date_columns_by_sampling(
    df: pl.DataFrame,
    sample_size: int = 200,
    threshold: float = 0.8,
    formats: list[str] | None = None
) -> list[str]:
    if formats is None:
        formats = ["%d/%m/%Y", "%Y/%m/%d", "%Y/%d/%m", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y-%d-%m", "%Y-%m-%d"]
    if df.height == 0:
        return []
    n = min(sample_size, df.height)
    samp = df.sample(n=n, with_replacement=False)

    date_cols: list[str] = []
    for col, dt in df.schema.items():
        if dt != pl.Utf8:
            continue
        best_success = 0.0
        best_min = None
        best_max = None
        for fmt in formats:
            parsed = samp.select(
                pl.col(col).str.strptime(pl.Date, format=fmt, strict=False, exact=True).alias("_p")
            ).get_column("_p")
            if len(parsed) == 0:
                continue
            success = parsed.is_not_null().sum() / len(parsed)
            if success > best_success:
                best_success = success
                # Guard for realistic calendar years
                parsed_non_null = parsed.drop_nulls()
                if len(parsed_non_null) > 0:
                    pmin, pmax = parsed_non_null.min(), parsed_non_null.max()
                    best_min, best_max = pmin, pmax
        if best_success >= threshold and best_min is not None and best_max is not None:
            if 1900 <= best_min.year <= 2100 and 1900 <= best_max.year <= 2100:
                date_cols.append(col)
    return date_cols


def reset_full_workflow():
    st.cache_data.clear()
    st.cache_resource.clear()

    # Core dataframes and flags
    for k in [
        "df_final", "df_final_list", "merge_done",
        "df_transformed", "transform_complete",
        "numerical_config_dict", "categorical_config_dict",
    ]:
        st.session_state.pop(k, None)

    # Uploaded file controls
    st.session_state.pop("uploaded_files", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1

    # Progress step
    st.session_state["current_step"] = 0

    # Clear widget state created in Mapping & Profiling
    for k in list(st.session_state.keys()):
        if k.startswith((
            "select_cols_", "rename_editor_", "date_format_", "date_col_",
            "date_cols_", "npi_", "groupby_", "analysis_", "granularity_",
            "download_", "kpi_", "modify_granularity_", "normalize_"
        )):
            st.session_state.pop(k, None)

def normalize_columns_pl(df: pl.DataFrame, columns: List[str], method: str = "zscore") -> pl.DataFrame:
    """
    Polars-only normalization that mirrors the pandas version:
    - zscore: adds <col>_z using (x - mean)/std only if std != 0
    - iqr:    adds <col>_iqr using (x - Q1)/IQR only if IQR != 0
    Existing columns are preserved; new suffix columns are added when valid.
    """
    out = df.clone()

    # Ensure selected columns exist and are numeric-like
    cols = [c for c in columns if c in out.columns]

    if method == "zscore":
        # Compute means and stds as Python scalars
        means = out.select([pl.col(c).mean().alias(c) for c in cols]).to_dicts()[0] if cols else {}
        stds  = out.select([pl.col(c).std().alias(c)  for c in cols]).to_dicts()[0] if cols else {}

        exprs = []
        for c in cols:
            mu = means.get(c, None)
            sd = stds.get(c, None)
            if sd is not None and sd != 0 and mu is not None:
                exprs.append(((pl.col(c) - pl.lit(mu)) / pl.lit(sd)).alias(f"{c}_z"))
        if exprs:
            out = out.with_columns(exprs)

    elif method == "iqr":
        # Compute Q1, Q3, IQR as Python scalars
        q1s = out.select([pl.col(c).quantile(0.25).alias(c) for c in cols]).to_dicts()[0] if cols else {}
        q3s = out.select([pl.col(c).quantile(0.75).alias(c) for c in cols]).to_dicts()[0] if cols else {}

        exprs = []
        for c in cols:
            q1 = q1s.get(c, None)
            q3 = q3s.get(c, None)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                if iqr != 0:
                    exprs.append(((pl.col(c) - pl.lit(q1)) / pl.lit(iqr)).alias(f"{c}_iqr"))
        if exprs:
            out = out.with_columns(exprs)

    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    return out


def last_working_day(year: int, month: int, work_days: int) -> date:
    """
    Matches prior behavior:
    - Compute last calendar day of the month
    - If work_days == 5, roll back to Fri if last day falls on weekend
    Returns a date (not datetime).
    """
    if month == 12:
        month = 0
        year += 1
    last_day_dt = datetime(year, month + 1, 1) - timedelta(days=1)
    if work_days == 5:
        while last_day_dt.weekday() > 4:  # Fri is 4; Sat=5, Sun=6
            last_day_dt -= timedelta(days=1)
    return last_day_dt.date()


def last_week_apportion_polars(
    tactic_df: pl.DataFrame,
    date_col_name: str,
    kpi_col_list: List[str],
    work_days: int
) -> pl.DataFrame:
    """
    Polars-only port of last_week_apportion:
    - Compute month-first and map to last working date per month
    - day_diff = (last_working_date - week_start_date) + 1 in days
    - For each KPI, adjusted_<kpi> = ((work_days - day_diff)/work_days) * KPI if day_diff < work_days else 0
    - Subtract adjusted_<kpi> from original KPI in the source month
    - Add new rows for next month-begin with KPI = adjusted_<kpi>
    - Drop helper columns and return combined frame
    """
    if date_col_name not in tactic_df.columns:
        raise ValueError(f"{date_col_name} not found in DataFrame")

    # Ensure date column is Date type
    df = tactic_df.with_columns(
        pl.col(date_col_name).cast(pl.Date).alias(date_col_name)
    )

    # Month-first key for mapping
    df = df.with_columns(
        pl.date(
            year=pl.col(date_col_name).dt.year(),
            month=pl.col(date_col_name).dt.month(),
            day=1
        ).alias("_month_first")
    )

    # Build month -> last_working_date mapping table
    months = (
        df.select("_month_first")
          .unique()
          .get_column("_month_first")
          .to_list()
    )
    month_map = pl.DataFrame({
        "_month_first": months,
        "_last_working_date": [last_working_day(m.year, m.month, work_days) for m in months],
    })

    # Join mapping
    df = df.join(month_map, on="_month_first", how="left")

    # day_diff = (LWD - week_start_date) + 1
    df = df.with_columns(
        ((pl.col("_last_working_date") - pl.col(date_col_name)).dt.days() + 1).alias("_day_diff")
    )

    # Compute adjusted KPI columns
    adjusted_cols = [f"adjusted_{k}" for k in kpi_col_list]
    adjust_exprs = [
        pl.when(pl.col("_day_diff") < work_days)
          .then(((work_days - pl.col("_day_diff")) / work_days) * pl.col(k))
          .otherwise(pl.lit(0.0))
          .alias(f"adjusted_{k}")
        for k in kpi_col_list
    ]
    if adjust_exprs:
        df = df.with_columns(adjust_exprs)

    # Subtract adjusted from original KPIs in current month
    if kpi_col_list:
        df = df.with_columns([
            (pl.col(k) - pl.col(f"adjusted_{k}")).alias(k)
            for k in kpi_col_list
        ])

    # Rows that have any adjusted > 0
    if adjusted_cols:
        df = df.with_columns(
            pl.max_horizontal([pl.col(c) for c in adjusted_cols]).alias("_max_adjusted")
        )
        new_rows = df.filter(pl.col("_max_adjusted") > 0)
    else:
        new_rows = df.head(0)

    # Next month-begin date
    next_month_begin_expr = pl.when(pl.col(date_col_name).dt.month() == 12).then(
        pl.date(pl.col(date_col_name).dt.year() + 1, 1, 1)
    ).otherwise(
        pl.date(pl.col(date_col_name).dt.year(), pl.col(date_col_name).dt.month() + 1, 1)
    )

    # Build the "carry-forward" rows for next month where KPIs are set to adjusted values
    if new_rows.height > 0 and adjusted_cols:
        new_rows = new_rows.with_columns(
            next_month_begin_expr.alias(date_col_name)
        ).with_columns([
            pl.col(f"adjusted_{k}").alias(k) for k in kpi_col_list
        ])
    else:
        new_rows = new_rows  # remains empty

    # Drop helper/adjusted columns from both sides
    helper_cols = ["_month_first", "_last_working_date", "_day_diff", "_max_adjusted"] + adjusted_cols
    base_clean = df.drop([c for c in helper_cols if c in df.columns])
    new_rows_clean = new_rows.drop([c for c in helper_cols if c in new_rows.columns])

    # Align columns and concatenate
    # Ensure both frames share the same schema columns
    all_cols = list({*base_clean.columns, *new_rows_clean.columns})
    base_aligned = base_clean.select([pl.col(c) if c in base_clean.columns else pl.lit(None).alias(c) for c in all_cols])
    new_aligned  = new_rows_clean.select([pl.col(c) if c in new_rows_clean.columns else pl.lit(None).alias(c) for c in all_cols])

    out = pl.concat([base_aligned, new_aligned], how="vertical", rechunk=True)
    return out

# ------------------------------------------------------------------------------
import polars as pl
import streamlit as st
from typing import Dict, List
def detect_date_granularity(df, date_column: str):
    # Accept pandas or Polars; normalize to Polars
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if date_column not in df.columns:
        st.warning(f"Column '{date_column}' not found.")
        return

    # Coerce to Date with lenient parsing
    s = df.get_column(date_column)
    if s.dtype in (pl.Utf8, pl.Categorical):
        df2 = df.with_columns(pl.col(date_column).str.strptime(pl.Date, strict=False).alias(date_column))
    elif s.dtype == pl.Datetime:
        df2 = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))
    elif s.dtype != pl.Date:
        df2 = df.with_columns(pl.col(date_column).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(date_column))
    else:
        df2 = df

    # Drop nulls, deduplicate, sort
    df2 = (
        df2
        .filter(pl.col(date_column).is_not_null())
        .unique(subset=[date_column])
        .sort(date_column)
    )

    # Need at least 2 rows
    if df2.height < 2:
        st.warning("Irregular Date. Please check the column format.")
        return

    # Compute consecutive day differences
    df2 = df2.with_columns(
        (pl.col(date_column) - pl.col(date_column).shift(1)).dt.total_days().alias("_gap_days")
    ).filter(pl.col("_gap_days").is_not_null())

    if df2.height == 0:
        st.warning("Irregular Date. Please check the column format.")
        return

    # Most common gap
    vc = (
        df2
        .group_by("_gap_days")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(1)
    )

    if vc.height == 0:
        st.warning("Irregular Date. Please check the column format.")
        return

    most_common_days = int(vc.get_column("_gap_days").item())

    if most_common_days == 1:
        return "Daily"
    elif most_common_days == 7:
        return "Weekly"
    elif most_common_days in (28, 29, 30, 31):
        return "Monthly"
    elif most_common_days >= 365:
        return "Yearly"
    else:
        st.warning("Irregular Date. Please check the column format.")
        return

# ------------------------------------------------------------------------------

def modify_granularity(  # Polars-only implementation kept under same name for drop-in replacement
    df: pl.DataFrame,
    geo_column: str,
    date_column: str,
    granularity_level_df: str,
    granularity_level_user_input: str,
    work_days: int,
    numerical_config_dict: Dict[str, str],
    categorical_config_dict: Dict[str, str]
):
    # Helper to build Polars aggregation expressions
    def get_agg_exprs(numerical_dict: Dict[str, str], categorical_dict: Dict[str, str]) -> List[pl.Expr]:
        exprs: List[pl.Expr] = []
        # Numerical ops
        for col, op in numerical_dict.items():
            if op == "sum":
                exprs.append(pl.col(col).sum().alias(col))
            elif op == "average":
                exprs.append(pl.col(col).mean().alias(col))
            elif op == "min":
                exprs.append(pl.col(col).min().alias(col))
            elif op == "max":
                exprs.append(pl.col(col).max().alias(col))
            elif op == "product":
                exprs.append(pl.col(col).drop_nulls().product().alias(col))
        # Categorical ops
        for col, op in categorical_dict.items():
            if op == "count":
                exprs.append(pl.col(col).count().alias(col))
            elif op == "distinct count":
                exprs.append(pl.col(col).n_unique().alias(col))
        return exprs

    # No transformation needed
    if granularity_level_df == granularity_level_user_input:
        selected_cols = [geo_column, date_column] + list(numerical_config_dict.keys()) + list(categorical_config_dict.keys())
        # Ensure date is Date
        out = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column)).select([c for c in selected_cols if c in df.columns])
        return out, date_column

    # Ensure date is Date
    base = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))
    agg_exprs = get_agg_exprs(numerical_config_dict, categorical_config_dict)

    # Daily → Weekly
    if granularity_level_df == "Daily" and granularity_level_user_input == "Weekly":
        base = base.with_columns(
            (pl.col(date_column) - pl.duration(days=pl.col(date_column).dt.weekday())).alias("week_date")
        )
        grouped = base.group_by([geo_column, "week_date"]).agg(agg_exprs)
        return grouped, "week_date"

    # Daily → Monthly
    elif granularity_level_df == "Daily" and granularity_level_user_input == "Monthly":
        base = base.with_columns(
            pl.date(
                year=pl.col(date_column).dt.year(),
                month=pl.col(date_column).dt.month(),
                day=1
            ).alias("month_date")
        )
        grouped = base.group_by([geo_column, "month_date"]).agg(agg_exprs)
        return grouped, "month_date"

    # Weekly → Monthly (requires apportioning)
    elif granularity_level_df == "Weekly" and granularity_level_user_input == "Monthly":
        # Requires last_week_apportion_polars() to be defined earlier
        apportioned = last_week_apportion_polars(base, date_column, list(numerical_config_dict.keys()), work_days)
        apportioned = apportioned.with_columns(
            pl.date(
                year=pl.col(date_column).dt.year(),
                month=pl.col(date_column).dt.month(),
                day=1
            ).alias("month_date")
        )
        grouped = apportioned.group_by([geo_column, "month_date"]).agg(agg_exprs)
        return grouped, "month_date"

    else:
        raise ValueError("Unsupported granularity transformation")

# --- CONFIGURATION ---
def kpi_table(df: pl.DataFrame):
    numerical_config_dict = {}
    categorical_config_dict = {}
    try:
        st.success("File loaded successfully!")
        st.write("### Preview of Data")
        st.dataframe(df.head().to_pandas())  # Polars head -> pandas for Streamlit preview

        # Polars dtypes for filtering
        numeric_dtypes = [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        ]
        category_dtypes = [pl.Utf8, pl.Categorical]

        # Numerical Column Selection
        num_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_dtypes]
        st.subheader("🔢 Numerical Columns")
        selected_num_cols = st.multiselect("Select numerical columns:", num_cols)

        if selected_num_cols:
            # Using pandas DataFrame purely for editor convenience
            num_ops_df = pd.DataFrame({
                "Column": selected_num_cols,
                "Operation": [""] * len(selected_num_cols)
            })
            operation_options = ["average", "sum", "product", "min", "max"]
            edited_num_df = st.data_editor(
                num_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=operation_options)
                },
                num_rows="fixed",
                key="numerical_editor"
            )

        # Categorical Column Selection
        cat_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in category_dtypes]
        st.subheader("🔠 Categorical Columns")
        selected_cat_cols = st.multiselect("Select categorical columns:", cat_cols)

        if selected_cat_cols:
            cat_ops_df = pd.DataFrame({
                "Column": selected_cat_cols,
                "Operation": [""] * len(selected_cat_cols)
            })
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
            if st.button("💾 Save All  Configurations"):
                if selected_num_cols:
                    numerical_config_dict = edited_num_df.set_index("Column")["Operation"].to_dict()
                if selected_cat_cols:
                    categorical_config_dict = edited_cat_df.set_index("Column")["Operation"].to_dict()
                st.success("✅ Configuration saved!")
                return numerical_config_dict, categorical_config_dict

    except Exception as e:
        st.error(f"Error processing KPI Table: {e}")

# --- PAGE CONFIG ---
st.set_page_config(page_title="ProcTimize", layout="wide")
st.title("Data Ingestion")
st.write("This page is for processing of one marketing channel at a time and saving")

st.markdown("""
<style>
    div[data-baseweb="tag"] > div {
    background-color: #001E96 !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 0.3rem 0.8rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: none !important;
    }

    div[data-baseweb="tag"] > div > span {
        color: white !important;
    }
            
    /* ✅ Sidebar with deep blue gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001E96 0%, #001E96 100%) !important;
        color: white;
    }

    /* ✅ Page background */
    html, body, [class*="stApp"] {
        background-color: #F6F6F6;
    }

    /* ✅ Force white text in sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ✅ Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #001E96;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #001E96 !important;
        filter: brightness(1.1);
    }

    /* ✅ Custom color for selected multiselect pills */
    div[data-baseweb="tag"] {
        background-color: #001E96 !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 0.3rem 0.8rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ✅ White "x" icon in pills */
    div[data-baseweb="tag"] svg {
        fill: white !important;
    }

    /* ✅ Selected text inside the pill */
    div[data-baseweb="tag"] div {
        color: white !important;
    }

    /* ✅ Inputs, selects, multiselects focus color */
    input:focus, textarea:focus, .stTextInput > div > div > input:focus {
        border-color: #001E96 !important;
        box-shadow: 0 0 0 2px #001E96 !important;
    }

    /* ✅ Select box border */
    div[data-baseweb="select"] > div {
        border-color: #001E96 !important;
        box-shadow: 0 0 0 1.5px #001E96 !important;
        border-radius: 6px !important;
    }

    /* ✅ Search input text color */
    div[data-baseweb="select"] input {
        color: black !important;
    }

    /* ✅ Clean input fields (remove red glow) */
    .stTextInput > div {
        box-shadow: none !important;
    }

    /* ✅ All generic buttons */
    .stButton > button {
        background-color: #001E96 !important;
        color: white !important;
        border: none;
        border-radius: 25px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #001E96 !important;
        filter: brightness(1.1);
        transform: translateY(-1px);
    }

    /* ✅ Kill red outlines everywhere */
    *:focus {
        outline: none !important;
        border-color: #001E96 !important;
        box-shadow: 0 0 0 2px #001E96 !important;
    }
            
    
</style>
""", unsafe_allow_html=True)


if st.button("Reset workflow", key="btn_reset_workflow"):
        reset_full_workflow()
        st.success("Workflow has been reset. Re-upload files to continue.")
        st.stop()


# --- FILE UPLOAD ---
with st.expander("Upload One or More CSV Files"):
    uploaded_files = st.file_uploader(
        "Select CSV file(s) to upload:",
        type="csv",
        accept_multiple_files=True
    )


# --- ☁️ BLOB STORAGE PLACEHOLDER ---
with st.expander("Connect to Azure Blob Storage (Coming Soon)"):
    st.markdown("This section will allow you to securely connect to an Azure Blob Storage account and load files directly.")

    st.subheader("Azure Credentials")
    blob_account_name = st.text_input("Azure Storage Account Name", disabled=True, placeholder="e.g., mystorageaccount")
    blob_container_name = st.text_input("Container Name", disabled=True, placeholder="e.g., marketing-data")
    blob_sas_token = st.text_input("SAS Token", type="password", disabled=True, placeholder="Your secure token")

    st.subheader("File Selection")
    st.selectbox("Browse files in container", options=["-- Coming soon --"], disabled=True)
    st.button("Refresh File List", disabled=True)

    st.markdown("---")
    st.info("You will be able to load files directly from your Azure Blob Storage container using the credentials above.")

    # 🔧 TO IMPLEMENT LATER:
    # from azure.storage.blob import ContainerClient
    # container_url = f"https://{blob_account_name}.blob.core.windows.net/{blob_container_name}?{blob_sas_token}"
    # container_client = ContainerClient.from_container_url(container_url)
    # blobs = [blob.name for blob in container_client.list_blobs()]
    # blob_client = container_client.get_blob_client(blob_name)
    # stream = blob_client.download_blob().readall()
    # df_final = pl.read_csv(io.BytesIO(stream), infer_schema_length=10000)


df_final: pl.DataFrame | None = None

# Persistent state
if "granularity" not in st.session_state:
    st.session_state["granularity"] = None
if "geo_col_selected" not in st.session_state:
    st.session_state["geo_col_selected"] = None
if "user_input_time_granularity" not in st.session_state:
    st.session_state["user_input_time_granularity"] = None
if "numerical_config_dict" not in st.session_state:
    st.session_state["numerical_config_dict"] = {}
if "categorical_config_dict" not in st.session_state:
    st.session_state["categorical_config_dict"] = {}

if uploaded_files:
    if len(uploaded_files) == 1:
        with st.expander("Standardize Columns for Single File"):
            file = uploaded_files[0]
            df = pl.read_csv(file, infer_schema_length=10000)
            st.markdown(f"**File: {file.name}**")

            # Step 1: Column Selection
            selected_cols = st.multiselect(
                f"Select columns from `{file.name}`:",
                df.columns,
                default=df.columns,
                key=f"select_cols_0"
            )
            if selected_cols:
                df = df.select(selected_cols)

            # Step 2a: Column Renaming (editor via pandas)
            rename_df = pd.DataFrame({
                "Current Column": df.columns,
                "New Column Name": df.columns
            })
            column_config = {
                "Current Column": st.column_config.Column(disabled=True),
                "New Column Name": st.column_config.TextColumn()
            }
            edited_rename_df = st.data_editor(
                rename_df,
                column_config=column_config,
                num_rows="dynamic",
                use_container_width=True,
                key="rename_editor"
            )
            rename_dict = dict(zip(
                edited_rename_df["Current Column"],
                edited_rename_df["New Column Name"]
            ))
            # Polars rename
            if rename_dict:
                df = df.rename(rename_dict)

          
            suggested_dates = detect_date_columns_by_sampling(df, sample_size=200, threshold=0.8)
            date_cols = st.multiselect(
                "Select Date Columns (if any):",
                df.columns,
                default=suggested_dates,
                key="date_cols_single"
            )
            if suggested_dates:
                st.info(f"Suggested date columns: {', '.join(suggested_dates)}")
            for date_col in date_cols:
                st.markdown(f"📅 **Standardizing `{date_col}`**")
                date_format = st.selectbox(
                    f"Select current date format for `{date_col}` being used:",
                    ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "Custom"],
                    index=None,
                    key=f"date_format_single_{date_col}"
                )

                if date_format == "Custom":
                    date_format = st.text_input(
                        "Enter custom date format (e.g., %d-%b-%Y):",
                        key=f"custom_date_format_single_{date_col}"
                    )

                if date_format:
                    try:
                        df = df.with_columns(
                            pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Date, format=date_format, strict=False).alias(date_col)
                        )
                        st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` (Date dtype)!")
                    except Exception as e:
                        st.error(f"❌ Failed to parse date column `{date_col}`: {e}")

            # Final clean-up before saving: fill nulls in string-like columns
            str_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
            if str_cols:
                df = df.with_columns([pl.col(c).fill_null("") for c in str_cols])

            # Canonical Polars DF
            df_final = df
            st.session_state["df_final"] = df_final

    else:
        with st.expander("🧱 Standardize Columns for Multiple Files"):
            column_mappings = []
            dfs = []
            renamed_columns_list = []

            for i, file in enumerate(uploaded_files):
                st.markdown(f"**File {i+1}: {file.name}**")
                dfi = pl.read_csv(file, infer_schema_length=10000)
                dfs.append(dfi)

                # ✅ Step 1: Column Selection
                selected_cols = st.multiselect(
                    f"Select columns from `{file.name}`:",
                    dfi.columns,
                    default=dfi.columns,
                    key=f"select_cols_{i}"
                )
                if selected_cols:
                    dfi = dfi.select(selected_cols)

                # ✅ Step 2: Interactive Data Editor for Renaming (editor via pandas)
                rename_df = pd.DataFrame({
                    "Current Column": dfi.columns,
                    "New Column Name": dfi.columns
                })
                column_config = {
                    "Current Column": st.column_config.Column(disabled=True),
                    "New Column Name": st.column_config.TextColumn()
                }
                edited_rename_df = st.data_editor(
                    rename_df,
                    column_config=column_config,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"rename_editor_{i}"
                )

                # ✅ Step 3: Extract Rename Mapping
                rename_dict = dict(zip(
                    edited_rename_df["Current Column"],
                    edited_rename_df["New Column Name"]
                ))

                # ✅ Collect Final Mappings
                column_mappings.append(rename_dict)
                renamed_columns_list.append(set(edited_rename_df["New Column Name"]))

                # ✅ Apply renaming
                if rename_dict:
                    dfi = dfi.rename(rename_dict)

                # 📅 Step: Ask for Date Format and Standardize Date Columns
                suggested_dates = detect_date_columns_by_sampling(dfi, sample_size=200, threshold=0.8)
                if suggested_dates:
                    st.info(f"Suggested date columns in `{file.name}`: {', '.join(suggested_dates)}")

                date_cols = st.multiselect(
                    f"Select Date Columns in `{file.name}` (if any):",
                    dfi.columns,
                    default=suggested_dates,
                    key=f"date_cols_{i}"
                )

                for date_col in date_cols:
                    st.markdown(f"📅 **Standardizing `{date_col}` in file `{file.name}`**")
                    date_format = st.selectbox(
                        f"Select the current date format used for `{date_col}`:",
                        ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "Custom"],
                        index=None,
                        key=f"date_format_{i}_{date_col}"
                    )

                    if date_format == "Custom":
                        date_format = st.text_input(
                            "Enter custom date format (e.g., %d-%b-%Y):",
                            key=f"custom_date_format_{i}_{date_col}"
                        )

                    if date_format:
                        try:
                            dfi = dfi.with_columns(
                                pl.col(date_col).cast(pl.Utf8).str.strptime(pl.Date, format=date_format, strict=False).alias(date_col)
                            )
                            st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` (Date dtype)!")
                        except Exception as e:
                            st.error(f"❌ Failed to parse date column `{date_col}`: {e}")

                # Replace the dfs[i] with standardized dfi
                dfs[i] = dfi

            # Optional: After all standardization, align and concatenate
            if dfs:
                df_final = pl.concat(dfs, how="vertical_relaxed", rechunk=True)
                # Fill string nulls
                str_cols = [c for c, dt in zip(df_final.columns, df_final.dtypes) if dt == pl.Utf8]
                if str_cols:
                    df_final = df_final.with_columns([pl.col(c).fill_null("") for c in str_cols])
                st.session_state["df_final"] = df_final
        with st.expander("Merge Files:"):
            st.subheader("🔗Select Merge Strategy")
            common_columns = set.intersection(*renamed_columns_list) if renamed_columns_list else set()
            merge_strategy = st.radio("Merge type:", ["Vertical Stack", "Horizontal Join"])
            if merge_strategy == "Horizontal Join":
                join_keys = st.multiselect("Select join key(s):", sorted(list(common_columns)))
                join_type = st.selectbox("Join type:", ["inner", "left", "right", "outer"])
            else:
                join_keys = None
                join_type = None

            if st.button("Merge Files", key="btn_merge_files"):
                # Perform merge (same logic you already had)
                if merge_strategy in ["vertical", "Vertical Stack"]:
                    if not dfs:
                        st.warning("No data frames to merge.")
                        st.stop()
                    df_final = pl.concat(dfs, how="vertical_relaxed", rechunk=True)
                elif merge_strategy in ["horizontal", "Horizontal Join"]:
                    if not join_keys:
                        st.warning("Join key must be provided for horizontal joins.")
                        st.stop()
                    if not dfs:
                        st.warning("No data frames to join.")
                        st.stop()
                    df_final = dfs[0]
                    how = {"inner":"inner","left":"left","right":"right","outer":"outer"}[join_type]
                    for dfi in dfs[1:]:
                        df_final = df_final.join(dfi, on=join_keys, how=how)
                else:
                    st.error("Invalid merge strategy. Choose 'vertical' or 'horizontal'.")
                    st.stop()

                # Identifier cleanup, date parsing, datetime->date casts (unchanged)
                if join_keys:
                    for col_name in join_keys:
                        if col_name in df_final.columns:
                            df_final = df_final.filter(pl.col(col_name).is_not_null())
                    df_final = df_final.unique(maintain_order=True)
                date_like_cols = [c for c, dt in zip(df_final.columns, df_final.dtypes) if dt == pl.Utf8 and "date" in c.lower()]
                for c in date_like_cols:
                    try:
                        df_final = df_final.with_columns(pl.col(c).str.strptime(pl.Date, strict=False).alias(c))
                    except Exception:
                        pass
                df_final = df_final.with_columns([
                    pl.col(col).cast(pl.Date).alias(col)
                    for col, dtype in zip(df_final.columns, df_final.dtypes)
                    if dtype == pl.Datetime
                ])

                st.session_state["df_final"] = df_final
                st.session_state["merge_done"] = True

                # Preview
                st.write("### Merged Preview")
                st.dataframe(df_final.head(100).to_pandas())

                # Quick stats (rows, cols, memory), plus column-wise non-null and missing%
                rows = df_final.height
                cols = len(df_final.columns)
                try:
                    # Polars 0.20+ provides estimated_size on lazy/collect; fallback to pandas if needed
                    mem_mb = df_final.to_pandas().memory_usage(deep=True).sum()/(1024**2)
                except Exception:
                    mem_mb = df_final.to_pandas().memory_usage(deep=True).sum()/(1024**2)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{rows:,}")
                c2.metric("Columns", f"{cols:,}")
                c3.metric("Memory", f"{mem_mb:.2f} MB")

                # Column profile
                pdf = df_final.to_pandas()
                profile = pd.DataFrame({
                    "column": pdf.columns,
                    "dtype": [str(t) for t in pdf.dtypes],
                    "non_null": [pdf[c].notna().sum() for c in pdf.columns],
                    "missing": [pdf[c].isna().sum() for c in pdf.columns],
                })
                profile["missing_pct"] = (profile["missing"] / len(pdf) * 100).round(2)
                st.write("### Column Statistics")
                st.dataframe(profile, use_container_width=True)
                st.download_button(
                    "📥 Download Column Statistics",
                    data=profile.to_csv(index=False).encode("utf-8"),
                    file_name="merged_column_stats.csv",
                    mime="text/csv"
                )

                # Download merged CSV
                merged_csv = pdf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download merged CSV",
                    data=merged_csv,
                    file_name="merged_final.csv",
                    mime="text/csv",
                    key="download_merged_csv"
                )

    
        # ---------------------- FILTERING SECTION ----------------------
        with st.expander("🔍Filter Data"):
            if "df_final" in st.session_state:
                df_final = st.session_state["df_final"]
            else:
                df_final = None
            df_final = st.session_state.get("df_final")
            if df_final is None or df_final.height == 0:
                st.error("❌ No data available for filtering. Please check the input files.")
                st.stop()

            st.write("### Sample of the Dataset")

            # Create display DataFrame with dates formatted as strings
            df_display = df_final.clone()
            for col, dtype in zip(df_display.columns, df_display.dtypes):
                if dtype in (pl.Date, pl.Datetime):
                    df_display = df_display.with_columns(
                        pl.col(col).dt.strftime("%Y-%m-%d").alias(col)
                    )

            st.dataframe(df_display.head(100).to_pandas())

            # Working frame for all filters
            df_filtered = df_final.clone()

            # 2) NPI validity first: remove rows where NPI invalid OR Date missing/invalid (optional Luhn)
            with st.expander("Row removal: NPI or Date missing"):
                # Candidate columns
                npi_candidates = [c for c in df_filtered.columns if "npi" in c.lower()] + \
                                [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt == pl.Utf8]
                npi_candidates = sorted(list(dict.fromkeys(npi_candidates)))

                # Auto-pick defaults
                auto_npi = npi_candidates[0] if npi_candidates else "-- none --"
                npi_col = st.selectbox("NPI column:", options=["-- none --"] + npi_candidates,
                                    index=(0 if auto_npi == "-- none --" else (["-- none --"] + npi_candidates).index(auto_npi)),
                                    key="npi_rm_select")

                # Auto-pick date col (prefer Date/Datetime, else name contains 'date')
                date_like_cols_auto = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt in (pl.Date, pl.Datetime)]
                if date_like_cols_auto:
                    preferred_auto = [c for c in date_like_cols_auto if any(k in c.lower() for k in ["date","dt","day"])]
                    auto_date_rm = preferred_auto[0] if preferred_auto else date_like_cols_auto[0]
                else:
                    name_candidates_auto = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt == pl.Utf8 and "date" in c.lower()]
                    auto_date_rm = name_candidates_auto[0] if name_candidates_auto else df_filtered.columns[0]

                date_col_for_rm = st.selectbox("Date column:", options=df_filtered.columns,
                                            index=df_filtered.columns.index(auto_date_rm),
                                            key="date_rm_select")

                use_luhn = st.checkbox("Treat Luhn failures as invalid (optional)", value=False, key="npi_luhn_toggle")

                # RUN AUTOMATICALLY (no button): ensure Date dtype and drop rows failing NPI or Date checks
                tmp = df_filtered

                if tmp.schema.get(date_col_for_rm) in (pl.Utf8, pl.Categorical):
                    try:
                        tmp = tmp.with_columns(pl.col(date_col_for_rm).str.strptime(pl.Date, strict=False).alias(date_col_for_rm))
                    except Exception:
                        pass

                # Build validity: pattern first (10 digits starting 1/2)
                if npi_col and npi_col != "-- none --":
                    s = pl.col(npi_col).cast(pl.Utf8).str.strip_chars()
                    pattern_ok = s.str.contains(r"^[12]\d{9}$", literal=False)
                    if use_luhn:
                        # Unique-first cache for speed
                        unique_npIs = (
                            tmp.select(s.alias("_npi"))
                            .filter(pattern_ok)
                            .select(pl.col("_npi").drop_nulls().unique())
                            .to_series().to_list()
                        )
                        cache = {n: luhn_valid_npi(n) for n in unique_npIs}
                        luhn_ok = s.map_elements(lambda x: cache.get(x, False), return_dtype=pl.Boolean)
                        npi_valid = pattern_ok & luhn_ok
                    else:
                        npi_valid = pattern_ok
                else:
                    # If no NPI column, treat as all invalid to avoid false keeps; user can pick a column to enable
                    npi_valid = pl.lit(False)

                date_ok = pl.col(date_col_for_rm).is_not_null()

                kept = tmp.filter(npi_valid & date_ok)
                dropped = df_filtered.height - kept.height

                # Persist to working frame/state so downstream filters and preview reflect the change
                df_filtered = kept
                st.session_state["df_filtered"] = df_filtered
                st.session_state["filter_complete"] = True

                st.success(f"Removed {dropped:,} rows where NPI or Date was missing/invalid.")

            # Date filtering
            # Auto-pick a default date column if possible
            try:
                auto_date = None
                date_like_cols_auto = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt in (pl.Date, pl.Datetime)]
                if date_like_cols_auto:
                    preferred_auto = [c for c in date_like_cols_auto if any(k in c.lower() for k in ["date", "dt", "day"])]
                    auto_date = preferred_auto[0] if preferred_auto else date_like_cols_auto[0]
                else:
                    name_candidates_auto = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt == pl.Utf8 and "date" in c.lower()]
                    if name_candidates_auto:
                        auto_date = name_candidates_auto[0]
                default_index = df_filtered.columns.index(auto_date) if (auto_date in df_filtered.columns) else 0
            except Exception:
                default_index = 0

            date_column = st.selectbox("Select the column representing Date in merged dataset", df_filtered.columns, index=default_index)
            st.session_state["date_col"] = date_column

            if date_column:
                dtype = df_filtered.schema.get(date_column)
                if dtype in (pl.Utf8,):
                    try:
                        df_filtered = df_filtered.with_columns(
                            pl.col(date_column).str.strptime(pl.Date, strict=False).alias(date_column)
                        )
                    except Exception:
                        st.warning("⚠️ Valid date column not selected or parsing failed.")
                        st.stop()
                elif dtype == pl.Datetime:
                    df_filtered = df_filtered.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))
                elif dtype != pl.Date:
                    st.warning("⚠️ Selected column is not a valid date column.")
                    st.stop()

                min_date = df_filtered.select(pl.col(date_column).min()).item()
                max_date = df_filtered.select(pl.col(date_column).max()).item()

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

                if start_date > end_date:
                    st.error("Start date must be before or equal to end date.")
                    st.stop()
                else:
                    df_filtered = df_filtered.filter((pl.col(date_column) >= pl.lit(start_date)) & (pl.col(date_column) <= pl.lit(end_date)))

            # Categorical filtering
            categorical_cols = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt == pl.Utf8]

            if categorical_cols:
                st.write("### Categorical Column(s) to filter on")
                selected_cat_cols = st.multiselect("Select categorical columns to filter:", categorical_cols)

                for col in selected_cat_cols:
                    unique_vals = df_filtered.select(pl.col(col)).unique().to_series().to_list()
                    display_vals = []
                    for val in unique_vals:
                        if val is None or str(val).strip() == "":
                            display_vals.append("<BLANK>")
                        else:
                            display_vals.append(str(val))
                    display_vals = sorted(display_vals)

                    selected_vals = st.multiselect(
                        f"Select values to retain in '{col}'",
                        options=display_vals,
                        default=display_vals,
                        key=f"filter_{col}"
                    )

                    filter_conditions = []

                    if "<BLANK>" in selected_vals:
                        filter_conditions.append(pl.col(col).is_null() | (pl.col(col).cast(pl.Utf8).str.strip_chars().eq("")))

                    selected_non_blank_vals = [val for val in selected_vals if val != "<BLANK>"]
                    if selected_non_blank_vals:
                        filter_conditions.append(pl.col(col).cast(pl.Utf8).is_in(selected_non_blank_vals))

                    if filter_conditions:
                        combined_condition = filter_conditions[0]
                        for cond in filter_conditions[1:]:
                            combined_condition = combined_condition | cond
                        df_filtered = df_filtered.filter(combined_condition)
                        st.session_state["df_filtered"] = df_filtered
                        st.session_state["filter_complete"] = True

            # --- NUMERICAL FILTERING ---
            numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
            numeric_cols = [c for c, dt in zip(df_filtered.columns, df_filtered.dtypes) if dt in numeric_dtypes]
            numerical_conditions = []

            if numeric_cols:
                st.write("### 🔢 Numerical Column(s) to filter on")
                selected_num_cols = st.multiselect("Select numerical columns to filter:", numeric_cols)

                for col in selected_num_cols:
                    st.markdown(f"**Filter for `{col}`**")
                    operator = st.selectbox(
                        f"Choose condition for `{col}`:",
                        options=[
                            "Equals", "Does not equal",
                            "Greater than", "Greater than or equal to",
                            "Less than", "Less than or equal to"
                        ],
                        key=f"num_op_{col}"
                    )

                    value = st.number_input(f"Enter value for `{col}`:", key=f"num_val_{col}")

                    # Round the column for equality comparison if float
                    if df_filtered.schema[col] in (pl.Float32, pl.Float64):
                        df_filtered = df_filtered.with_columns(pl.col(col).round(3))

                    # Build condition
                    if operator == "Equals":
                        numerical_conditions.append(pl.col(col) == value)
                    elif operator == "Does not equal":
                        numerical_conditions.append(pl.col(col) != value)
                    elif operator == "Greater than":
                        numerical_conditions.append(pl.col(col) > value)
                    elif operator == "Greater than or equal to":
                        numerical_conditions.append(pl.col(col) >= value)
                    elif operator == "Less than":
                        numerical_conditions.append(pl.col(col) < value)
                    elif operator == "Less than or equal to":
                        numerical_conditions.append(pl.col(col) <= value)
                # Apply numerical filters together
                if numerical_conditions:
                    combined_condition = numerical_conditions[0]
                    for cond in numerical_conditions[1:]:
                        combined_condition = combined_condition & cond
                    df_filtered = df_filtered.filter(combined_condition)

                    st.info(f"🔎 Rows after numerical filtering: {df_filtered.height}")

                    st.session_state["df_filtered"] = df_filtered
                    st.session_state["filter_complete"] = True

            if st.session_state.get("filter_complete") and "df_filtered" in st.session_state:
                st.write("Final Filtered Dataset")

                # Display the formatted DataFrame
                df_filtered = st.session_state["df_filtered"]
                df_filtered_display = df_filtered.clone()

                for col, dtype in zip(df_filtered_display.columns, df_filtered_display.dtypes):
                    if dtype in (pl.Date, pl.Datetime):
                        df_filtered_display = df_filtered_display.with_columns(
                            pl.col(col).dt.strftime("%Y-%m-%d").alias(col)
                        )

                st.dataframe(df_filtered_display.head(100).to_pandas())

                csv_bytes = df_filtered.write_csv()
                st.download_button("📥 Download CSV", data=csv_bytes, file_name="final_filtered_data.csv", mime="text/csv")

                # ---------------------- GRANULARITY SECTION ----------------------
        with st.expander("Modify Time Granularity"):
            # Retrieve state
            df_filtered = st.session_state.get("df_filtered")
            date_column = st.session_state.get("date_col") or st.session_state.get("date_column")

            if df_filtered is not None:
                column_names = df_filtered.columns
                geo_col = st.selectbox(
                    "Select the Grouping Column",
                    options=column_names,
                    key="geo_col_select"
                )
                st.session_state["geo_col_selected"] = geo_col

                # Initialize granularity in session state (use None, not "")
                if "granularity" not in st.session_state:
                    st.session_state["granularity"] = None

                # Detect granularity (Polars-only) with stable key
                if st.button("Detect Granularity", key="btn_detect_granularity"):
                    date_column = st.session_state.get("date_col") or st.session_state.get("date_column") or date_column
                    if not date_column or date_column not in df_filtered.columns:
                        st.warning("⚠️ Valid date column not selected or missing.")
                    else:
                        df_for_detect = df_filtered

                        # Normalize to Date dtype
                        orig_dtype = df_for_detect.schema.get(date_column)
                        if orig_dtype in (pl.Utf8, pl.Categorical):
                            df_for_detect = df_for_detect.with_columns(
                                pl.col(date_column).str.strptime(pl.Date, strict=False).alias(date_column)
                            )
                        elif orig_dtype == pl.Datetime:
                            df_for_detect = df_for_detect.with_columns(
                                pl.col(date_column).cast(pl.Date).alias(date_column)
                            )
                        elif orig_dtype != pl.Date:
                            df_for_detect = df_for_detect.with_columns(
                                pl.col(date_column).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(date_column)
                            )

                        # Drop null dates
                        df_for_detect = df_for_detect.filter(pl.col(date_column).is_not_null())

                        if df_for_detect.height == 0:
                            st.warning("⚠️ No valid dates after parsing.")
                        else:
                            granularity = detect_date_granularity(df_for_detect, date_column)
                            if granularity:
                                st.session_state["granularity"] = granularity
                                st.success(f"📈 Detected Granularity: **{granularity}**")
                            else:
                                st.warning("⚠️ No result found")

                # Read stored granularity once
                granularity = st.session_state.get("granularity")

                # KPI configuration
                st.subheader("Creating KPI Table")
                config_result = kpi_table(df_filtered)
                if config_result is not None:
                    numerical_config_dict, categorical_config_dict = config_result
                    st.session_state["numerical_config_dict"] = numerical_config_dict
                    st.session_state["categorical_config_dict"] = categorical_config_dict
                else:
                    numerical_config_dict = st.session_state.get("numerical_config_dict", {})
                    categorical_config_dict = st.session_state.get("categorical_config_dict", {})

                # Integrated Analytics Database
                st.subheader("Creating Unified Database for Channel")

                if granularity:
                    if granularity == 'Daily':
                        granularity_options = ['Weekly', 'Monthly']
                    elif granularity == 'Weekly':
                        granularity_options = ['Weekly', 'Monthly']
                    elif granularity == 'Monthly':
                        granularity_options = ['Monthly']
                    else:
                        granularity_options = []

                    time_granularity_user_input = st.selectbox(
                        "Choose the time granularity level",
                        granularity_options,
                        key="granularity_level_select",
                        index=(granularity_options.index(st.session_state["user_input_time_granularity"])
                            if st.session_state.get("user_input_time_granularity") in granularity_options else 0)
                    )
                    st.session_state["user_input_time_granularity"] = time_granularity_user_input

                    # Modifying the granularity
                    if st.button("Modify Granularity", key="btn_modify_granularity"):
                        # Read everything from session for stability
                        df_filtered = st.session_state.get("df_filtered")
                        granularity = st.session_state.get("granularity")
                        geo_sel = st.session_state.get("geo_col_selected") or geo_col
                        date_sel = st.session_state.get("date_col") or date_column
                        usr_gran = st.session_state.get("user_input_time_granularity") or time_granularity_user_input
                        num_cfg = st.session_state.get("numerical_config_dict", {})
                        cat_cfg = st.session_state.get("categorical_config_dict", {})

                        if not (df_filtered is not None and granularity and geo_sel and date_sel and usr_gran):
                            st.warning("Please enter the required columns")
                        else:
                            # Polars-only modify granularity
                            df_transformed, new_date_col = modify_granularity(
                                df=df_filtered,
                                geo_column=geo_sel,
                                date_column=date_sel,
                                granularity_level_df=granularity,
                                granularity_level_user_input=usr_gran,
                                work_days=7,
                                numerical_config_dict=num_cfg,
                                categorical_config_dict=cat_cfg
                            )

                            # Reapply rename dict if available
                            if st.session_state.get("rename_dict"):
                                df_transformed = df_transformed.rename(st.session_state["rename_dict"])

                            st.session_state["df_transformed"] = df_transformed
                            st.session_state["transform_complete"] = True


                # --- NORMALIZATION SECTION ---
                if st.session_state.get("transform_complete") and "df_transformed" in st.session_state:
                    df_transformed = st.session_state["df_transformed"]

                    st.subheader("🧮 Normalize Numerical Columns")

                    # Determine numeric columns from Polars dtypes
                    numeric_dtypes = {
                        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64
                    }
                    numeric_cols = [c for c, dt in zip(df_transformed.columns, df_transformed.dtypes) if dt in numeric_dtypes]

                    selected_norm_cols = st.multiselect("Select columns to normalize:", numeric_cols, key="norm_cols")
                    norm_method = st.radio("Normalization method:", ["Z-Score", "Interquartile Range (IQR)"], key="norm_method")

                    if st.button("Normalize Selected Columns", key="normalize_button"):
                        method_key = "zscore" if norm_method == "Z-Score" else "iqr"
                        df_normalized = normalize_columns_pl(df_transformed, selected_norm_cols, method=method_key)
                        st.session_state["df_transformed"] = df_normalized
                        st.success("✅ Normalization applied.")
                        st.dataframe(df_normalized.head(100).to_pandas())

                    # --- Display renaming and download UI after transformation ---
                    if st.session_state.get("transform_complete") and "df_transformed" in st.session_state:
                        df_transformed = st.session_state["df_transformed"]
                        st.success("Modified Granular data loaded")
                        st.dataframe(df_transformed.head(100).to_pandas())

                        # Rename editor (pandas for editor UI only)
                        rename_df = pd.DataFrame({
                            "Current Column": df_transformed.columns,
                            "New Column Name": df_transformed.columns
                        })
                        edited_rename_df = st.data_editor(
                            rename_df,
                            num_rows="dynamic",
                            use_container_width=True,
                            key="rename_editor_transformed"
                        )

                        # Save renaming and apply
                        rename_dict = dict(zip(
                            edited_rename_df["Current Column"],
                            edited_rename_df["New Column Name"]
                        ))
                        st.session_state["rename_dict"] = rename_dict
                        if rename_dict:
                            df_transformed = df_transformed.rename(rename_dict)
                            st.session_state["df_transformed"] = df_transformed

                        # File name input
                        file_name_input = st.text_input(
                            "Enter file name for download (with .csv extension):",
                            value="final_filtered_transformed_data.csv"
                        )

                        # Clean file name
                        file_name = file_name_input.strip()
                        if not file_name.lower().endswith(".csv"):
                            st.warning("Filename should end with '.csv'. Adding extension automatically.")
                            file_name += ".csv"

                        # Download button (Polars)
                        csv_bytes = df_transformed.write_csv()
                        st.download_button(
                            "📥 Download CSV",
                            data=csv_bytes,
                            file_name=file_name,
                            mime="text/csv"
                        )
            else:
                st.warning("⚠️ Please load granularity before selecting granularity level.")
