import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import re
import io
st.set_page_config(page_title= "ProcTimize", layout = "wide")
st.title("DATA INGESTION")

from datetime import datetime, timedelta 

RE_NPI = re.compile(r"^[12]\d{9}$")

def luhn_valid_npi(npi: str) -> bool:
    if not RE_NPI.fullmatch(npi):
        return False
    base, chk = npi[:9], ord(npi[9]) - 48
    s = 24  # CMS '80840' tweak
    # Double every second from right on the base-9
    for i in range(9):
        d = ord(base[8 - i]) - 48
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return ((10 - (s % 10)) % 10) == chk

def detect_date_columns_by_sampling(df, sample_size=100, threshold=0.8, formats=None):
    if formats is None:
        # List of date formats to try detecting
        formats = [ "%d/%m/%Y", "%Y/%m/%d", "%Y/%d/%m","%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y-%d-%m", "%Y-%m-%d"]

    date_cols = []
    for col in df.columns:
        try:
            non_null_vals = df[col].dropna()
            if non_null_vals.empty:
                continue

            sample_vals = non_null_vals.sample(min(sample_size, len(non_null_vals)), random_state=42)

            # Try each format and keep the best parsing success
            best_success = 0
            best_parsed = None
            for fmt in formats:
                parsed_dates = pd.to_datetime(sample_vals, format=fmt, errors='coerce')
                success_ratio = parsed_dates.notna().mean()
                if success_ratio > best_success:
                    best_success = success_ratio
                    best_parsed = parsed_dates

            if best_success >= threshold:
                min_date, max_date = best_parsed.min(), best_parsed.max()
                if pd.notna(min_date) and pd.notna(max_date) and 1900 <= min_date.year <= 2100 and 1900 <= max_date.year <= 2100:
                    date_cols.append(col)
        except Exception:
            pass
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
    st.session_state.pop("uploaded_files", None)
    st.session_state["uploader_key"]=st.session_state.get("uploader_key",0)+1
    # Progress step
    st.session_state["current_step"] = 0

    # Uploaded files selection (optional: keep or clear depending on UX)
    # st.session_state.pop("uploaded_files", None)  # only if you store it in session

    # Clear widget state created in Mapping & Profiling
    for k in list(st.session_state.keys()):
        if k.startswith((
            "select_cols_", "rename_editor_", "date_format_", "date_col_",
            "date_cols_", "npi_", "groupby_", "analysis_", "granularity_",
            "download_", "kpi_", "modify_granularity_", "normalize_"
        )):
            st.session_state.pop(k, None)


# Last week apportioning 

# Step 1: Set the 'last Non Weekend day of every month' as the last working date
# create a dictionary for this date for each unique month in the data

def last_working_day(year, month, work_days):
    """""
    Sets the last non-weekend day of every month as the last working date

    Args:
        year (scalar): Year of the date
        month (scalar): Month of the date
        work_days (scalar): Number of working days in the week

    Returns
        last_day (scalar): The last working day of the month
    """


    if month==12:
        month=0
        year+=1

    last_day = datetime(year, month+1, 1) - timedelta(days=1)

    if work_days==5:
        while last_day.weekday() > 4:  # Friday is weekday 4
            last_day -= timedelta(days=1) #subtracting 1 day at a time

    return last_day


def last_week_apportion(tactic_df,date_col_name,kpi_col_list,work_days):
    """""
    Proportionately allocates KPIs of last week in that month accurately to each month 
    based on number of working days in that week
    
    Args:
        tactic_df (dataframe): Dataframe containing geo-month and KPI information
        date_col_name (string): Column in tactic_df which corresponds to date
        kpi_col_list (list): List of KPI columns to be apportioned 
        work_days (scalar): Number of working days in the week

    Returns
        tactic_df (dataframe): Dataframe with KPI columns apportioned
    """
    def rename_adjusted(kpi_name):
        new_name = "adjusted_" + kpi_name
        return new_name

    # Step 1: Calculate last working date and create month level column

    tactic_df['month'] = tactic_df[date_col_name].dt.to_period('M')
    last_working_day_dict = {month: last_working_day(month.year, month.month,work_days) for month in tactic_df['month'].unique()}
    tactic_df['last_working_date'] = tactic_df['month'].map(last_working_day_dict)

    # Step 2: Calculate day difference from week_start_date to working_date
    tactic_df['day_diff'] = (tactic_df['last_working_date'] - tactic_df[date_col_name] + timedelta(days=1)).dt.days

    # Step 3: Filter weeks with day_diff < work_days and calculate adjusted calls
    adjusted_col_list = []
    for kpi_name in kpi_col_list:
        tactic_df[rename_adjusted(kpi_name)] = tactic_df.apply(lambda row: ((work_days-row['day_diff']) / work_days) * row[kpi_name] if row['day_diff'] < work_days else 0, axis=1)
        adjusted_col_list.append(rename_adjusted(kpi_name))

    # Step 4: Subtract adjusted calls from original calls and add new rows with adjusted calls for the next month
    # Original rows with calls subtracted


    for kpi_name in kpi_col_list:
        tactic_df[kpi_name] = tactic_df[kpi_name] - tactic_df[rename_adjusted(kpi_name)]

    # New rows with adjusted calls on the first day of the month
    new_rows = tactic_df[tactic_df[adjusted_col_list].gt(0).any(axis=1)].copy()
    new_rows[date_col_name] = new_rows[date_col_name] + pd.offsets.MonthBegin()

    # Add new rows
    for kpi_name in kpi_col_list:
        new_rows[kpi_name] = new_rows[rename_adjusted(kpi_name)]

    # Combine original and new rows
    tactic_df = pd.concat([tactic_df, new_rows], ignore_index=True)
    tactic_df.drop(['last_working_date','day_diff','month'], axis=1, inplace=True)

    #Removing the adjusted calls columns
    for adj_col in adjusted_col_list:
        tactic_df.drop(adj_col, axis=1, inplace=True)

    return tactic_df


# ------------------------------------------------------------------------------


# --- CONFIGURATION ---
def kpi_table(df: pl.DataFrame):
    numerical_config_dict = {}
    categorical_config_dict = {}
    try:
        st.success("File loaded successfully!")
        st.write("### Preview of Data")
        st.dataframe(df.head().to_pandas())  # Convert polars df head to pandas for Streamlit
        
        # Polars dtype to python type mapping for filtering
        # Numeric types in Polars: Int32, Int64, Float32, Float64, etc.
        numeric_dtypes = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        category_dtypes = [pl.Utf8, pl.Categorical]

        # Step 2: Numerical Column Selection
        num_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_dtypes]
        st.subheader("🔢 Numerical Columns")
        selected_num_cols = st.multiselect("Select numerical columns:", num_cols)

        if selected_num_cols:
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

            
        # Step 3: Categorical Column Selection
        cat_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in category_dtypes]
        st.subheader("🔠 Categorical Columns")
        selected_cat_cols = st.multiselect("Select categorical columns:", cat_cols)

        if selected_cat_cols:
            cat_ops_df = pd.DataFrame({
                "Column": selected_cat_cols,
                "Operation": [""] * len(selected_cat_cols)
            })
            #Remove pivot option
            cat_operation_options = ["count", "distinct count"]
            edited_cat_df = st.data_editor(
                cat_ops_df,
                column_config={
                    "Operation": st.column_config.SelectboxColumn("Operation", options=cat_operation_options)
                },
                num_rows="fixed",
                key="categorical_editor"
            )

            #st.write("📋 Categorical Operations")
            #st.dataframe(edited_cat_df)

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
    with st.expander("Upload CSV Files"):
        uploaded_files = st.file_uploader(
            "Select CSV file(s) to upload:",
            type="csv",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}"
        )
        st.session_state["uploaded_files"] = uploaded_files


    with st.expander("Connect to Azure Blob Storage"):
        st.markdown("This section will allow you to securely connect to " \
        "an Azure Blob Storage account and load files directly.")

        st.subheader("Azure Credentials")
        blob_account_name = st.text_input("Azure Storage Account Name", 
                                          disabled=True, 
                                          placeholder="e.g., mystorageaccount")
        blob_container_name = st.text_input("Container Name",
                                            disabled=True,
                                            placeholder="e.g., marketing-data")
        blob_sas_token = st.text_input("SAS Token", 
                                       type="password", 
                                       disabled=True, 
                                       placeholder="Your secure token")

        st.subheader("File Selection")
        st.selectbox("Browse files in container",
                     options=["-- Coming soon --"], 
                     disabled=True)
        st.button("Refresh File List", disabled=True)

        st.markdown("---")
        st.info("You will be able to load files directly from your " \
        "Azure Blob Storage container using the credentials above.")

        # TO IMPLEMENT LATER:
        # from azure.storage.blob import ContainerClient
        
        # container_url = f"https://{blob_account_name}.blob.core.windows.net/{blob_container_name}?{blob_sas_token}"
        # container_client = ContainerClient.from_container_url(container_url)
        # blobs = [blob.name for blob in container_client.list_blobs()]
        # blob_client = container_client.get_blob_client(blob_name)
        # stream = blob_client.download_blob().readall()
        # df_final = pd.read_csv(io.BytesIO(stream))

    df_final = None
        
tab1, tab2 = st.tabs(["Data Mapping", "Data Profiling"])

# --- DATA MAPPING PAGE ---
with tab1:
    if uploaded_files:
        st.session_state["current_step"] = 1
        st.write(st.session_state["current_step"])
       
        if len(uploaded_files) == 1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Standardize columns for a single file(abdul is here)")
                file = uploaded_files[0]
                df = pd.read_csv(file)
                st.write(f"**Currently editing:** `{file.name}`")
                st.dataframe(df[:3], hide_index=True)

                # Step 1: Column Selection
                selected_cols = st.multiselect(
                    f"Select columns from `{file.name}`:",
                    df.columns.tolist(),
                    default=df.columns.tolist(),
                    key="select_cols_0"
                )
                df = df[selected_cols]

                # Step 2a: Column Renaming
                rename_columns = pd.DataFrame({
                    "Rename columns if necessary": df.columns
                })
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
                rename_dict = dict(zip(
                    df.columns,
                    edited_renamed_df["Rename columns if necessary"]
                ))
                df = df.rename(columns=rename_dict)

                # Step 2b: Date Standardization
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
                            df[date_col] = pd.to_datetime(
                                df[date_col],
                                format=date_format,
                                errors='coerce'
                            )
                            st.success(f"Date column `{date_col}` standardized!")
                        except Exception as e:
                            st.error(f"Failed to parse `{date_col}`: {e}")

                npi_col = "account_npi"  # adjust if this was renamed above
                if npi_col in df.columns:
                    s = df[npi_col].astype(str).str.strip()
                    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
                    s = s.str.replace(r"\.0$", "", regex=True)          # drop trailing .0
                    mask_numeric = s.str.isnumeric()                    # or s.str.isdigit()
                    mask_len = s.str.len().eq(10)
                    mask_start = s.str.startswith(("1", "2"))
                    mask_valid = mask_numeric & mask_len & mask_start
                    df["npi_clean"] = s.where(mask_valid)               # invalid/missing -> NaN
                    df["npi_valid"] = df["npi_clean"].notna()
                # Final clean-up
                str_cols = df.select_dtypes(include=["object", "string"]).columns
                df[str_cols] = df[str_cols].fillna("")

                st.session_state["df_final"] = df
                st.session_state["current_step"] = 3

            with col2:
                st.markdown("### Final Preview")
                st.dataframe(df, hide_index=True)

        else:
            st.markdown("### Standardize columns for multiple files")
            renamed_columns_list = []

            for i, file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 3])
                with col1:
                    with st.expander(f"**Currently editing:** `{file.name}`"):
                        df = pd.read_csv(file)
                        st.dataframe(df[:3], hide_index=True)

                        selected_cols = st.multiselect(
                            f"Select columns from `{file.name}`:",
                            df.columns.tolist(),
                            default=df.columns.tolist(),
                            key=f"select_cols_{i}"
                        )
                        df = df[selected_cols]

                        rename_columns = pd.DataFrame({
                            "Rename columns if necessary": df.columns
                        })
                        column_config = {
                            "Rename columns if necessary": st.column_config.TextColumn()
                        }

                        edited_renamed_df = st.data_editor(
                            rename_columns,
                            column_config=column_config,
                            num_rows="fixed",
                            use_container_width=True,
                            key=f"rename_editor_{i}",
                            hide_index=True
                        )
                        rename_dict = dict(zip(
                            df.columns,
                            edited_renamed_df["Rename columns if necessary"]
                        ))
                        df = df.rename(columns=rename_dict)
                        renamed_columns_list.append(set(df.columns))
                        detected_dates = detect_date_columns_by_sampling(df)
                        date_cols = st.multiselect("Select date columns", options=df.columns, default=detected_dates, key=f"date_col_{i}")
                        for date_col in date_cols:
                            date_format = st.selectbox(
                                f"Select format for `{date_col}`:",
                                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                                index=None,
                                key=f"date_format_{i}_{date_col}"
                            )
                            if date_format:
                                try:
                                    df[date_col] = pd.to_datetime(
                                        df[date_col],
                                        format=date_format,
                                        errors='coerce'
                                    )
                                    st.success(f"Standardized `{date_col}`!")
                                except Exception as e:
                                    st.error(f"Could not parse `{date_col}`: {e}")
                        

                        npi_col = "account_npi"
                        if npi_col in df.columns:
                            s = df[npi_col].astype(str).str.strip()
                            s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
                            s = s.str.replace(r"\.0$", "", regex=True)
                            mask_valid = s.str.isnumeric() & s.str.len().eq(10) & s.str.startswith(("1", "2"))
                            df[npi_col] = s.where(mask_valid)  # keep NaN for invalids
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
                    st.markdown(f"**Data Frame Preview**: `{file.name}`")
                    st.dataframe(df, hide_index=True)



            with st.expander("Merge Files"):
                st.subheader("Select Merge Strategy")

                common_columns = set.intersection(*renamed_columns_list)
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

                    # Post-processing: cleanup and standardize date columns
                    if join_keys:
                        for col_name in join_keys:
                            if col_name in df_final.columns:
                                df_final = df_final[df_final[col_name].notnull()]
                    else:
                        print("No join_keys provided — skipping null identifier filtering.")

                    for col in df_final.select_dtypes(include="object").columns:
                        if "date" in col.lower():
                            try:
                                df_final[col] = pd.to_datetime(df_final[col], errors='coerce')
                            except:
                                pass

                    df_final = pl.from_pandas(df_final)
                    df_final = df_final.with_columns([
                        pl.col(col).cast(pl.Date)
                        for col, dtype in zip(df_final.columns, df_final.dtypes)
                        if dtype == pl.Datetime
                    ])

                    st.session_state["df_final"] = df_final

                    st.session_state["merge_done"] = True  

                    st.markdown("### Final Data Frame")
                    st.dataframe(df_final)
                    try:
                        if isinstance(df_final, pl.DataFrame):
                            merged_csv= df_final.to_pandas().to_csv(index=False).encode("utf-8")
                        else:
                            merged_csv = df_final.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Download merged CSV",
                            data= merged_csv,
                            file_name=f"merged_final.csv",
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

    # Get merged dataframe (Polars or pandas)
    df_any = st.session_state["df_final"]
    if isinstance(df_any, pl.DataFrame):
        pdf = df_any.to_pandas()
    else:
        pdf = df_any

    # --- Overview ---
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(pdf):,}")
    c2.metric("Columns", f"{pdf.shape[1]:,}")
    try:
        mem_mb = pdf.memory_usage(deep=True).sum() / (1024**2)
        c3.metric("Memory", f"{mem_mb:.2f} MB")
    except Exception:
        c3.metric("Memory", "n/a")
    st.dataframe(pdf.head(10), use_container_width=True)

    # --- Schema / dtypes ---
    st.subheader("Schema")
    schema_df = pd.DataFrame({"column": pdf.columns, "dtype": [str(t) for t in pdf.dtypes]})
    st.dataframe(schema_df, use_container_width=True)

    # --- Missingness ---

    # Button to remove rows with BOTH missing NPI and missing Date
    st.subheader("No HCP, Date Row removal")
    # Let the user pick the relevant columns
    npi_candidates = [c for c in pdf.columns if "npi" in c.lower()]
    date_candidates = [c for c in pdf.columns if "date" in c.lower()] + \
                    [c for c in pdf.select_dtypes(include=["datetime64[ns]"]).columns]

    npi_col_rm = st.selectbox("NPI column for removal rule:", options=pdf.columns, 
                            index=pdf.columns.get_loc(npi_candidates[0]) if npi_candidates else 0)
    date_col_rm = st.selectbox("Date column for removal rule:", options=pdf.columns, 
                            index=pdf.columns.get_loc(date_candidates[0]) if date_candidates else 0)

    if st.button("Remove rows where BOTH NPI and Date are missing"):
        # Normalize NPI string empties to NaN for the check
        npi_series_rm = pdf[npi_col_rm].astype("string")
        npi_na = npi_series_rm.isna() | (npi_series_rm.str.strip() == "")
        # Coerce date to datetime for robust NA detection
        date_series_rm = pdf[date_col_rm]
        if not np.issubdtype(date_series_rm.dtype, np.datetime64):
            date_series_rm = pd.to_datetime(date_series_rm, errors="coerce")
        date_na = date_series_rm.isna()

        # Identify rows to drop: both missing
        drop_mask = npi_na & date_na
        dropped = int(drop_mask.sum())
        pdf_clean = pdf.loc[~drop_mask].copy()

        st.session_state["df_final"] = pdf_clean
        st.success(f"Dropped {dropped:,} rows where BOTH `{npi_col_rm}` and `{date_col_rm}` were missing.")
        st.dataframe(pdf_clean.head(10), use_container_width=True)


    st.subheader("NPI Format check")
    npi_col = st.selectbox(
        "Select the column to check for NPI format:",
        options=pdf.columns,
        help="Choose the column containing NPI numbers"
    )

    # Convert entire column to string explicitly using pandas
    npi_series = pdf[npi_col].apply(lambda x: str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x))

    # Now strip spaces and remove all non-digits
    npi_strs = npi_series.str.strip().str.replace(r"[^\d]", "", regex=True)

    # Validate pattern: start with 1 or 2 followed by 9 digits (total 10 digits)
    npi_pattern = re.compile(r"^[12]\d{9}$")
    npi_valid = npi_strs.str.match(npi_pattern)

    # --- Format validation results ---
    n_valid = npi_valid.sum()
    n_invalid = (~npi_valid).sum()

    st.write(f"Format-valid NPIs (10 digits, starts 1/2): {n_valid:,}")
    st.write(f"Format-invalid NPIs: {n_invalid:,}")

    if n_invalid > 0:
        st.write("Sample format-invalid NPIs:")
        st.dataframe(npi_strs[~npi_valid].drop_duplicates().head(10))

    # --- Luhn validation with memo on format-valid NPIs ---
    if n_valid > 0:
        format_valid_npis = npi_strs[npi_valid]
        
        # Use set/dict memo to validate unique NPIs once
        cache = {}
        luhn_results = []
        
        for npi in format_valid_npis:
            if npi in cache:
                luhn_results.append(cache[npi])
            else:
                valid = luhn_valid_npi(npi)
                cache[npi] = valid
                luhn_results.append(valid)
        
        luhn_valid_series = pd.Series(luhn_results, index=format_valid_npis.index)
        
        n_luhn_valid = luhn_valid_series.sum()
        n_luhn_invalid = len(luhn_valid_series) - n_luhn_valid
        
        st.write(f"Luhn-valid NPIs (among format-valid): {n_luhn_valid:,}")
        st.write(f"Luhn-invalid NPIs (among format-valid): {n_luhn_invalid:,}")
        
        if n_luhn_invalid > 0:
            luhn_invalid_npis = format_valid_npis[~luhn_valid_series]
            st.write("Sample Luhn-invalid NPIs:")
            st.dataframe(luhn_invalid_npis.drop_duplicates().head(10))

    # Store cleaned strings for reuse in duplicates section
    npi_col_for_dups = npi_col



                    