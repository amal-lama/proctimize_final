import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import requests
import io
import json
import time
from datetime import datetime,timedelta

# Helper functions


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

def rename_adjusted(kpi_name):
    new_name = "adjusted_" + kpi_name
    return new_name


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

# Detect time granularity present in dataframe input by user

def detect_date_granularity(df, date_column):
    # Convert to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        df = df.drop_duplicates(subset = [date_column])

        # Sort dates
        df = df.sort_values(by=date_column)

        # Calculate differences between consecutive dates
        date_diffs = df[date_column].diff().dropna()

        # Get the most common time difference
        most_common_diff = date_diffs.mode()[0]  

        # Determine granularity based on the most common difference
        if most_common_diff.days == 1:
            return "Daily"
        elif most_common_diff.days == 7:
            return "Weekly"
        elif most_common_diff.days in [28, 29, 30, 31]:
            return "Monthly"
        elif most_common_diff.days >= 365:
            return "Yearly"
        else:
            st.warning(f"Irregular Date. Please check the column format.")
        
    except Exception as e:
        st.warning(f"The column '{date_column}' is not in a valid date format. Please check the column format.")


# ------------------------------------------------------------------------------

def modify_granularity_pandas(
    df,
    geo_column,
    date_column,
    granularity_level_df,
    granularity_level_user_input,
    work_days,
    numerical_config_dict,
    categorical_config_dict
):
    def get_agg_dict(numerical_dict, categorical_dict):
        agg_dict = {}
        
        for col, op in numerical_dict.items():
            if op == "sum":
                agg_dict[col] = "sum"
            elif op == "average":
                agg_dict[col] = "mean"
            elif op == "min":
                agg_dict[col] = "min"
            elif op == "max":
                agg_dict[col] = "max"
            elif op == "product":
                agg_dict[col] = lambda x: np.prod(x.dropna())

        for col, op in categorical_dict.items():
            if op == "count":
                #agg_dict[f"{col}_count"] = (col, "count")
                agg_dict[col] = (col, "count")
            elif op == "distinct count":
                #agg_dict[f"{col}_count_distinct"] = (col, pd.Series.nunique)
                agg_dict[col] = (col, pd.Series.nunique)

        return agg_dict

    # No transformation needed
    if granularity_level_df == granularity_level_user_input:
        selected_cols = [geo_column, date_column] + list(numerical_config_dict.keys()) + list(categorical_config_dict.keys())
        return df[selected_cols], date_column

    df[date_column] = pd.to_datetime(df[date_column])
    agg_dict = get_agg_dict(numerical_config_dict, categorical_config_dict)

    # Daily → Weekly
    if granularity_level_df == "Daily" and granularity_level_user_input == "Weekly":
        df["week_date"] = df[date_column] - pd.to_timedelta(df[date_column].dt.weekday, unit='D')
        grouped = df.groupby([geo_column, "week_date"]).agg(**agg_dict).reset_index()
        #grouped = df.groupby([geo_column, "week_date"]).agg(agg_dict).reset_index()
        return grouped, "week_date"

    # Daily → Monthly
    elif granularity_level_df == "Daily" and granularity_level_user_input == "Monthly":
        df["month_date"] = df[date_column].values.astype('datetime64[M]')
        grouped = df.groupby([geo_column, "month_date"]).agg(**agg_dict).reset_index()
        #grouped = df.groupby([geo_column, "month_date"]).agg(agg_dict).reset_index()
        return grouped, "month_date"

    # Weekly → Monthly (requires apportioning)
    elif granularity_level_df == "Weekly" and granularity_level_user_input == "Monthly":
        df = last_week_apportion(df, date_column, list(numerical_config_dict.keys()), work_days)
        df["month_date"] = df[date_column].values.astype('datetime64[M]')
        grouped = df.groupby([geo_column, "month_date"]).agg(**agg_dict).reset_index()
        #grouped = df.groupby([geo_column, "month_date"]).agg(agg_dict).reset_index()
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

# --- PAGE CONFIG ---
st.set_page_config(page_title="ProcTimize", layout="wide")
st.title("Data Ingestion")

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




# --- FILE UPLOAD ---
uploaded_files = st.file_uploader("📄 Upload one or more CSV files", type="csv", accept_multiple_files=True)

df_final = None

if uploaded_files:
    
    if len(uploaded_files) == 1:
        with st.expander("🧱 Standardize Columns for Single File"):
            file = uploaded_files[0]
            df = pd.read_csv(file)
            st.markdown(f"**File: {file.name}**")

        
            rename_df = pd.DataFrame({"Current Column": df.columns, "New Column Name": df.columns})

            edited_rename_df = st.data_editor(
                rename_df,
                num_rows="dynamic",
                use_container_width=True,
                key="rename_editor"
            )
            rename_dict = dict(zip(
                edited_rename_df["Current Column"], 
                edited_rename_df["New Column Name"]
            ))
            df = df.rename(columns=rename_dict)


            # ✅ Step 2b: Date Standardization
            date_cols = st.multiselect(
                "Select Date Columns (if any):", 
                df.columns.tolist(), 
                key="date_cols_single"
            )

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
                        df[date_col] = pd.to_datetime(
                            df[date_col], 
                            format=date_format, 
                            errors='coerce'
                        ).dt.strftime("%Y/%m/%d")
                        st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                    except Exception as e:
                        st.error(f"❌ Failed to parse date column `{date_col}`: {e}")

            # Final clean-up before conversion
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            df[str_cols] = df[str_cols].fillna("")

            # Convert to Polars for filtering
            df_final = pl.from_pandas(df)


    else:
        
        with st.expander("🧱 Standardize Columns for Multiple Files"):
            column_mappings = []
            dfs = []
            renamed_columns_list = []

            for i, file in enumerate(uploaded_files):
                st.markdown(f"**File {i+1}: {file.name}**")
                df = pd.read_csv(file)
                dfs.append(df)

                # ✅ Step 1: Column Selection
                selected_cols = st.multiselect(
                    f"Select columns from `{file.name}`:", 
                    df.columns.tolist(), 
                    default=df.columns.tolist(), 
                    key=f"select_cols_{i}"
                )
                df = df[selected_cols]

                # ✅ Step 2: Interactive Data Editor for Renaming
                rename_df = pd.DataFrame({
                    "Current Column": selected_cols, 
                    "New Column Name": selected_cols
                })

                edited_rename_df = st.data_editor(
                    rename_df,
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

                # ✅ Optional: Apply renaming immediately if needed
                df = df.rename(columns=rename_dict)

                # 📅 Step: Ask for Date Format and Standardize Date Columns
                date_cols = st.multiselect(
                    f"Select Date Columns in `{file.name}` (if any):", 
                    selected_cols, 
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

                    # Convert and standardize to YYYY/MM/DD format
                    if date_format:
                        try:
                            df[date_col] = pd.to_datetime(
                                df[date_col], 
                                format=date_format, 
                                errors='coerce'
                            ).dt.strftime("%Y/%m/%d")
                            df[date_col] = df[date_col].fillna("")

                            st.success(f"✅ Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                        except Exception as e:
                            st.error(f"❌ Failed to parse date column `{date_col}`: {e}")
                            

        with st.expander("Merge Files:"):

            common_columns = set.intersection(*renamed_columns_list)
            st.subheader("🔗Select Merge Strategy")
            merge_strategy = st.radio("Merge type:", ["Vertical Stack", "Horizontal Join"])

            if merge_strategy == "Horizontal Join":
                join_keys = st.multiselect("Select join key(s):", list(common_columns))
                join_type = st.selectbox("Join type:", ["inner", "left", "right", "outer"])
            else:
                join_keys = join_type = None
        
            # Merge DataFrames
            if merge_strategy in ["vertical", "Vertical Stack"]:
                df_final = pd.concat(dfs, ignore_index=True, sort=False)

            elif merge_strategy in ["horizontal", "Horizontal Join"]:
                if not join_keys:
                    #raise ValueError("Join key must be provided for horizontal joins.")
                    st.warning("Join key must be provided for horizontal joins.")
                    st.stop()
                df_final = dfs[0]
                for df in dfs[1:]:
                    df_final = pd.merge(df_final, df, on=join_keys, how=join_type)
            else:
                #raise ValueError("Invalid merge strategy. Choose 'vertical' or 'horizontal'.")
                st.error("Invalid merge strategy. Choose 'vertical' or 'horizontal'.")
                st.stop()
                
            # Remove rows with nulls in identifier columns and drop duplicates
            if join_keys:
                for col_name in join_keys:
                    if col_name in df_final.columns:
                        df_final = df_final[df_final[col_name].notnull()]
                # merged_df = df_final.drop_duplicates()
            else:
                print("⚠️ No join_keys provided — skipping null identifier filtering.")

            df_final = pl.from_pandas(df_final)
            st.session_state["df_final"] = df_final

    # ---------------------- FILTERING SECTION ----------------------
    with st.expander("🔍Filter Data"):

        if "df_final" in st.session_state:
            df_final = st.session_state["df_final"]

        if df_final is None or df_final.shape[0] == 0:
            st.error("❌ No data available for filtering. Please check the input files.")
            st.stop()
            
        if df_final is not None:
            st.write("### Sample of the Dataset")



            st.dataframe(df_final.head())

            date_column = st.selectbox("Select the column representing Date in merged dataset", df_final.columns)
            st.session_state["date_col"] = date_column


            if date_column:
                if df_final[date_column].dtype in [pl.Utf8, pl.Object]:
                    try:
                        df_final = df_final.with_columns(
                            pl.col(date_column).str.strptime(pl.Date, strict=False).alias(date_column)
                        )
                    except Exception:
                        st.warning("⚠️ Valid date column not selected or parsing failed.")
                        st.stop()
                elif df_final[date_column].dtype != pl.Date:
                    st.warning("⚠️ Selected column is not a valid date column.")
                    st.stop()

                min_date = df_final[date_column].min()
                max_date = df_final[date_column].max()

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

                if start_date > end_date:
                    st.error("Start date must be before or equal to end date.")
                    st.stop()
                else:
                    df_final = df_final.filter((pl.col(date_column) >= start_date) & (pl.col(date_column) <= end_date))

            df_filtered = df_final.clone()
            categorical_cols = [col for col in df_filtered.columns if df_filtered[col].dtype == pl.Utf8]

            if categorical_cols:
                st.write("### Categorical Column(s) to filter on")
                selected_cat_cols = st.multiselect("Select categorical columns to filter:", categorical_cols)

                for col in selected_cat_cols:
                    unique_vals = df_filtered[col].unique().to_list()
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
                        filter_conditions.append(pl.col(col).is_null() | (pl.col(col).cast(str).str.strip_chars().eq("")))

                    selected_non_blank_vals = [val for val in selected_vals if val != "<BLANK>"]
                    if selected_non_blank_vals:
                        filter_conditions.append(pl.col(col).cast(str).is_in(selected_non_blank_vals))

                    if filter_conditions:
                        combined_condition = filter_conditions[0]
                        for cond in filter_conditions[1:]:
                            combined_condition = combined_condition | cond
                        df_filtered = df_filtered.filter(combined_condition)
                        st.session_state["df_filtered"] = df_filtered
                        st.session_state["filter_complete"] = True

        if st.session_state.get("filter_complete") and "df_filtered" in st.session_state:
            st.write("Final Filtered Dataset")
            st.dataframe(st.session_state["df_filtered"].head(100).to_pandas())

            csv_bytes = df_filtered.write_csv()
            st.download_button("📥 Download CSV", data=csv_bytes, file_name="final_filtered_data.csv", mime="text/csv")
            
    
    # ---------------------- GRANULARITY SECTION ----------------------
    with st.expander("Modify Time Granularity"):

        if all(k in st.session_state for k in ["df_filtered", "date_column"]): 
            df_filtered = st.session_state["df_filtered"]
            date_column = st.session_state["date_column"]
        
        if df_filtered is not None:
            #st.write("📊 Sample of Uploaded Data", df_filtered.head(10))

            column_names = df_filtered.columns
            geo_col = st.selectbox("Select the Grouping Column", options=column_names)
            date_col = date_column
            
            # Initialize granularity in session state
            if "granularity" not in st.session_state:
                st.session_state["granularity"] = ""

            #Incorporate Job polling
            if st.button("Detect Granularity"):
                df_filtered_pandas = df_filtered.to_pandas()
                granularity = detect_date_granularity(df_filtered_pandas, date_col)
                st.session_state["granularity"] = granularity
                if granularity:
                    st.success(f"📈 Detected Granularity: **{granularity}**")
                else:
                    st.warning("⚠️ No result found")

            # Get granularity from session state
            granularity = st.session_state.get("granularity", "")


            #NEW 
            # Calling the KPI function
            numerical_config_dict={}
            categorical_config_dict={}

            st.subheader("Creating KPI Table")
            config_result = kpi_table(df_filtered)
            if config_result is not None:
                numerical_config_dict, categorical_config_dict = config_result
                st.session_state["numerical_config_dict"] = numerical_config_dict
                st.session_state["categorical_config_dict"] = categorical_config_dict
                #st.json(numerical_config_dict)
                #st.json(categorical_config_dict)

            # Integrated Analytics Database
            st.subheader("Creating Unified Database for Channel")

            if granularity:
                if granularity == 'Daily':
                    granularity_options = ['Weekly','Monthly']
                elif granularity == 'Weekly':
                    granularity_options = ['Weekly','Monthly']
                elif granularity == 'Monthly':
                    granularity_options = ['Monthly']

                time_granularity_user_input = st.selectbox('Choose the time granularity level', granularity_options)
                st.write('You selected time granularity: ', time_granularity_user_input)
                st.session_state["user_input_time_granularity"] = time_granularity_user_input

                # Modifying the granularity
                if st.button("Modify Granularity"):
                    numerical_config_dict = st.session_state.get('numerical_config_dict', {})
                    categorical_config_dict = st.session_state.get('categorical_config_dict', {})

                    #st.json(numerical_config_dict)
                    #st.json(categorical_config_dict)

                    if date_col.strip() == "" or geo_col.strip() == "" or time_granularity_user_input.strip() == "":
                        st.warning("Please enter the required columns")
                    else:
                        # Modify granularity
                        df_transformed, new_date_col = modify_granularity_pandas(
                            df=df_filtered.to_pandas(),
                            geo_column=geo_col,
                            date_column=date_col,
                            granularity_level_df=granularity,
                            granularity_level_user_input=time_granularity_user_input,
                            work_days=7,
                            numerical_config_dict=numerical_config_dict,
                            categorical_config_dict=categorical_config_dict
                        )

                        # Reapply rename dict if available
                        if "rename_dict" in st.session_state:
                            df_transformed = df_transformed.rename(columns=st.session_state["rename_dict"])

                        st.session_state["df_transformed"] = df_transformed
                        st.session_state["transform_complete"] = True

                # --- Display renaming and download UI after transformation ---
                if st.session_state.get("transform_complete") and "df_transformed" in st.session_state:
                    df_transformed = st.session_state["df_transformed"]
                    st.success("Modified Granular data loaded")
                    st.dataframe(df_transformed)

                    # Rename editor
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

                    df_transformed = df_transformed.rename(columns=rename_dict)
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

                    # Download button
                    csv_bytes = df_transformed.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_bytes,
                        file_name=file_name,
                        mime="text/csv"
                    )
                   
            else:
                st.warning("⚠️ Please load granularity before selecting granularity level.")

            