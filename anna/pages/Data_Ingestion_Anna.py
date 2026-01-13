import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import requests
import io
import json
import time
from datetime import datetime,timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="ProcTimize", layout="wide")
st.title("DATA INGESTION")

st.markdown("""
<style>        
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

# Helper functions

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

    def _rename_adjusted(kpi_name):
        return "adjusted_" + kpi_name

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


# Define steps and current step (0-based index)
steps = ["Upload", "Standardize", "Merge", "Filter", "Pivot", "Save"]
current_step = 1 # Change this to match your current progress (e.g., 3 = "Filter")

# Calculate progress percentage
progress_percent = int((current_step / (len(steps) - 1)) * 100)
st.write(progress_percent)
st.write(current_step)
st.write(len(steps)-1)

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


# === Entry Point ===
st.divider()
            
col1, col2  = st.columns(2)

with col1:
    st.markdown("Upload files for a single marketing channel to standardize the data merge multiple files, filter your data or create a pivot table.")

with col2:
    # --- FILE UPLOAD ---
    with st.expander("Upload CSV Files"):
        uploaded_files = st.file_uploader(
            "Select CSV file(s) to upload:",
            type="csv",
            accept_multiple_files=True
        )

    # --- ☁️ BLOB STORAGE PLACEHOLDER ---
    with st.expander("Connect to Azure Blob Storage"):
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

        # TO IMPLEMENT LATER:
        # from azure.storage.blob import ContainerClient
        #
        # container_url = f"https://{blob_account_name}.blob.core.windows.net/{blob_container_name}?{blob_sas_token}"
        # container_client = ContainerClient.from_container_url(container_url)
        # blobs = [blob.name for blob in container_client.list_blobs()]
        # blob_client = container_client.get_blob_client(blob_name)
        # stream = blob_client.download_blob().readall()
        # df_final = pd.read_csv(io.BytesIO(stream))

    df_final = None

tab1, tab2 = st.tabs(["Data Mapping", "Data Profiling"])

with tab1:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

    # Sample data
    df = pd.DataFrame({
        "Product": ["A", "B", "C", "A", "B"],
        "Region": ["East", "West", "East", "North", "South"],
        "Sales": [100, 150, 120, 130, 160]
    })

    # Configure grid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        filter=True, 
        sortable=True, 
        editable=False, 
        resizable=True,
        autoHeight=True
    )
    # Add autoSizeAllColumns script on grid ready
    gb.configure_grid_options(onGridReady=JsCode("""
        function(params) {
            params.api.sizeColumnsToFit();
            setTimeout(function() {
                params.api.resetRowHeights();
                params.api.sizeColumnsToFit();
            }, 100);
        }
    """))

    grid_options = gb.build()

    # Render AgGrid
    AgGrid(
        df,
        gridOptions=grid_options,
        theme="streamlit",              # Options: 'streamlit', 'light', 'dark', 'blue', etc.
        enable_enterprise_modules=False,
        use_container_width=True,
        height=400,                     # Adjust as needed
        fit_columns_on_grid_load=False, # Disable default stretching to use autosizing instead
        allow_unsafe_jscode=True        # Required for JsCode
    )
    # Sample Data
    df = pd.DataFrame({
        "Product": ["A", "B", "C", "A", "B"],
        "Region": ["East", "West", "East", "North", "South"],
        "Sales": [100, 150, 120, 130, 160]
    })

    # Use columns to align filters horizontally
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        product_filter = st.multiselect(
            "Product", 
            options=df["Product"].unique(), 
            default=df["Product"].unique()
        )

    with col2:
        region_filter = st.multiselect(
            "Region", 
            options=df["Region"].unique(), 
            default=df["Region"].unique()
        )

    # You can leave col3 empty or use it for numerical filtering
    with col3:
        sales_min, sales_max = st.slider(
            "Sales Range", 
            int(df["Sales"].min()), 
            int(df["Sales"].max()), 
            (int(df["Sales"].min()), int(df["Sales"].max()))
        )

    # Apply filters
    filtered_df = df[
        (df["Product"].isin(product_filter)) & 
        (df["Region"].isin(region_filter)) & 
        (df["Sales"].between(sales_min, sales_max))
    ]

    # Show table
    st.data_editor(
        filtered_df,
        use_container_width=True,
        disabled=True
    )
with tab2:
    st.write("work in progress")


