import streamlit as st
import pandas as pd

st.set_page_config(page_title= "Anna ProcTimize", layout = "wide")
st.title("DATA INGESTION")

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
current_step = 1 # Change this to match your current progress (e.g., 3 = "Filter")

# Calculate progress percentage
progress_percent = int((current_step / (len(steps) - 1)) * 100)

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

col1, col2  = st.columns(2)

with col1:
    st.markdown("Add description of the functionality of this page later")

with col2:

    with st.expander("Upload CSV Files"):
        uploaded_files = st.file_uploader(
            "Select CSV file(s) to upload:",
            type="csv",
            accept_multiple_files=True
        )

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
        if len(uploaded_files) == 1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Standardize columns for a single file")
                file = uploaded_files[0]
                df = pd.read_csv(file)
                st.write(f"**Currently editing:** `{file.name}`")
                st.dataframe(df[:3], hide_index=True)
                 # Step 1: Column Selection
                selected_cols = st.multiselect(
                     f"Select columns from `{file.name}`:",
                     df.columns.tolist(),
                     default=df.columns.tolist(),
                     key=f"select_cols_0")
                
                df = df[selected_cols]

                # Step 2a: Column Renaming and Selection
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
                    key="rename_editor",
                    hide_index=True
                )

                rename_dict = dict(zip(
                    df.columns,  # limit to available renamed rows
                    edited_renamed_df["Rename columns if necessary"]
                ))

                df = df.rename(columns=rename_dict)
                
                # Step 2b: Date Standardization
                date_cols = st.multiselect(
                    "Select Date Columns (if any):", 
                    df.columns.tolist(), 
                    key="date_cols_single"
                )
                
                for date_col in date_cols:
                    date_format = st.selectbox(
                        f"Select current date format for `{date_col}` being used:",
                        ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                        index=None,
                        key=f"date_format_single_{date_col}"
                    )

                    if date_format:
                        try:
                            df[date_col] = pd.to_datetime(
                                df[date_col], 
                                format=date_format, 
                                errors='coerce'
                            )
                            st.success(f"Date column `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                        except Exception as e:
                            st.error(f"Failed to parse date column `{date_col}`: {e}")

                # Final clean-up before conversion
                str_cols = df.select_dtypes(include=["object", "string"]).columns
                df[str_cols] = df[str_cols].fillna("")

                st.session_state["df_final"] = df_final


            with col2:
                st.markdown("### Data frame preview")
                st.dataframe(df, hide_index=True)

# --- DATA PROFILING PAGE ---
with tab2:
    pass