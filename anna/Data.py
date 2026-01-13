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
                # Step 2a: Column Renaming and Selection
                rename_columns = pd.DataFrame({ 
                    "Rename columns if necessary": df.columns})
                
                column_config = {"New Column Name": st.column_config.TextColumn()}

                edited_renamed_df = st.data_editor(
                    rename_columns,
                    column_config = column_config,
                    num_rows = "dynamic",
                    use_container_width = True,
                    key="rename_editor"
                )
                rename_dict = dict(zip(
                    df.columns, 
                    edited_renamed_df["Rename columns if necessary"]
                ))

                renamed_df = df.rename(columns=rename_dict) 
                if len(edited_renamed_df['Rename columns if necessary']) <= len(df.columns):
                    edited_df = renamed_df[edited_renamed_df['Rename columns if necessary']]
                else:
                    st.error("Raw data does not contain additional columns.")
                
                # Step 2b: Date Standardization

                # Initialize session state storage if not already there
                if "date_column_formats" not in st.session_state:
                    st.session_state.date_column_formats = {}

                # Step 2b: Date Standardization
                date_cols = st.multiselect(
                    "Select Date Columns (if any):", 
                    edited_df.columns.tolist(), 
                    default=list(st.session_state.date_column_formats.keys()),  # show previously selected
                    key="date_cols_single"
                )

                # Remove any date columns that were deselected
                removed_cols = set(st.session_state.date_column_formats.keys()) - set(date_cols)
                for col in removed_cols:
                    del st.session_state.date_column_formats[col]

                # Let user pick/update date format for each selected column
                for date_col in date_cols:
                    default_format = st.session_state.date_column_formats.get(date_col, None)
                    date_format = st.selectbox(
                        f"Select current date format for `{date_col}` being used:",
                        ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                        index=None,
                        key=f"date_format_single_{date_col}"
                    )

                    if date_format:
                        # Save/update the format in session state
                        st.session_state.date_column_formats[date_col] = date_format

                        # Apply the format
                        try:
                            edited_df[date_col] = pd.to_datetime(
                                edited_df[date_col], 
                                format=date_format, 
                                errors='coerce'
                            )
                            st.success(f"✅ `{date_col}` standardized from `{date_format}` to `YYYY/MM/DD` format!")
                        except Exception as e:
                            st.error(f"❌ Failed to parse `{date_col}`: {e}")

                 # Final clean-up before conversion
                str_cols = edited_df.select_dtypes(include=["object", "string"]).columns
                edited_df[str_cols] = edited_df[str_cols].fillna("")
                
            with col2:
                st.markdown("### Data frame preview")
                st.dataframe(df, hide_index= True)


    