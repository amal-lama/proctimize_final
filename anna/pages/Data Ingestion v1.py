import streamlit as st
import pandas as pd


with st.expander("Upload CSV Files"):
        uploaded_files = st.file_uploader(
            "Select CSV file(s) to upload:",
            type="csv",
            accept_multiple_files=True)

def update_dataframe():
    edited_renamed_df = st.session_state["edited_renamed_df"]
    rename_dict = dict(zip(df.columns, edited_renamed_df))
    df = df.rename(columns=rename_dict)
    st.session_state["renamed_df"] = 


if uploaded_files:
      if len(uploaded_files) == 1:
            file = uploaded_files[0]
            df = pd.read_csv(file)
            altered_df = df.copy()
            st.session_state["altered_df"] = altered_df
            edited_renamed_df = None
            st.dataframe(df[:3], hide_index=True)

            # Step 1: Column Renaming and Selection
            rename_columns = pd.DataFrame({ "Rename columns if necessary": altered_df.columns})
            column_config = {"New Column Name": st.column_config.TextColumn()}

            edited_renamed_df = st.data_editor(
                    rename_columns,
                    column_config=column_config,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="rename_editor",
                    on_change= update_dataframe()
                    hide_index=True)
            
            st.session_state["edited_renamed_df"] = edited_renamed_df['Rename columns if necessary']
        
            