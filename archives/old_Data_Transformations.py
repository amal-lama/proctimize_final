import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ProcTimize", layout="wide")
#st.image("img/data-transform.png")

st.markdown("""
    <style>
    /* Sidebar background and base text color */
    section[data-testid="stSidebar"] {
        background-color: #001E96 !important;
        color: white;
    }

    /* Force white text in sidebar headings, labels, etc. */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Optional: style buttons */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #1ABC9C;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Define Adstock function
def geometric_adstock(series, lags, adstock_coeff):
    """Applies geometric Adstock transformation within each region."""

    series = np.array(series, dtype=np.float64)  # Ensure it's numeric
    adstocked = np.zeros_like(series)

    for i in range(len(series)):
        for j in range(lags + 1):
            if i - j >= 0:
                adstocked[i] += (adstock_coeff ** j) * series[i - j]

    return adstocked



# Define Saturation function
def apply_saturation(series, method, power_k=0.5):
    """Applies saturation function on a Pandas Series or NumPy array."""
    series = np.array(series, dtype=np.float64)  # Ensure it's numeric
    
    if method.lower() == "log":
        return np.log1p(series)  # log(1 + x) to avoid log(0)
    elif method.lower() == "power":
        return np.power(series, power_k)
    
    return series  # Return unchanged if no valid method is given


# # Function to apply transformations on df (original data), grouped by geo_column  
# def transform_edited_df(df, edited_df, geo_column):
#     transformed_df = df.copy()

#     for _, row in edited_df.iterrows():

#         channel = row["Channel Name"]
#         lags = int(row["Lags"]) if pd.notna(row["Lags"]) else None
#         adstock_coeff = float(row["Adstock"]) if pd.notna(row["Adstock"]) else None
#         sat_function = row["Saturation Function"]
#         power_k = float(row["Power (k)"]) if sat_function == "Power" else None

#         if channel in transformed_df.columns:

#             if sat_function is None and adstock_coeff is None and lags is None:
#                 transformed_df[channel] = transformed_df[channel]

#             elif sat_function is None and lags is None and adstock_coeff is not None:
#                 st.warning("Please ensure value for Lag is not 'None' when applying Ad Stock")
#                 return None

#             elif sat_function is None and lags is not None and adstock_coeff is None:
#                 transformed_df[f"{channel}_transformed"] = (
#                     transformed_df.groupby(geo_column, as_index=False)[channel]
#                     .shift(lags, fill_value=0)
#                     .fillna(0)
#                 )

#             elif sat_function is None and lags is not None and adstock_coeff is not None:
#                 transformed_df[f"{channel}_transformed"] = (
#                     transformed_df.groupby(geo_column)[channel]
#                     .transform(lambda x: geometric_adstock(x, lags, adstock_coeff))
#                 )

#             elif sat_function is not None and lags is None and adstock_coeff is None:
#                 transformed_df[f"{channel}_transformed"] = (
#                     transformed_df.groupby(geo_column)[channel]
#                     .transform(lambda x: apply_saturation(x, sat_function, power_k))
#                 )

#             elif sat_function is not None and lags is None and adstock_coeff is not None:
#                 st.warning("Please ensure value for Lag is not 'None' when applying Ad Stock")
#                 return None

#             elif sat_function is not None and lags is not None and adstock_coeff is None:
#                 lagged_series = (
#                     transformed_df.groupby(geo_column, as_index=False)[channel]
#                     .shift(lags, fill_value=0)
#                     .fillna(0)
#                 )
#                 transformed_df[f"{channel}_transformed"] = (
#                     lagged_series.groupby(transformed_df[geo_column])
#                     .transform(lambda x: apply_saturation(x, sat_function, power_k))
#                 )

#             else:
#                 transformed_df[f"{channel}_transformed"] = (
#                     transformed_df.groupby(geo_column)[channel]
#                     .transform(lambda x: apply_saturation(
#                         geometric_adstock(x, lags, adstock_coeff),
#                         sat_function,
#                         power_k
#                     ))
#                 )

#     return transformed_df

def transform_edited_df(df, edited_df, geo_column, dependent_variable):
    transformed_df = df.copy()

    for _, row in edited_df.iterrows():
        channel = row["Channel Name"]
        lags = int(row["Lags"]) if pd.notna(row["Lags"]) else None
        adstock_coeff = float(row["Adstock"]) if pd.notna(row["Adstock"]) else None
        sat_function = row["Saturation Function"]
        power_k = float(row["Power (k)"]) if sat_function == "Power" else None

        if channel in transformed_df.columns:
            # Case 1: No transformation
            if sat_function is None and adstock_coeff is None and lags is None:
                transformed_df[channel] = transformed_df[channel]

            # Case 2: Adstock with no lag – invalid
            elif sat_function is None and lags is None and adstock_coeff is not None:
                st.warning("Please ensure value for Lag is not 'None' when applying Ad Stock")
                return None

            # Case 3: Lag only
            elif sat_function is None and lags is not None and adstock_coeff is None:
                lagged_series = (
                    transformed_df.groupby(geo_column, as_index=False)[channel]
                    .shift(lags, fill_value=0)
                    .fillna(0)
                )
                transformed_df[f"{channel}_transformed"] = lagged_series

            # Case 4: Lag + Adstock
            elif sat_function is None and lags is not None and adstock_coeff is not None:
                transformed_df[f"{channel}_transformed"] = (
                    transformed_df.groupby(geo_column)[channel]
                    .transform(lambda x: geometric_adstock(x, lags, adstock_coeff))
                )

            # Case 5: Saturation only
            elif sat_function is not None and lags is None and adstock_coeff is None:
                transformed_df[f"{channel}_transformed"] = (
                    transformed_df.groupby(geo_column)[channel]
                    .transform(lambda x: apply_saturation(x, sat_function, power_k))
                )

            # Case 6: Saturation + Adstock but no lag – invalid
            elif sat_function is not None and lags is None and adstock_coeff is not None:
                st.warning("Please ensure value for Lag is not 'None' when applying Ad Stock")
                return None
            
            # Case 7: Lag + Saturation
            elif sat_function is not None and lags is not None and adstock_coeff is None:
                lagged_series = (
                    transformed_df.groupby(geo_column, as_index = False)[channel]
                    .shift(lags, fill_value=0).fillna(0))

            # Store raw lagged value if this is the special channel
                if channel == f"Lagged {dependent_variable}":
                    st.session_state["raw_lagged_dependent_variable"] = lagged_series.copy()

                transformed_df[f"{channel}_transformed"] = (
                    lagged_series.groupby(transformed_df[geo_column])
                    .transform(lambda x: apply_saturation(x, sat_function, power_k))
                    )

            # Case 8: Full pipeline – Lag + Adstock + Saturation
            else:
                transformed_df[f"{channel}_transformed"] = (
                    transformed_df.groupby(geo_column)[channel]
                    .transform(lambda x: apply_saturation(
                        geometric_adstock(x, lags, adstock_coeff),
                        sat_function,
                        power_k
                    ))
                )

    return transformed_df


# Function to handle user input and apply transformations
def user_input(date_column, geo_column, df):

    remove_cols = [date_column, geo_column]
    filtered_df = df.drop(columns=remove_cols)

    # Define initial data for user input
    channel_names = filtered_df.columns
    

    data = pd.DataFrame({
        "Channel Name": channel_names,
        "Saturation Function": [None] * len(channel_names),
        "Power (k)": [0.5] * len(channel_names),
        "Lags": [1] * len(channel_names),
        "Adstock": [0.5] * len(channel_names)
    })

    # Streamlit Data Editor
    edited_df = st.data_editor(
        data,
        column_config={
            "Channel Name": st.column_config.TextColumn("Channel Name", disabled=True),
            "Saturation Function": st.column_config.SelectboxColumn(
                                    "Saturation Function",
                                    options=["Power", "Log"],
                                    help="Select the saturation function for the channel"),
            "Power (k)": st.column_config.NumberColumn("Power (k)", min_value=0.0, step=0.1, max_value=1),
            "Lags": st.column_config.NumberColumn("Lags", min_value=0, step=1, max_value=12),
            "Adstock": st.column_config.NumberColumn("Adstock", min_value=0.0, step=0.1, max_value=1)
        },
        hide_index=True,
        key="editable_table"
    )

        # Programmatically override Lags and Adstock where needed
    for i, row in edited_df.iterrows():
        if row["Channel Name"] == dependent_variable:
            edited_df.at[i, "Lags"] = None
            edited_df.at[i, "Adstock"] = None
        elif row["Channel Name"] == f"Lagged {dependent_variable}":
            edited_df.at[i, "Adstock"] = None
    
    # Display helpful info
    if dependent_variable in channel_names:
        st.info(f"Lag and Adstock values for '{dependent_variable}' will be ignored.")
    if f"Lagged {dependent_variable}" in channel_names:
        st.info(f"Adstock value for 'Lagged {dependent_variable}' will be ignored.")


    # Post-processing logic
    for i, row in edited_df.iterrows():
        if row["Saturation Function"] in ["Log" , None]:
            edited_df.at[i, "Power (k)"] = None  # or some ignored flag like np.nan

    # Optionally display a note
    if edited_df["Saturation Function"].isin(["Log" , None]).any():
        st.warning("Note: For channels using the 'Log' saturation function, 'Power (k)' will be ignored.")



    if st.button("Process Data"):

        # # Append both summary and coefficients
        st.session_state['configuration_list'].append(edited_df)

        st.write("Final Inputs Received:")
        st.write(edited_df)

        # Apply transformations to df (actual spend data), grouped by geo_column
        column_list = [col for col in df.columns if col not in [geo_column, date_column]]
        transformed_df = transform_edited_df(df, edited_df, geo_column, dependent_variable)
        # to reorder the columns:
        transformed_df = transformed_df[[col for col in transformed_df.columns if col not in column_list] + column_list]

        # Display transformed DataFrame
        if transformed_df is not None:
            st.success("Data has been transformed")
            st.subheader("Transformed Data")

            # Apply formatting only to numeric columns
            transform_df_display = transformed_df.copy()
            transform_df_display = transform_df_display.rename(columns={'Lagged Sales': 'Carryover'})
            transform_df_display = transform_df_display.rename(columns={'Lagged Sales ': 'Carryover'})
            format_dict = {col: "{:,.2f}" for col in transformed_df.select_dtypes(include='number').columns}
            styled_df = transformed_df.head(50).style.format(format_dict)
            st.write(styled_df)
            #st.dataframe(transformed_df)

            return transformed_df

# Run the Streamlit app
if __name__ == '__main__':

    st.title("Transform your Data")

    if 'configuration_list' not in st.session_state:
        st.session_state['configuration_list'] = []    #Store regression outputs as a list in session state to preserve multiple versions


    if ('date_column' in st.session_state 
        and 'geo_column' in st.session_state 
        and 'granular_df' in st.session_state 
        and 'dependent_variable' in st.session_state):

        # Loading the session state variables
        date_column = st.session_state['date_column']
        geo_column = st.session_state['geo_column']
        df_raw = st.session_state['granular_df'].copy()
        df = st.session_state['granular_df'].copy()
        dependent_variable = st.session_state['dependent_variable']
        #granularity_level_user_input = st.session_state['granularity_level_user_input']

        st.subheader("Original Granular Data")
        #st.dataframe(df_raw)

        # Apply formatting only to numeric columns
        format_dict = {col: "{:,.2f}" for col in df_raw.select_dtypes(include='number').columns}
        styled_df = df_raw.head(50).style.format(format_dict)
        st.write(styled_df)

        

        option = st.selectbox(f"Do you want to have a lagged version of the dependent variable - ({dependent_variable})?", [None ,'Yes', 'No'])
        if option == 'Yes':
            column_list = [col for col in df.columns if col != dependent_variable]
            df[f'Lagged {dependent_variable}'] = df[dependent_variable]
            df = df[[dependent_variable, f"Lagged {dependent_variable}"] + column_list]
            st.success(f"Lagged {dependent_variable} variable is created. Please specify number of lags in the data frame below.")
        elif option == "No":
            st.warning(f"No lagged version of dependent variable {dependent_variable} will be used.")
        else:
            pass
        
        if option is not None:
            st.subheader("Please specify data transformations below")
            transformed_df = user_input(date_column, geo_column, df)
            st.session_state["transformed_df"] = transformed_df

        # Show transformed_df if it already exists in session_state
        # if "transformed_df" in st.session_state:
        #     st.subheader("Previously Transformed Data")
        #     df_to_show = st.session_state["transformed_df"]
        #     format_dict_transformed = {col: "{:,.2f}" for col in df_to_show.select_dtypes(include='number').columns}
        #     styled_transformed = df_to_show.head(50).style.format(format_dict_transformed)
        #     st.write(styled_transformed)
    

    else:
        st.warning("No file has been uploaded")