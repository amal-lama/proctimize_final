import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="ProcTimize", layout="wide")
# st.image("img/modelling.png")

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

def modeling_input(date_column, geo_column, granular_df, transformed_df, dependent_variable, dependent_variable_user_input):
    try:
        transformed_df[date_column] = pd.to_datetime(transformed_df[date_column])
        granular_df[date_column] = pd.to_datetime(granular_df[date_column])
    except Exception as e:
        st.error(f"Error converting date columns: {e}")
        return None

    try:
        max_date = transformed_df[date_column].max().date()
        all_years = sorted(transformed_df[date_column].dt.year.unique())
        if len(all_years) < 2:
            st.error("Not enough years of data to exclude the first year.")
            return None
        adjusted_min_date = pd.to_datetime(f"{all_years[1]}-01-01").date()
    except Exception as e:
        st.error(f"Could not compute min/max dates: {e}")
        return None

    try:
        start_date = st.date_input("Select Start Date (First year excluded from modeling)",
                                   value=adjusted_min_date, min_value=adjusted_min_date, max_value=max_date, key="start_date")
        end_date = st.date_input("Select End Date",
                                 value=max_date, min_value=adjusted_min_date, max_value=max_date, key="end_date")

        modeling_duration_days = (end_date - start_date).days + 1
        prior_end_date = start_date - pd.Timedelta(days=1)
        prior_start_date = prior_end_date - pd.Timedelta(days=modeling_duration_days - 1)

        transformed_df_date_filtered = transformed_df[(transformed_df[date_column] >= pd.to_datetime(start_date)) &
                                                      (transformed_df[date_column] <= pd.to_datetime(end_date))]
        
        granular_df_date_filtered = granular_df[(granular_df[date_column] >= pd.to_datetime(start_date)) &
                                                (granular_df[date_column] <= pd.to_datetime(end_date))]
        
        granular_df_prior_date_filtered = granular_df[(granular_df[date_column] >= pd.to_datetime(prior_start_date)) &
                                                       (granular_df[date_column] <= pd.to_datetime(prior_end_date))]
        if dependent_variable == dependent_variable_user_input:
            try:
                remove_cols = [date_column, geo_column, dependent_variable, f"{dependent_variable}_transformed"]
                available_channels = [
                    col for col in transformed_df_date_filtered.drop(columns=remove_cols).columns
                    if col.endswith('_transformed')]
            except:
                remove_cols = [date_column, geo_column, dependent_variable]
                available_channels = [
                    col for col in transformed_df_date_filtered.drop(columns=remove_cols).columns
                    if col.endswith('_transformed')]
        else:
            remove_cols = [date_column, geo_column, dependent_variable, dependent_variable_user_input]
            available_channels = [
                col for col in transformed_df_date_filtered.drop(columns=remove_cols).columns
                if col.endswith('_transformed')]

        selected_channels = st.multiselect("Select channels to model on", available_channels, key="channel_selector")


        if selected_channels:
            selected_channels_df = pd.DataFrame({'Channel Name': selected_channels})
            selected_channels_df.index = np.arange(1, len(selected_channels_df) + 1)

            transformed_df_channel_filtered = transformed_df_date_filtered[[date_column, geo_column, dependent_variable_user_input] + selected_channels]

            st.session_state['selected_channels_df'] = selected_channels_df
            st.session_state['transformed_df_channel_filtered'] = transformed_df_channel_filtered
            st.session_state['granular_df_date_filtered'] = granular_df_date_filtered
            st.session_state['granular_df_prior_date_filtered'] = granular_df_prior_date_filtered
            st.session_state['selected_channels'] = selected_channels
            st.session_state['selected_start_date'] = start_date
            st.session_state['selected_end_date'] = end_date
            st.session_state['dependent_variable_user_input'] = dependent_variable_user_input

            st.success(f"Filtered data from {start_date} to {end_date}")
            st.dataframe(selected_channels_df)
            st.success(f'Total {dependent_variable_user_input} in selected range: {transformed_df_channel_filtered[dependent_variable_user_input].sum():,.0f}')

            return selected_channels_df

    except Exception as e:
        st.error(f"An error occurred during input processing: {e}")

    return None


def run_regression():
    
    transformed_df_channel_filtered = st.session_state['transformed_df_channel_filtered'].copy()
    selected_channels = st.session_state['selected_channels']
    granular_df_date_filtered = st.session_state['granular_df_date_filtered']
    granular_df_prior_date_filtered = st.session_state['granular_df_prior_date_filtered']

    y = transformed_df_channel_filtered[dependent_variable_user_input]

    X = transformed_df_channel_filtered[selected_channels]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    regression_summary = model.summary().as_text()

    st.subheader("OLS Regression Results")
    st.code(regression_summary, language='text')

    st.subheader("Model Summary Table")
    coefficients = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values
    })

    

    # Use raw sales for impactable % and ROI
    sum_sales = transformed_df_channel_filtered[dependent_variable_user_input].sum()
    sum_raw_sales = granular_df_date_filtered[dependent_variable].sum()
    sum_raw_sales_prior = granular_df_prior_date_filtered[dependent_variable].sum()

    transformed_df_channel_filtered['const'] = 1
    granular_df_date_filtered['const'] = 1

    transformed_to_raw = { col: 'const' if col == 'const' else col.replace('_transformed', '')
                          for col in coefficients['Variable']
                          }
    
    coefficients['Raw Variable'] = coefficients['Variable'].map(transformed_to_raw)

    coefficients['Raw Activity'] = coefficients['Variable'].apply(
        lambda var: granular_df_date_filtered[var.replace('_transformed', '')].sum()
        if var.replace('_transformed', '') in granular_df_date_filtered.columns else 0
    )

    coefficients['Modelled Activity'] = coefficients['Variable'].apply(
        lambda var: transformed_df_channel_filtered[var].sum()
        if var in transformed_df_channel_filtered.columns else 0
    )

    lagged_col = "Carryover"
    no_spend_vars = ['const', lagged_col]

    #st.dataframe(granular_df_date_filtered)
    #st.dataframe(coefficients)
    
    #Changed
    coefficients['Spend'] = coefficients['Raw Variable'].apply(
    lambda var: 0 if var in no_spend_vars else (
        granular_df_date_filtered[var].sum()
        if 'Spend' in var and var in granular_df_date_filtered.columns else (
            granular_df_date_filtered[var + ' Spend'].sum()
            if var + ' Spend' in granular_df_date_filtered.columns else 0
            )
        )
    )

    coefficients['Impactable %'] = coefficients.apply(
        lambda row: (row['Coefficient'] * row['Modelled Activity'] * 100) / sum_sales if y.sum() != 0 else 0,
        axis=1
    )

    coefficients['Impactable (%)'] = coefficients['Impactable %'].map(lambda x: f"{x:.2f}%")
    coefficients['Impactable Sales'] = coefficients['Impactable %'] * sum_raw_sales / 100  #this should be raw sales

    coefficients['ROI'] = coefficients.apply(
        lambda row: row['Impactable Sales'] / row['Spend'] if row['Spend'] != 0 else 0,
        axis=1
    )

    coefficients['Note'] = coefficients['Raw Variable'].apply(
        lambda var: 'Intercept' if var == 'const' else ('Carryover' if var == lagged_col else '')
    )

    carryover_percentage_series = coefficients[coefficients['Note'] == 'Carryover']['Impactable %'] / 100
    if not carryover_percentage_series.empty:
        carryover_percentage = carryover_percentage_series.iloc[0]
        carryover_rate = (carryover_percentage * sum_raw_sales) / sum_raw_sales_prior
        long_term_factor = (3 + 2 * carryover_rate + carryover_rate ** 2) / 3
        st.write("(3-year) Long Term Factor:", f"{long_term_factor:,.2f}")
        coefficients['Long Term ROI'] = long_term_factor * coefficients['ROI']
    else:
        st.warning("No 'Carryover' term found. Long Term ROI cannot be calculated.")
        coefficients['Long Term ROI'] = 0

    coefficients = coefficients.drop(['Raw Variable'], axis=1)

    format_dict = {col: "{:,.2f}" for col in coefficients.select_dtypes(include='number').columns}
    styled_df = coefficients.drop(columns=['Impactable %', 'Note'], axis=1).style.format(format_dict)
    st.write(styled_df)
    #st.dataframe(coefficients.drop(columns=['Impactable %', 'Note'], axis=1))

    if 'regression_outputs' not in st.session_state:
        st.session_state['regression_outputs'] = []

    st.session_state['regression_outputs'].append({
        'summary': regression_summary,
        'coefficients': coefficients,
        'start_date': st.session_state['selected_start_date'],
        'end_date': st.session_state['selected_end_date']
    })

    # except Exception as e:
    #     st.error(f"An error occurred while running the regression: {e}")



# Run the Streamlit app
if __name__ == '__main__':

    st.title("Modelling")


    if all(k in st.session_state for k in ['date_column', 'geo_column', 'granular_df', 'transformed_df', 'dependent_variable']):
        date_column = st.session_state['date_column']
        geo_column = st.session_state['geo_column']
        granular_df = st.session_state['granular_df']
        transformed_df = st.session_state['transformed_df']
        dependent_variable = st.session_state['dependent_variable']
        if "Carryover" in transformed_df.columns:
            raw_lagged_dependent_variable = st.session_state['raw_lagged_dependent_variable']

        st.subheader("Transformed Data")
        st.write(transformed_df.head(50))
       
        if f"{dependent_variable}_transformed" in transformed_df.columns:
            granular_df["Carryover"] = raw_lagged_dependent_variable
            dependent_variable_user_input = st.selectbox("Select Dependent Variable", list((dependent_variable, f"{dependent_variable}_transformed")), index=0)
        else:
            dependent_variable_user_input = dependent_variable
        
        selected_channels_df = modeling_input(date_column, geo_column, granular_df, transformed_df, dependent_variable, dependent_variable_user_input)

        if 'transformed_df_channel_filtered' in st.session_state:
            st.subheader("Filtered DataFrame Preview")

            df = st.session_state['transformed_df_channel_filtered']
            format_dict = {col: "{:,.2f}" for col in df.select_dtypes(include='number').columns}
            styled_df = df.head(50).style.format(format_dict)
            st.write(styled_df)
            #st.dataframe(st.session_state['transformed_df_channel_filtered'])

            if st.button("Run OLS Regression"):
                run_regression()
    else:
        st.warning("No file has been uploaded")
