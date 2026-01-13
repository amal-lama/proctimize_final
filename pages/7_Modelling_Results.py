import streamlit as st
st.set_page_config(page_title="ProcTimize", layout="wide")
#st.image("img/results.png")
st.title("Display Model Results")


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

# Page for displaying and selecting regression results
def display_regression_results():
    if 'regression_outputs' in st.session_state and st.session_state['regression_outputs']:
        st.subheader("Previous Regression Results")

        # Dropdown to select model run
        model_labels = [f"Iteration {i+1}" for i in range(len(st.session_state['regression_outputs']))]
        selected_run_label = st.selectbox("Select the iteration to view", model_labels)

        # Get index of selected model
        selected_index = model_labels.index(selected_run_label)
        selected_result = st.session_state['regression_outputs'][selected_index]

        # Display selected model details
        st.subheader("Selected Model: " + selected_run_label)

        # Display modeling time period from the selected result
        start_date = selected_result.get('start_date', None)
        end_date = selected_result.get('end_date', None)
        if start_date and end_date:
            st.info(f"Modeling Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Display configuration table
        # st.subheader('Configuration Table')
        # configuration_list = st.session_state['configuration_list']
        # st.dataframe(configuration_list[selected_index])

        st.subheader("OLS Regression Summary")
        st.code(selected_result['summary'])

        st.subheader("Coefficients Table")
        coefficients = selected_result['coefficients']
        format_dict = {col: "{:,.2f}" for col in coefficients.select_dtypes(include='number').columns}
        styled_df = coefficients.drop(columns=['Impactable %', 'Note'], axis=1).style.format(format_dict)
        st.write(styled_df)

        #st.dataframe(selected_result['coefficients'])

        # Save only the coefficients DataFrame
        if st.button("Save this model's coefficients for response curve generation"):
            st.session_state['selected_model_coefficients'] = selected_result['coefficients'].copy()
            st.success("Only coefficients DataFrame saved!")

    else:
        st.warning("No regression results found. Please run a regression first.")

# Run the Streamlit app
if __name__ == '__main__':
    display_regression_results()

# Please Do not Delete:
# import streamlit as st

# # Page for displaying and selecting regression results
# def display_regression_results():
#     if 'regression_outputs' in st.session_state and st.session_state['regression_outputs']:
#         st.subheader("Previous Regression Results")

#         # Dropdown to select model run
#         model_labels = [f"Iteration {i+1}" for i in range(len(st.session_state['regression_outputs']))]
#         selected_run_label = st.selectbox("Select the best model iteration", model_labels)

#         # Get index of selected model
#         selected_index = model_labels.index(selected_run_label)
#         selected_result = st.session_state['regression_outputs'][selected_index]

#         # Display selected model details
#         st.subheader("Selected Model: " + selected_run_label)

#         # Display modeling time period from the selected result
#         start_date = selected_result.get('start_date', None)
#         end_date = selected_result.get('end_date', None)
#         if start_date and end_date:
#             st.info(f"Modeling Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

#         # Display configuration table
#         st.subheader('Configuration Table')
#         configuration_list = st.session_state['configuration_list']
#         st.dataframe(configuration_list[selected_index])

#         st.subheader("OLS Regression Summary")
#         st.code(selected_result['summary'],language='text')

#         st.subheader("Coefficients Table")
#         selected_result_view = selected_result['coefficients'].drop(columns=['Impactable %']) 
#         st.dataframe(selected_result_view)           

#         # Option to save coefficients for response curve generation
#         if st.button("Save this model's configurations for response curve generation"):
#             st.session_state['selected_model_coefficients'] = selected_result['coefficients']
#             st.success("Configurations saved for response curve generation!")

#     else:
#         st.warning("No regression results found. Please run a regression first.")

# # Run the Streamlit app
# if __name__ == '__main__':
#     st.title("Model Output")
#     display_regression_results()