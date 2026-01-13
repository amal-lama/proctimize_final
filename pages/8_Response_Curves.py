import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
import plotly.express as px
import re

st.set_page_config(page_title="ProcTimize", layout="wide")
#st.image("img/response_curves.png")
st.title("Create Response Curves")



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

# Global constants - to change
num_time = 12 #number of months 
num_geo = 2614  # number of bricks [geo_id.nunique()] 
# To fetch from dummy data 

#Styled tab;es
#pd.set_option("styler.render.max_elements", 2_000_000)

# Set page config
# st.set_page_config(page_title="Response Curves", layout="centered")
# st.title("Response Curve Viewer")

# ----------------------------------------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------------------------------------

def calc_calibration_factor(impactable_sales_nation, beta_coeff, spend_nation, saturation_function, power_value):   #to fetch choice of function  
    
    if saturation_function == 'log':
        calib_factor = (impactable_sales_nation / (num_time * num_geo)) / (
        beta_coeff * np.log(1 + (spend_nation / (num_time * num_geo))))   

    elif saturation_function == 'power':
        calib_factor = (impactable_sales_nation / (num_time * num_geo)) / (
        beta_coeff * np.power(spend_nation / (num_time * num_geo), power_value))  

    return calib_factor


def create_response_curve(channel_name, impactable_sales_nation, beta_coeff, spend_nation, start, stop, step, price, saturation_function, power_value):
    calibration_factor = calc_calibration_factor(impactable_sales_nation, beta_coeff, spend_nation, saturation_function, power_value)
    spend_values = range(start, stop + 1, step)

    response_df = pd.DataFrame({
        'spend': spend_values,
        'impactable_geo_time': [None] * len(spend_values),
        'impactable_nation': [None] * len(spend_values),
        'impactable_nation_currency': [None] * len(spend_values),
        'roi': [None] * len(spend_values),
        'mroi': [None] * len(spend_values)
    })

    if saturation_function == 'log':
        response_df['impactable_geo_time'] = calibration_factor * beta_coeff * np.log(1 + (response_df['spend'] / (num_time * num_geo)))

    elif saturation_function == 'power':
        response_df['impactable_geo_time'] = calibration_factor * beta_coeff * np.power((response_df['spend'] / (num_time * num_geo)),power_value)


    response_df['impactable_nation'] = num_time * num_geo * response_df['impactable_geo_time']
    response_df['impactable_nation_currency'] = price * response_df['impactable_nation']
    response_df['roi'] = response_df['impactable_nation_currency'] / response_df['spend']
    response_df['mroi'] = (response_df['impactable_nation_currency'].shift(-1) -
                           response_df['impactable_nation_currency']) / (response_df['spend'].shift(-1) -
                                                                          response_df['spend'])

    channel_prefix = channel_name + '_'
    response_df = response_df.add_prefix(channel_prefix)

    return response_df


def create_final_merged_response_curve(model_result_df, start, stop, step, price):
    final_merged_response_curve = pd.DataFrame()

    for channel_name in model_result_df['channel']:
        impactable_sales_nation = float(model_result_df[model_result_df['channel'] == channel_name]['impactable_sensors'])
        beta_coeff = float(model_result_df[model_result_df['channel'] == channel_name]['coefficient'])
        spend_nation = float(model_result_df[model_result_df['channel'] == channel_name]['spend'])
        saturation_function = model_result_df[model_result_df['channel'] == channel_name]['saturation'].values[0]
        power_value = model_result_df[model_result_df['channel'] == channel_name]['power'].values[0]

        response_curve = create_response_curve(channel_name, impactable_sales_nation, beta_coeff, spend_nation,
                                               start, stop, step, price,saturation_function,power_value)

        if final_merged_response_curve.empty:
            final_merged_response_curve = response_curve
        else:
            final_spend_col = final_merged_response_curve.columns[0]
            new_spend_col = response_curve.columns[0]
            final_merged_response_curve = final_merged_response_curve.merge(response_curve,
                                                                            left_on=final_spend_col,
                                                                            right_on=new_spend_col,
                                                                            how='inner')

    return final_merged_response_curve


# ----------------------------------------------------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------------------------------------------------

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV file (Model Results)", type=["csv"])


# Response curve generation inputs
start = st.number_input("Start Spend Value", value=1000, step=1000)
stop = st.number_input("Stop Spend Value", value=40000000, step=1000000)
price = st.number_input("Unit Price of Product", value=49.6, format="%.2f")
step = 1000

if uploaded_file:
    try:
        model_result_df = pd.read_csv(uploaded_file)

        if {'channel', 'impactable_sensors', 'coefficient', 'spend'}.issubset(model_result_df.columns):
            merged_rc = create_final_merged_response_curve(model_result_df, start, stop, step, price)

            # Storing session state variables
            st.session_state['model_result_df'] = model_result_df
            st.session_state["merged_rc"] = merged_rc

            st.success("✅ Response curves generated successfully!")
            st.subheader("Merged Response Curve Data")
            
            format_dict = {col: "{:,.2f}" for col in merged_rc.select_dtypes(include='number').columns}
            styled_df = merged_rc.head(1000).style.format(format_dict)
            st.write(styled_df)

            #st.dataframe(merged_rc)

            # ---- Interactive Viewer ----
            st.markdown("---")
            st.subheader("Plot Individual Channel Curves")

            # Extract channel prefixes
            prefix_pattern = re.compile(r"^((hcp|dtc)_[a-z]+)_")
            prefixes = sorted({match.group(1) for col in merged_rc.columns if (match := prefix_pattern.match(col))})

            if not prefixes:
                st.error("No valid channel prefixes found in column names.")
            else:
                selected_prefix = st.selectbox("Select a Channel to Plot", prefixes)

                x_col = f"{selected_prefix}_spend"
                y_col = f"{selected_prefix}_impactable_nation_currency"
                mroi_col = f"{selected_prefix}_mroi"

                required_cols = [x_col, y_col, mroi_col]
                if not all(col in merged_rc.columns for col in required_cols):
                    st.warning(f"Required columns not found: {', '.join(col for col in required_cols if col not in merged_rc.columns)}")
                else:
                    # Find the spend value where MROI is closest to 1
                    mroi_df = merged_rc[[x_col, mroi_col]].dropna()
                    mroi_df['abs_diff'] = (mroi_df[mroi_col] - 1).abs()
                    closest_row = mroi_df.loc[mroi_df['abs_diff'].idxmin()]
                    mroi1_spend = closest_row[x_col]

                    # Create the plot
                    fig = px.scatter(
                        merged_rc, x=x_col, y=y_col,
                        title=f"Response Curve for {selected_prefix.replace('_', ' ').title()}",
                        labels={x_col: "Spend", y_col: "Impactable Sales"},
                        hover_data={mroi_col: True} 
                    )

                    # Add vertical line and annotation for MROI = 1
                    fig.add_vline(
                        x=mroi1_spend,
                        line_dash="dash", line_color="red",
                        annotation_text=f"MROI = 1\nSpend = {mroi1_spend:,.0f}",
                        annotation_position="top",
                        annotation_font_size=12
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📄 Filtered Channel Data")
                    
                    filtered_df = merged_rc[[x_col, y_col, mroi_col]]
                    format_dict_filtered_df = {col: "{:,.2f}" for col in filtered_df.select_dtypes(include='number').columns}
                    styled_filtered_df = filtered_df.head(1000).style.format(format_dict_filtered_df)
                    st.write(styled_filtered_df)
                    #st.dataframe(merged_rc[[x_col, y_col, mroi_col]])



            # ---- Multi-Channel Comparison ----
            st.markdown("---")
            st.subheader("Compare Multiple Channel Curves")

            multi_selected_prefixes = st.multiselect("Select Channels to Compare", prefixes)

            if multi_selected_prefixes:
                # Fix the y-metric to "impactable_nation_currency"
                selected_y_metric = "impactable_nation_currency"

                fig_multi = px.line()

                for prefix in multi_selected_prefixes:
                    spend_col = prefix + "_spend"
                    y_col = prefix + "_" + selected_y_metric
                    mroi_col = prefix + "_mroi"

                    if spend_col in merged_rc.columns and y_col in merged_rc.columns and mroi_col in merged_rc.columns:
                        # Add the scatter plot for the channel
                        fig_multi.add_scatter(
                            x=merged_rc[spend_col],
                            y=merged_rc[y_col],
                            mode="lines+markers",
                            name=prefix.replace("_", " ").title()
                        )
                        
                        # Find the row with mroi closest to 1
                        mroi_df = merged_rc[[spend_col, mroi_col]].dropna()
                        mroi_df['abs_diff'] = (mroi_df[mroi_col] - 1).abs()
                        closest_row = mroi_df.loc[mroi_df['abs_diff'].idxmin()]
                        mroi_1_spend = closest_row[spend_col]

                        # Add vertical line at spend closest to where mroi is 1
                        fig_multi.add_vline(
                            x=mroi_1_spend,
                            line=dict(color="red", dash="dash"),
                            name=f"{prefix} mroi ≈ 1"
                        )

                        # Separate out the spend values on top of the vertical lines and display the corresponding spend
                        offset = 0  # Initialize offset for separation
                        while mroi_1_spend in [ann['x'] for ann in fig_multi['layout']['annotations']]:
                            offset += 10  # Increase separation if the value already exists
                        fig_multi.add_annotation(
                            x=mroi_1_spend,
                            y=max(merged_rc[y_col]) + offset,  # Apply the offset to separate annotations
                            text=f"{prefix.replace('_', ' ').title()} Spend: {mroi_1_spend:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-40,
                            font=dict(size=12, color="black")
                        )

                fig_multi.update_layout(
                    #title=f"Comparison of Channels on {selected_y_metric.replace('_', ' ').title()}",
                    xaxis_title="Spend",  # Change x-axis label
                    yaxis_title="Impactable Sales",  # Change y-axis label
                    legend_title="Channel"
                )

                st.plotly_chart(fig_multi, use_container_width=True)


        else:
            st.error("Uploaded CSV does not contain required columns: 'channel', 'impactable_sensors', 'coefficient', 'spend'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
