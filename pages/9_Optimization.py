import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ProcTimize", layout="wide")
#st.image("img/optimization.png")

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


# All functions

# Function to sum up kpi's
def calc_rc_kpi(merged_rc,optimizer_dict,kpi,k):
    total_kpi = 0

    for channel in list(optimizer_dict.keys()):
        iterator = optimizer_dict[channel]['iter']
        iterator -= k

        if iterator>=0:
            total_kpi += merged_rc[f'{channel}_{kpi}'][iterator]
    return total_kpi

# Function to calculate total iterations
def total_iter(optimizer_dict):
    total = 0
    for channel in optimizer_dict.keys():
        total += optimizer_dict[channel]['iter']
    return (total - 11)


# Function to run the optimizer
def run_optimizer(optimizer_dict,merged_rc,target,opt_type,k):


    if (opt_type=='Budget Goal'):
        criteria = 'spend'
    
    elif opt_type == 'Sales Goal' :
        criteria = 'impactable_nation'

    # Ensuring min criteria
    current_val = calc_rc_kpi(merged_rc,optimizer_dict,criteria,k)

    for channel in list(optimizer_dict.keys()):

        iterator = optimizer_dict[channel]['iter']
        while(merged_rc[f'{channel}_spend'][iterator] <= optimizer_dict[channel]['min'] ):

            # Debugging step
            # print(channel,"----",iterator,'----',calc_rc_kpi(merged_rc,optimizer_dict,criteria,k), '----',total_iter(optimizer_dict))
            optimizer_dict[channel]['iter'] += k
            iterator = optimizer_dict[channel]['iter']


    # Running for Fixed Budget and Sensor Goal
    current_val = calc_rc_kpi(merged_rc,optimizer_dict,criteria,k)
    max_mroi_channel=None
    
    while current_val < target:
        
        max_mroi =-1
        max_mroi_channel = None

        for channel in list(optimizer_dict.keys()):
            
            iterator = optimizer_dict[channel]['iter']
            if iterator < len(merged_rc)  :
                #Checking the max limit
                if merged_rc[f'{channel}_spend'][iterator] <= optimizer_dict[channel]['max'] :

                    if merged_rc[f'{channel}_mroi'][iterator] > max_mroi:
                        max_mroi =  merged_rc[f'{channel}_mroi'][iterator]
                        max_mroi_channel = channel

        # Debugging step
        # print(max_mroi_channel,'----',optimizer_dict[max_mroi_channel]['iter'],'----',current_val, '----',total_iter(optimizer_dict))
        
        optimizer_dict[max_mroi_channel]['iter'] += k

        
        current_val  = calc_rc_kpi(merged_rc,optimizer_dict,criteria,k)


        if(current_val>target) and opt_type=='Budget Goal':
            optimizer_dict[max_mroi_channel]['iter'] -= k
            break


    current_val  = calc_rc_kpi(merged_rc,optimizer_dict,criteria,k)

    return optimizer_dict


def optimizer_result(optimizer_dict,model_result_df,merged_rc,k):

    channel_list = list(optimizer_dict.keys())

    optimizer_result_df = pd.DataFrame({
        'channel' : channel_list,
        'optimal_spend' : [None] * len(channel_list),
        'sensor_volume' : [None] * len(channel_list),
        'net_sales' : [None] * len(channel_list),
        'roi' : [None] * len(channel_list),
        'mroi' : [None] * len(channel_list)
    })

    # Calculation of all columns
    for channel in channel_list:

        iterator = optimizer_dict[channel]['iter'] - k

        if iterator < 0:
            iterator = 0

        optimizer_result_df.loc[optimizer_result_df['channel']==channel,'optimal_spend'] = merged_rc[f'{channel}_spend'][iterator]
        optimizer_result_df.loc[optimizer_result_df['channel']==channel,'sensor_volume'] = merged_rc[f'{channel}_impactable_nation'][iterator]
        optimizer_result_df.loc[optimizer_result_df['channel']==channel,'net_sales'] = merged_rc[f'{channel}_impactable_nation_currency'][iterator]
        optimizer_result_df.loc[optimizer_result_df['channel']==channel,'roi'] = merged_rc[f'{channel}_roi'][iterator]
        optimizer_result_df.loc[optimizer_result_df['channel']==channel,'mroi'] = merged_rc[f'{channel}_mroi'][iterator]


    optimizer_result_df = pd.merge(model_result_df,optimizer_result_df,on='channel',how='inner')

    optimizer_result_df = optimizer_result_df.drop(columns=['impactable%','impactable_sensors','coefficient','saturation','power'])
    # Converting from object to float type
    optimizer_result_df = optimizer_result_df.apply(pd.to_numeric, errors='ignore')


    # Adding total row
    total_row = optimizer_result_df.select_dtypes(include='number').sum(numeric_only=True)
    total_row['roi']  = total_row['net_sales']/total_row['optimal_spend']
    total_row['mroi'] = optimizer_result_df['mroi'].min()
    total_row['channel'] = 'Total'
    # Append the total row to the DataFrame
    optimizer_result_df = pd.concat([optimizer_result_df, pd.DataFrame([total_row])], ignore_index=True)
        
    return optimizer_result_df

def plot_delta_spend(optimizer_result_df):

    #Dropping total
    optimizer_result_df = optimizer_result_df.loc[optimizer_result_df['channel']!='Total'].reset_index()

    channel_names = list(optimizer_result_df['channel'])
    current_spend = list(optimizer_result_df['spend'])
    optimal_spend = list(optimizer_result_df['optimal_spend'])

    # Calculate delta
    delta_spend = np.array(optimal_spend) - np.array(current_spend)

    # Format for M labels
    def format_m(value):
        return f"{value/1e6:.0f}M" if abs(value) >= 1e6 else f"{value/1e6:.1f}M"

    # Color assignment: red for negative, blue for positive
    colors = ['#FF7F7F' if val < 0 else '#76D7EA' for val in delta_spend]

    # Create Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=channel_names,
        y=delta_spend,
        marker_color=colors,
        text=[format_m(val) for val in delta_spend],
        textposition='outside'
    ))

    # Update layout
    fig.update_layout(
        title="Change in optimized spend for each channel",
        yaxis_title="$",
        xaxis_tickangle=-45,
        showlegend=False,
        height=500,
        margin=dict(t=60, l=40, r=40, b=60),
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_optimal_spend_pie(optimizer_result_df):
    # Drop 'Total' row
    optimizer_result_df = optimizer_result_df[optimizer_result_df['channel'] != 'Total'].reset_index(drop=True)

    labels = optimizer_result_df['channel']
    values = optimizer_result_df['optimal_spend']

    # Normalize values for shading (between 0 and 1)
    percentages = values / values.sum()
    normalized = (percentages - percentages.min()) / (percentages.max() - percentages.min() + 1e-9)**2

    # Use base color #146C94 (RGB: 20, 108, 148) and vary opacity
    base_rgb = (0, 30, 150)
    colors = [f'rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {0.3 + 0.7 * val:.2f})' for val in normalized]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        marker=dict(colors=colors,
                    line=dict(color='black', width=1)),
        hole=0.3
    )])

    fig.update_layout(
        title_text='Optimal Spend Allocation by Channel',
        height=500,
        margin=dict(t=60, l=40, r=40, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

def dict_sum(optimizer_dict, value):

    sum =0 

    for channel in optimizer_dict.keys():
        sum += optimizer_dict[channel][value]

    return sum

def min_max_check(optimizer_dict):

    for channel in optimizer_dict.keys():
        if optimizer_dict[channel]['min'] >= optimizer_dict[channel]['max']:
            return False
        
    return True


# End of functions -----------------------------------------------------------------------------------------------------------------

# Run the Streamlit app
if __name__ == '__main__':

    st.title("Optimization")
    if 'model_result_df' in st.session_state and 'merged_rc' in st.session_state: 
        # Loading the session state variables
        model_result_df = st.session_state['model_result_df']
        merged_rc = st.session_state["merged_rc"]
        


        # Define initial data for user input

        st.subheader("Input Min/Max constraints")


        channel_names = list(model_result_df['channel'])

        # Multi-select dropdown with all options selected by default
        selected_options = st.multiselect("Select Channels:", channel_names, default=channel_names)

        # Filter model_result_df and channel_names based on selection
        model_result_df = model_result_df[model_result_df['channel'].isin(selected_options)].reset_index(drop=True)
        channel_names = list(model_result_df['channel'])

            

        data = pd.DataFrame({
            "Channel Name": channel_names,
            "Min Value": [0] * len(channel_names),
            "Max Value": [10000000] * len(channel_names),
        })
        
        # Choose type of optimization

        options = ["Budget Goal", "Sales Goal"]
        opt_type = st.selectbox("Select the optimization type", options)

        target = st.number_input("Enter the target size", value=12000000, step=1000)
        iter_size = st.number_input("Enter the step size", value=1000, step=1000)

        

        # Streamlit Data Editor
        edited_df = st.data_editor(
            data,
            column_config={
                "Channel Name": st.column_config.TextColumn("Channel Name", disabled=True),
                "Min Value": st.column_config.NumberColumn("Min Value", min_value=0, step=1000),
                "Max Value": st.column_config.NumberColumn("Max Value", min_value=0, step=1000)
            },
            hide_index=True,
            key="editable_table"
        )

        k = iter_size/1000
    
        optimizer_dict = {
        channel : {'iter':k-1, 'min':0, 'max':0}
        for channel in channel_names
        }

        for channel in optimizer_dict.keys():
            optimizer_dict[channel]['min'] = edited_df[edited_df['Channel Name']==channel]['Min Value'].values[0]
            optimizer_dict[channel]['max'] = edited_df[edited_df['Channel Name']==channel]['Max Value'].values[0]


        if st.button("Run Optimizer"):

            if (dict_sum(optimizer_dict,'min') < target ) and min_max_check(optimizer_dict) and target <= 1000000000:

                updated_optimizer_dict = run_optimizer(optimizer_dict,merged_rc,target,opt_type,k)
                # st.json(updated_optimizer_dict)

                st.subheader("Optimization Results")
                optimizer_result_df = optimizer_result(updated_optimizer_dict, model_result_df, merged_rc, k)
                format_dict = {col: "{:,.2f}" for col in optimizer_result_df.select_dtypes(include='number').columns}
                styled_df = optimizer_result_df.style.format(format_dict)

                st.dataframe(styled_df)


                #st.dataframe(optimizer_result_df)

                plot_delta_spend(optimizer_result_df)
                plot_optimal_spend_pie(optimizer_result_df)
            
            else:
                
                st.warning("Ensure the constraints are met")

    else:
        st.warning("Generate Response Curves before running optimizer")











