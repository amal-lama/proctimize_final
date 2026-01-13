import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import plotly.express as px
import warnings

st.set_page_config(page_title="ProcTimize", layout="wide")
# st.image("img/long_logo.png")
# st.markdown(
#     """
#     <div style='text-align: left; color: black; font-weight: 500;'>
#         The Marketing Mix Tool helps you measure and optimize the impact of your marketing efforts. 
#         Upload your data, apply custom transformations, and build models to understand what’s driving performance.
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown(
#     "<hr style='border: 0; height: 1px; background-color: lightgrey;' />",
#     unsafe_allow_html=True
# )

# st.markdown("""
# ### **Key Features**
            
# - Upload Marketing and Sales Data Sources
# - Apply Standard Modeling Transformations: Lag, Adstock, and Saturation Effects
# - Generate Data Review and Validation Summaries
# - Perform Exploratory Data Analysis (EDA)
# - Build and Evaluate Marketing Mix Models
# - Analyze Media Spend vs. KPI Relationships Using Response Curves
# - Simulate Scenarios and Forecast Outcomes for Budget Optimization
# """)

# st.image("img/workflow.png")

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

def eda_sales_trend(uploaded_file):

    if uploaded_file is not None:
        # Read CSV File
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded CSV")
        st.dataframe(df.head(100))

        # Let user select columns
        st.subheader("Select Columns")
        dependent_variable = st.selectbox("Select Dependent Variable", [None] + list(df.columns), index=0 )
        #geo_column = st.selectbox("Select Modeling Granularity Column", [None] + list(df.columns), index=0)
        #date_column = st.selectbox("Select Date Column", [None] + list(df.columns), index=0)

        if ("geo_column" in st.session_state 
            and "date_column" in st.session_state 
            and "ZIP_column" in st.session_state
            and "DMA_column" in st.session_state):

            geo_column = st.session_state["geo_column"]
            date_column = st.session_state["date_column"]
            zip_column = st.session_state["ZIP_column"]
            dma_column = st.session_state["DMA_column"]

        if "geo_column" not in st.session_state or "date_column" not in st.session_state:
            st.warning("Modelling Granularity or date column not detected. Please select manually.")
            geo_column = st.selectbox("Select Modeling Granularity Column", [None] + list(df.columns), index=0)
            date_column = st.selectbox("Select Date Column", [None] + list(df.columns), index=0)
            
        if geo_column is not None:
            num_geo = df[geo_column].nunique()
            st.success(f"Number of unique entries in granularity column: {num_geo}")
        
        # Time granularity detection
        if date_column is not None:
            #time_granularity = detect_date_granularity(df, date_column )
            #st.write(f' The time granularity is : {time_granularity} ')
        
            # Convert to datetime format
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.dropna(subset=[date_column])

            df_copy = df   # Original dataframe
           
            # Set up date range filtering
            min_date = df[date_column].min()
            max_date = df[date_column].max()

            # Convert to datetime.date if needed
            min_date = pd.to_datetime(min_date).date()
            max_date = pd.to_datetime(max_date).date()

            # Side-by-side date selectors
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="end_date"
                )

            # Validate selected dates
            if start_date > end_date:
                st.warning("⚠️ Start date must be before or equal to end date.")
                st.stop()

            # Filter the DataFrame based on selected range
            df = df[(df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))]

            # Handle case when only one date or empty data
            if df.empty:
                st.warning("No data available for the selected date range. Please choose a broader range.")
                st.stop()

            if (pd.to_datetime(start_date) == pd.to_datetime(end_date)):
                st.warning("Start and end dates are the same. Please select a wider date range for meaningful analysis.")
                st.stop()

            # Control totals    
            st.subheader("Control Totals")

            remove_cols = [date_column, geo_column, zip_column, dma_column]
            filtered_df = df.drop(columns=remove_cols)
            filtered_df = filtered_df.select_dtypes(include='number')
            
            subtotal_df = pd.DataFrame({
                'Channel': filtered_df.columns,
                'Total': filtered_df.sum()
            })

            #styled_df = subtotal_df.style.format({"Total": "{:,}"})
            #st.dataframe(styled_df)

            format_dict = {col: "{:,.2f}" for col in subtotal_df.select_dtypes(include='number').columns}
            styled_df = subtotal_df.reset_index(drop = True).style.format(format_dict)
            st.write(styled_df)


            # Correlation Matrix
            st.subheader("Correlation Matrix")
            # corr_matrix = filtered_df.corr()
            # fig, ax = plt.subplots(figsize=(2, 1.75), dpi=200)
            # fig = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.5}).get_figure()
            # # changing font size of axis values
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=3)
            # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=3)
            # # 4. Shrink colorbar tick font (access last axis)
            # plt.tight_layout()
            # cbar = ax.collections[0].colorbar
            # cbar.ax.tick_params(labelsize=3)

            # Compute correlation matrix
            corr_matrix = filtered_df.corr()

            # Create Plotly heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
            )

            fig.update_layout(
                width=600,
                height=600,
                xaxis_title="Features",
                yaxis_title="Features"
            )

            st.plotly_chart(fig, use_container_width=True)

            #st.pyplot(fig)

            # Metric to visualize
            st.subheader("Trend visualization")

            remove_cols = [date_column, geo_column, zip_column, dma_column]
            numeric_cols = df.drop(columns=remove_cols).select_dtypes(include='number').columns.tolist()
            numeric_cols = [item for item in numeric_cols]

            if not numeric_cols:
                st.warning("No numeric columns available for visualization.")
                return

            visualize_columns = st.multiselect("Select Data to visualize", numeric_cols)

            if not visualize_columns:
                pass
                return

            aggregation_level = st.selectbox("Select Aggregation Level", ["Weekly", "Monthly"])

            if date_column not in df.columns or geo_column not in df.columns:
                st.warning("Date or Geo column is not properly selected.")
                return

            try:
                if aggregation_level == "Weekly":
                    visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='W'))[visualize_columns].sum().reset_index()
                else:
                    visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='M'))[visualize_columns].sum().reset_index()

                st.subheader(f"Trend of {', '.join(visualize_columns)} ({aggregation_level})")

                trend_long = visualize_trend_time.melt(id_vars=date_column, value_vars=visualize_columns, 
                                                    var_name="Metric", value_name="Value")

                fig = px.line(
                    trend_long,
                    x=date_column,
                    y="Value",
                    color="Metric",
                    markers=True,
                    title=f"{' & '.join(visualize_columns)} Trend ({aggregation_level})",
                    labels={date_column: "Time", "Value": "Value", "Metric": "Metric"}
                )

                fig.update_layout(
                    xaxis_tickangle=45,
                    xaxis=dict(tickformat="%b-%Y") if aggregation_level == "Monthly" else {}
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while plotting: {e}")

            return geo_column, date_column, dependent_variable

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    st.title("Exploratory Data Analysis")

    if "joined_output_df" in st.session_state and not st.session_state["joined_output_df"].empty:
        st.markdown("### ✅ Using previously joined dataset")

        # Convert Polars DataFrame to Pandas
        joined_df = st.session_state["joined_output_df"]

        # Call the EDA function with DataFrame wrapped as file-like
        from io import StringIO
        csv_buffer = StringIO()
        joined_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        result = eda_sales_trend(csv_buffer)  # Simulate uploaded_file-like object

        if result:
            geo_column, date_column, dependent_variable = result  
            st.session_state["geo_column"] = geo_column
            st.session_state["date_column"] = date_column
            st.session_state["dependent_variable"] = dependent_variable
            st.session_state["granular_df"] = joined_df

    else:
        st.warning("⚠️ No joined dataset found. Please complete the file join step first.")
