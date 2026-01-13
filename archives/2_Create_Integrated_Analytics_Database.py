import streamlit as st
import polars as pl
from datetime import timedelta
from PIL import Image

st.set_page_config(page_title="ProcTimize", layout="wide")
st.title("Integrated Analytics Dataset Builder")

# with st.sidebar:
#     logo_image = Image.open("logo.png")  # replace with your actual path
#     st.image(logo_image, width=200)
#     st.markdown("---")  # horizontal line
#     st.markdown("## Navigation")

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

st.subheader("Step 1: Upload CSV Files for Each Channel")

# Accept multiple files for each channel
sales_files = st.file_uploader("Upload Sales Data CSVs", type=["csv"], accept_multiple_files=True, key="sales")
hcp_files = st.file_uploader("Upload HCP Channel CSVs", type=["csv"], accept_multiple_files=True, key="hcp")

# Store uploaded data
uploaded_data = {
    "Sales": {},
    "HCP Channels": {}
}

# Helper to read with Polars and get columns
def get_column_names(file):
    try:
        df = pl.read_csv(file, n_rows=5)  # read just a few rows for column detection
        return df.columns
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return []

# Sales File Pills
if sales_files:
    st.markdown("### Sales Files")
    cols = st.columns(len(sales_files))
    for idx, file in enumerate(sales_files):
        with cols[idx]:
            if st.button(file.name, key=f"sales_{idx}"):
                columns = get_column_names(file)
                st.markdown(f"**Columns in {file.name}:**")
                st.code(columns)

# HCP Channel File Pills
if hcp_files:
    st.markdown("### HCP Channel Files")
    cols = st.columns(len(hcp_files))
    for idx, file in enumerate(hcp_files):
        with cols[idx]:
            if st.button(file.name, key=f"hcp_{idx}"):
                columns = get_column_names(file)
                st.markdown(f"**Columns in {file.name}:**")
                st.code(columns)

all_files = [("Sales", f) for f in sales_files] + [("HCP", f) for f in hcp_files]  #DTC files

st.subheader("Step 2: Select Granularity and Date Columns for Each HCP File")

# Initialize session state
if "column_mappings" not in st.session_state:
    st.session_state["column_mappings"] = {}

hcp_column_mapping = {}
date_column_mapping = {}

col_hcp, col_date = st.columns(2)

with col_hcp:
    st.markdown("### Granularity Columns")
    for source, file in all_files:
        try:
            df_cols = pl.read_csv(file, n_rows=5).columns
            column_options = [""] + [col for col in df_cols if col.strip() != ""]

            st.markdown(f"**{file.name}**")
            default_hcp = st.session_state["column_mappings"].get(file.name, {}).get("hcp", "")
            selected_hcp = st.selectbox(
                "Select HCP Column",
                column_options,
                index=column_options.index(default_hcp) if default_hcp in column_options else 0,
                key=f"hcp_{file.name}"
            )

            if selected_hcp:
                hcp_column_mapping[file.name] = (file, selected_hcp)
                st.session_state["column_mappings"].setdefault(file.name, {})["hcp"] = selected_hcp

        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")

with col_date:
    st.markdown("### Date Columns")
    for source, file in all_files:
        try:
            df_cols = pl.read_csv(file, n_rows=5).columns
            column_options = [""] + [col for col in df_cols if col.strip()  != ""]

            st.markdown(f"**{file.name}**")
            default_date = st.session_state["column_mappings"].get(file.name, {}).get("date", "")
            selected_date = st.selectbox(
                "Select Date Column",
                column_options,
                index=column_options.index(default_date) if default_date in column_options else 0,
                key=f"date_{file.name}"
            )

            if selected_date:
                date_column_mapping[file.name] = (file, selected_date)
                st.session_state["column_mappings"].setdefault(file.name, {})["date"] = selected_date

        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")

#Same for DTC Separately



st.subheader("Step 3: Generate HCP-Date Base")

st.subheader("Step 3A: Choose Date Granularity")

# Step 3A: Ensure we get granularity from previous page or fallback to manual input
if "user_input_time_granularity" in st.session_state and st.session_state["user_input_time_granularity"]:
    st.session_state["granularity"] = st.session_state["user_input_time_granularity"]
    st.success(f"✅ Using granularity from previous page: **{st.session_state['granularity']}**")

if "granularity" not in st.session_state or not st.session_state["granularity"]:
    granularity = st.radio("Select Date Granularity", ["Weekly", "Monthly"])
    st.session_state["granularity"] = granularity
    st.info("📌 Granularity set manually.")

# Now safe to use it
granularity = st.session_state["granularity"]

# Step 3B: Build unique HCP list
unique_hcps = pl.DataFrame()
for file_obj, hcp_col in hcp_column_mapping.values():
    try:
        df = pl.read_csv(file_obj, columns=[hcp_col])
        df = df.rename({hcp_col: "HCP_ID"}).drop_nulls()
        unique_hcps = unique_hcps.vstack(df)
    except Exception as e:
        st.error(f"❌ Error reading HCPs from {file_obj.name}: {e}")

unique_hcps = unique_hcps.unique()

# Step 3C: Find min and max date across all selected files
all_dates = []
for file_obj, date_col in date_column_mapping.values():
    if not date_col:
        continue
    try:
        df = pl.read_csv(file_obj, columns=[date_col])
        df = df.rename({date_col: "Date"}).drop_nulls()
        df = df.with_columns(pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
        all_dates.append(df)
    except Exception as e:
        st.error(f"❌ Error reading dates from {file_obj.name}: {e}")

if all_dates:
    all_dates_df = pl.concat(all_dates)
    min_date = all_dates_df.select(pl.col("Date").min())[0, 0]
    max_date = all_dates_df.select(pl.col("Date").max())[0, 0]

    # Step 3D: Generate date range
    date_range = []
    granularity = granularity.lower()

    if granularity == "weekly":
        current = min_date
        while current <= max_date:
            date_range.append(current)
            current += timedelta(weeks=1)

    elif granularity == "monthly":
        current = min_date.replace(day=1)
        while current <= max_date:
            date_range.append(current)
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    dates_df = pl.DataFrame({"Date": date_range})

    # Step 3E: Cartesian Product of HCPs × Dates
    cartesian_df = unique_hcps.join(dates_df, how="cross")

    st.success(f"✅ Generated {cartesian_df.shape[0]} rows: {unique_hcps.shape[0]} HCPs × {dates_df.shape[0]} dates")
    st.dataframe(cartesian_df.head(50).to_pandas())
else:
    st.warning("⚠️ No valid date columns found to determine date range.")

st.subheader("Step 4: Configure File Joins")

file_join_configs = {}  # Will hold configs per file

for idx, (source, file) in enumerate(all_files):
    try:
        df_cols = pl.read_csv(file, n_rows=5).columns

        # Get pre-selected HCP/Date columns from Step 2
        default_hcp = st.session_state["column_mappings"].get(file.name, {}).get("hcp", df_cols[0])
        default_date = st.session_state["column_mappings"].get(file.name, {}).get("date", df_cols[1] if len(df_cols) > 1 else df_cols[0])

        with st.expander(f"📄 Configure Join for: `{file.name}`", expanded=False):
            st.markdown(f"**Source Type:** `{source}`")

            # Join Column Mapping (pre-filled from Step 2)
            join_hcp_col = st.selectbox(
                "Select column to join on HCP ID",
                df_cols,
                index=df_cols.index(default_hcp) if default_hcp in df_cols else 0,
                key=f"join_hcp_{file.name}_{idx}"
            )

            join_date_col = st.selectbox(
                "Select column to join on Date",
                df_cols,
                index=df_cols.index(default_date) if default_date in df_cols else 1,
                key=f"join_date_{file.name}_{idx}"
            )

            # Columns to Import
            import_cols = st.multiselect(
                "Select columns to join (excluding HCP & Date)",
                [col for col in df_cols if col not in [join_hcp_col, join_date_col]],
                key=f"import_cols_{file.name}_{idx}"
            )

            # Optional renaming fields
            col_renames = {}
            if import_cols:
                st.markdown("📝 Optional: Rename KPI columns")
                for c in import_cols:
                    new_name = st.text_input(
                        f"Rename `{c}` (leave blank to keep original)",
                        key=f"rename_{file.name}_{c}_{idx}"
                    )
                    if new_name:
                        col_renames[c] = new_name

            # Store the mapping
            file_join_configs[file.name] = {
                "file_obj": file,
                "source": source,
                "join_hcp_col": join_hcp_col,
                "join_date_col": join_date_col,
                "import_cols": import_cols,
                "col_renames": col_renames
            }

    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")

st.subheader("Step 5: Join Files to HCP-Date Base")

if st.button("🔗 Join Files to HCP-Date Base"):
    final_df = cartesian_df.clone()

    for fname, config in file_join_configs.items():
        file = config["file_obj"]
        hcp_col = config["join_hcp_col"]
        date_col = config["join_date_col"]
        import_cols = config["import_cols"]
        col_renames = config["col_renames"]

        try:
            use_cols = [hcp_col, date_col] + import_cols
            df = pl.read_csv(file, columns=use_cols)
            df = df.rename({hcp_col: "HCP_ID", date_col: "Date"})
            if col_renames:
                df = df.rename(col_renames)
            df = df.with_columns(pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
            final_df = final_df.join(df, on=["HCP_ID", "Date"], how="left")
            st.success(f"✅ Joined `{fname}` successfully")
        except Exception as e:
            st.error(f"❌ Failed to join `{fname}`: {e}")

    # Store in session state
    final_df = final_df.fill_null(0)
    st.session_state["geo_column"] = hcp_col
    st.session_state["date_column"] = date_col
    st.session_state["joined_output_df"] = final_df
    st.session_state["merge_complete"] = True

# --- Display final_df if available in session state
if st.session_state.get("merge_complete") and "joined_output_df" in st.session_state:
    final_df = st.session_state["joined_output_df"]

    if not final_df.is_empty():
        st.markdown("### 📄 Final Joined Dataset (Top 50 rows)")
        st.dataframe(final_df.head(50).to_pandas())

        csv_bytes = final_df.write_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Final Dataset as CSV",
            data=csv_bytes,
            file_name="final_joined_output.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Final dataframe is empty or join failed.")

#if st.button("Get Started"):
            #st.switch_page("pages/1_Data_Ingestion.py")

            #Optional 

            #Create DTC-Date Base

            #User inputs ZIP-DMA mapping

            #Joining
