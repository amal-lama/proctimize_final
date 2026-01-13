# tab_profiling.py
import numpy as np
import pandas as pd
import streamlit as st
from helper import luhn_valid_npi

def render_profiling():
    st.markdown("### Data Profiling")
    if not st.session_state.get("merge_done") or "df_final" not in st.session_state:
        st.info("Merge files first (Vertical Stack or Horizontal Join), then return to profiling.")
        return

    df_any = st.session_state["df_final"]
    pdf = df_any.to_pandas() if hasattr(df_any, "to_pandas") else df_any

    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(pdf):,}")
    c2.metric("Columns", f"{pdf.shape[1]:,}")
    try:
        mem_mb = pdf.memory_usage(deep=True).sum() / (1024**2)
        c3.metric("Memory", f"{mem_mb:.2f} MB")
    except Exception:
        c3.metric("Memory", "n/a")
    st.dataframe(pdf.head(10), use_container_width=True)

    st.subheader("Schema")
    schema_df = pd.DataFrame({"column": pdf.columns, "dtype": [str(t) for t in pdf.dtypes]})
    st.dataframe(schema_df, use_container_width=True)

    st.subheader("No HCP, Date Row removal")
    npi_candidates = [c for c in pdf.columns if "npi" in c.lower()]
    date_candidates = [c for c in pdf.columns if "date" in c.lower()] + \
                      [c for c in pdf.select_dtypes(include=["datetime64[ns]"]).columns]
    npi_col_rm = st.selectbox("NPI column for removal rule:", options=pdf.columns,
                              index=pdf.columns.get_loc(npi_candidates[0]) if npi_candidates else 0)
    date_col_rm = st.selectbox("Date column for removal rule:", options=pdf.columns,
                               index=pdf.columns.get_loc(date_candidates[0]) if date_candidates else 0)

    if st.button("Remove rows where BOTH NPI and Date are missing"):
        npi_series_rm = pdf[npi_col_rm].astype("string")
        npi_na = npi_series_rm.isna() | (npi_series_rm.str.strip() == "")
        date_series_rm = pdf[date_col_rm]
        if not np.issubdtype(date_series_rm.dtype, np.datetime64):
            date_series_rm = pd.to_datetime(date_series_rm, errors="coerce")
        date_na = date_series_rm.isna()
        drop_mask = npi_na & date_na
        dropped = int(drop_mask.sum())
        pdf_clean = pdf.loc[~drop_mask].copy()
        st.session_state["df_final"] = pdf_clean
        st.success(f"Dropped {dropped:,} rows where BOTH `{npi_col_rm}` and `{date_col_rm}` were missing.")
        st.dataframe(pdf_clean.head(10), use_container_width=True)

    st.subheader("NPI Format check")
    npi_col = st.selectbox(
        "Select the column to check for NPI format:",
        options=pdf.columns,
        help="Choose the column containing NPI numbers"
    )

    def _coerce_to_str(x):
        try:
            if pd.notna(x) and float(x).is_integer():
                return str(int(x))
        except Exception:
            pass
        return str(x)

    npi_series = pdf[npi_col].apply(_coerce_to_str)
    npi_strs = npi_series.str.strip().str.replace(r"[^\d]", "", regex=True)
    npi_valid = npi_strs.str.match(r"^[12]\d{9}$")

    n_valid = int(npi_valid.sum())
    n_invalid = int((~npi_valid).sum())
    st.write(f"Format-valid NPIs (10 digits, starts 1/2): {n_valid:,}")
    st.write(f"Format-invalid NPIs: {n_invalid:,}")
    # if n_invalid > 0:
    #     st.write("Sample format-invalid NPIs:")
    #     st.dataframe(npi_strs[~npi_valid].drop_duplicates().head(10))

    # if n_valid > 0:
    #     format_valid_npis = npi_strs[npi_valid]
    #     cache = {}
    #     luhn_results = []
    #     for npi in format_valid_npis:
    #         if npi in cache:
    #             luhn_results.append(cache[npi])
    #         else:
    #             valid = luhn_valid_npi(npi)
    #             cache[npi] = valid
    #             luhn_results.append(valid)
    #     luhn_valid_series = pd.Series(luhn_results, index=format_valid_npis.index)
    #     n_luhn_valid = int(luhn_valid_series.sum())
    #     n_luhn_invalid = int(len(luhn_valid_series) - n_luhn_valid)
    #     st.write(f"Luhn-valid NPIs (among format-valid): {n_luhn_valid:,}")
    #     st.write(f"Luhn-invalid NPIs (among format-valid): {n_luhn_invalid:,}")
    #     if n_luhn_invalid > 0:
    #         luhn_invalid_npis = format_valid_npis[~luhn_valid_series]
    #         st.write("Sample Luhn-invalid NPIs:")
    #         st.dataframe(luhn_invalid_npis.drop_duplicates().head(10))
