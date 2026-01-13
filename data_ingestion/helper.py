# helpers.py
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import polars as pl

RE_NPI = re.compile(r"^[12]\d{9}$")

def luhn_valid_npi(npi: str) -> bool:
    if not RE_NPI.fullmatch(npi):
        return False
    base, chk = npi[:9], ord(npi[9]) - 48
    s = 24  # CMS '80840' tweak
    for i in range(9):
        d = ord(base[8 - i]) - 48
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return ((10 - (s % 10)) % 10) == chk

def detect_date_columns_by_sampling(df, sample_size=100, threshold=0.8, formats=None):
    if formats is None:
        formats = ["%d/%m/%Y", "%Y/%m/%d", "%Y/%d/%m", "%m/%d/%Y",
                   "%m-%d-%Y", "%d-%m-%Y", "%Y-%d-%m", "%Y-%m-%d"]
    date_cols = []
    for col in df.columns:
        try:
            non_null_vals = df[col].dropna()
            if non_null_vals.empty:
                continue
            sample_vals = non_null_vals.sample(min(sample_size, len(non_null_vals)), random_state=42)
            best_success, best_parsed = 0, None
            for fmt in formats:
                parsed_dates = pd.to_datetime(sample_vals, format=fmt, errors='coerce')
                success_ratio = parsed_dates.notna().mean()
                if success_ratio > best_success:
                    best_success, best_parsed = success_ratio, parsed_dates
            if best_success >= threshold:
                min_date, max_date = best_parsed.min(), best_parsed.max()
                if pd.notna(min_date) and pd.notna(max_date) and 1900 <= min_date.year <= 2100 and 1900 <= max_date.year <= 2100:
                    date_cols.append(col)
        except Exception:
            pass
    return date_cols

def reset_full_workflow():
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in [
        "df_final", "df_final_list", "merge_done",
        "df_transformed", "transform_complete",
        "numerical_config_dict", "categorical_config_dict",
    ]:
        st.session_state.pop(k, None)
    st.session_state.pop("uploaded_files", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state["current_step"] = 0
    for k in list(st.session_state.keys()):
        if k.startswith((
            "select_cols_", "rename_editor_", "date_format_", "date_col_",
            "date_cols_", "npi_", "groupby_", "analysis_", "granularity_",
            "download_", "kpi_", "modify_granularity_", "normalize_"
        )):
            st.session_state.pop(k, None)

def last_working_day(year, month, work_days):
    if month == 12:
        month = 0
        year += 1
    last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    if work_days == 5:
        while last_day.weekday() > 4:
            last_day -= timedelta(days=1)
    return last_day

def last_week_apportion(tactic_df, date_col_name, kpi_col_list, work_days):
    def rename_adjusted(kpi_name): return "adjusted_" + kpi_name
    tactic_df['month'] = tactic_df[date_col_name].dt.to_period('M')
    last_working_day_dict = {m: last_working_day(m.year, m.month, work_days) for m in tactic_df['month'].unique()}
    tactic_df['last_working_date'] = tactic_df['month'].map(last_working_day_dict)
    tactic_df['day_diff'] = (tactic_df['last_working_date'] - tactic_df[date_col_name] + timedelta(days=1)).dt.days
    adjusted_col_list = []
    for kpi_name in kpi_col_list:
        tactic_df[rename_adjusted(kpi_name)] = tactic_df.apply(
            lambda row: ((work_days - row['day_diff']) / work_days) * row[kpi_name] if row['day_diff'] < work_days else 0,
            axis=1
        )
        adjusted_col_list.append(rename_adjusted(kpi_name))
    for kpi_name in kpi_col_list:
        tactic_df[kpi_name] = tactic_df[kpi_name] - tactic_df[rename_adjusted(kpi_name)]
    new_rows = tactic_df[tactic_df[adjusted_col_list].gt(0).any(axis=1)].copy()
    new_rows[date_col_name] = new_rows[date_col_name] + pd.offsets.MonthBegin()
    for kpi_name in kpi_col_list:
        new_rows[kpi_name] = new_rows[rename_adjusted(kpi_name)]
    tactic_df = pd.concat([tactic_df, new_rows], ignore_index=True)
    tactic_df.drop(['last_working_date', 'day_diff', 'month'], axis=1, inplace=True)
    for adj_col in adjusted_col_list:
        tactic_df.drop(adj_col, axis=1, inplace=True)
    return tactic_df

def to_polars_date_safe(df_final: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    if not isinstance(df_final, pl.DataFrame):
        df_final = pl.from_pandas(df_final)
    df_final = df_final.with_columns([
        pl.col(col).cast(pl.Date)
        for col, dtype in zip(df_final.columns, df_final.dtypes)
        if dtype == pl.Datetime
    ])
    return df_final
