# styles.py
import streamlit as st

def inject_sidebar_css():
    st.markdown(
        """
        <style>
         section[data-testid="stSidebar"] { background-color: #001E96 !important; }
         section[data-testid="stSidebar"] * { color: white !important; }
         section[data-testid="stSidebar"] .stButton > button:hover {
             background-color: #001E96 !important; filter: brightness(1.1);
         }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_progress(steps: list[str], current_step: int):
    progress_percent = int((current_step / (len(steps) - 1)) * 100) if steps else 0
    st.markdown(f"""
    <style>
    .progress-container {{ width: 100%; margin-top: 20px; }}
    .progress-bar {{ height: 10px; background: linear-gradient(to right, #001E96, #007aff);
                     border-radius: 10px; width: {progress_percent}%; transition: width 0.4s ease-in-out; }}
    .track {{ background-color: #e0e0e0; height: 10px; width: 100%; border-radius: 10px; margin-bottom: 10px; }}
    .step-labels {{ display: flex; justify-content: space-between; font-size: 14px; color: #333; font-weight: 600; letter-spacing: 0.5px; }}
    </style>
    <div class="progress-container">
        <div class="track"><div class="progress-bar"></div></div>
        <div class="step-labels">
            {''.join([f"<span>{step}</span>" for step in steps])}
        </div>
    </div>
    """, unsafe_allow_html=True)
