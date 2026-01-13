import streamlit as st
import os
import base64

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # -> tool_dev/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set page configuration
st.set_page_config(
    page_title="ProcTimize",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def get_base64_gif(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Custom CSS to make the page more aesthetic
def local_css():
    gif_data = get_base64_gif("new_land.gif")  # Make sure this file exists in the same directory as Home.py

    st.markdown(f"""
    <style>
        html, body, [class*="stApp"] {{
            position: relative;
            background: url("data:image/gif;base64,{gif_data}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Helvetica Neue', sans-serif;
            animation: fadeIn 0.75s ease-in;
            z-index: 0;
        }}

        section[data-testid="stSidebar"] {{
            display: none !important;
        }}


        /* Glass box behind content */
        .glass-box-visual {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 50%;
            height: 50%;
            padding: 3rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 0 15px rgba(26, 188, 156, 0.5), 0 0 30px rgba(26, 188, 156, 0.3);
            z-index: 0;
            border: 2px solid rgba(26, 188, 156, 0.2);
            border-image: linear-gradient(90deg, #146C94, #1ABC9C) 1;
            animation: borderMove 4s linear infinite;
        }}



        html::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            filter: blur(7px);
            background: inherit;
            background-color: rgba(0,0,0,0.4);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            opacity: 0.1;
            z-index: -1;
        }}

        .block-container {{
            position: relative;
            z-index: 1;
        }}

        .main-title {{
            font-size: 5rem !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            letter-spacing: 2px;
            text-align: center !important;
            margin-top: 25vh !important;
            margin-bottom: 1rem !important;
            animation: fadeSlideUp 1.2s ease-out forwards;
        }}

        .subtitle {{
            font-size: 1.5rem;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 2rem;
        }}

        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            text-align: center;
            padding: 1rem;
            font-size: 0.9rem;
            color: #606060;
        }}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}

        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}

        .button-container {{
            display: flex;
            justify-content: center;
            margin-top: 3rem;
        }}

        .stButton > button {{
            background: linear-gradient(90deg, #ffffff 0%, #ffffff 100%);
            color: #146C94 !important;
            border: 1px solid #146C94;
            padding: 0.8rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            text-decoration: none;
            display: inline-block;
            animation: pulse 2s infinite;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(0, 0, 0, 0.25);
            background: linear-gradient(180deg, #1ABC9C 0%, #001E96 100%);
            color: white !important;
        }}

        @keyframes fadeSlideUp {{
            0% {{ opacity: 0; transform: translateY(30px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(26, 188, 156, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(26, 188, 156, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(26, 188, 156, 0); }}
        }}

        @keyframes borderMove {{
            0% {{ border-image-slice: 1; }}
            100% {{ border-image-slice: 1; }}
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
    </style>
    """, unsafe_allow_html=True)


# Main app
def main():
    # local_css()
    # st.markdown('<div class="glass-box-visual"></div>', unsafe_allow_html=True)

    
    # # Center the title using markdown with custom class
    # st.markdown("<h1 class='main-title'>ProcTimize</h1>", unsafe_allow_html=True)
    
    # # Subtitle
    # st.markdown("<p class='subtitle'>A modern platform for Marketing Mix Modeling — ingest, clean, and transform your data effortlessly.</p>", unsafe_allow_html=True)
    
    # # Create direct link button using markdown + HTML with absolute path to pages
    # col1, col2, col3 = st.columns([2, 1, 2])
    # with col2:
    #     if st.button("Get Started"):
            # st.switch_page("pages/1_Data_Ingestion.py")
    if st.button("Get Started"):
        st.switch_page("pages/1_Data_Ingestion.py")

    

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 ProcDNA. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
