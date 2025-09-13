import streamlit as st
import pandas as pd

# Import the functions from your separate logic file
from validator import validate_social_media_post, assets

# --- Page Configuration and Styling ---
st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp { background-color: #FFFDE7; }
.info-box {
    background-color: #FFFFFF;
    border-left: 6px solid #FFC107;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.info-box p { margin: 0; font-family: sans-serif; }
.info-box .title { font-weight: bold; color: #333; font-size: 1.1em; }
.info-box .value { color: #555; font-size: 1.0em; }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
# st.title("Social Media Post Validation System")
st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
        YellowSense Fraud Detection Using CCR Demo
    </h1>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'current_post' not in st.session_state:
    st.session_state.current_post = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None

# Create two columns for the buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Fetch Latest Post", type="primary", use_container_width=True):
        st.session_state.validation_results = None # Clear old results
        if assets:
            random_post = assets['social_df'].sample(1).iloc[0]
            st.session_state.current_post = {
                "post_text": random_post["post_text"],
                "company": random_post["company"],
                "date_str": random_post["date"].strftime('%Y-%m-%d'),
                "advisor_name": random_post["advisor_name"]
            }
            company_cat = assets['company_cat']
            st.session_state.company_cat = company_cat.loc[company_cat['company'] == random_post["company"], 'company_cat'].iloc[0]

# This entire block only runs if a post has been fetched
if st.session_state.current_post and st.session_state.company_cat:
    post = st.session_state.current_post
    company_cat = st.session_state.company_cat
    st.markdown("---")
    st.subheader("Fetched Post Details")
    
    # Display post details
    st.markdown(f"""
    <div class="info-box">
        <p><span class="title">Company:</span> <span class="value">{post['company']}</span></p>
        <p><span class="title">Post Date:</span> <span class="value">{post['date_str']}</span></p>
        <p><span class="title">Advisor Name:</span> <span class="value">{post['advisor_name']}</span></p>
        <p><span class="title">Post Text:</span> <span class="value">"{post['post_text']}"</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Place the "Validate" button here, inside the block
    with col2:
        if st.button("Validate Post", use_container_width=True):
            with st.spinner('Analyzing...'):
                st.session_state.validation_results = validate_social_media_post(
                    post['post_text'], post['company'], post['date_str'], company_cat, post['advisor_name']
                )

    # Display results if they exist for the current post
    if st.session_state.validation_results:
        results = st.session_state.validation_results
        st.markdown("---")
        st.subheader("Validation Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.markdown(f'<div class="info-box"><p><span class="title">Market/Financial Risk:</span><br><span class="value">{results["market_financial_risk"]:.2f}</span></p></div>', unsafe_allow_html=True)
        res_col2.markdown(f'<div class="info-box"><p><span class="title">Contradiction Score:</span><br><span class="value">{results["contradiction_score"]:.2f}</span></p></div>', unsafe_allow_html=True)
        res_col3.markdown(f'<div class="info-box"><p><span class="title">Advisor Risk Score:</span><br><span class="value">{results["advisor_risk"]:.2f}</span></p></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="info-box"><p><span class="title">FINAL RISK SCORE:</span><br><span class="value" style="font-size: 1.5em; font-weight: bold;">{results["genuinity_score"]:.2f}</span></p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box"><p><span class="title">FINAL VERDICT:</span><br><span class="value" style="font-size: 1.5em; font-weight: bold;">{results["final_result_text"]}</span></p></div>', unsafe_allow_html=True)