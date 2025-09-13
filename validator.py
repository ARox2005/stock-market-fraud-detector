import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_all_models_and_data():
    """Loads all models and data files, cached to run only once."""
    try:
        models_data = {
            "embedding_model": SentenceTransformer('all-MiniLM-L6-v2'),
            "model_1": joblib.load('model_data/model_1_market_financial.joblib'),
            "model_1_features": joblib.load('model_data/model_1_market_financial_features.joblib'),
            "advisor_df": pd.read_csv('data/advisor_data_labeled.csv', parse_dates=['date']),
            "press_df": pd.read_csv('data/press_release_data_labeled.csv', parse_dates=['date']),
            "market_df": pd.read_csv('data/market_data_labeled.csv', parse_dates=['date']),
            "financial_df": pd.read_csv('data/financial_data_labeled.csv', parse_dates=['date']),
            "social_df": pd.read_csv('data/raw_social_data_labeled (1).csv', parse_dates=['date']),
            "company_cat": pd.read_csv('data/company_to_category_map.csv'),
            "test_data": pd.read_csv('test_data/model_1_X_test_data.csv')
        }
        models_data['press_df']['press_release_text'] = models_data['press_df']['press_release_text'].fillna('')
        print("All models and data loaded successfully.")
        return models_data
    except FileNotFoundError as e:
        # st.error(f"Fatal Error: Could not load required file: {e}.")
        return (f"Fatal Error: Could not load required file: {e}.")

# Load all assets into a global variable
assets = load_all_models_and_data()

def get_contradiction_score(post_text, company, post_date):
    """Calculates the contradiction between a post and a company's press release."""
    press_df = assets['press_df']
    embedding_model = assets['embedding_model']
    relevant_releases = press_df[(press_df['company'] == company) & (press_df['date'] <= post_date)].sort_values(by='date', ascending=False)
    if relevant_releases.empty: return 0.5 
    latest_release_text = relevant_releases.iloc[0]['press_release_text']
    embedding1 = embedding_model.encode(post_text, convert_to_tensor=True)
    embedding2 = embedding_model.encode(latest_release_text, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return (1 - cosine_similarity) / 2

def get_advisor_risk(company, advisor_name):
    """Determines the risk associated with a company's advisor."""
    advisor_df = assets['advisor_df']
    company_advisor_info = advisor_df[(advisor_df['company'] == company) & (advisor_df['advisor_name'].fillna('None') == advisor_name)]
    if company_advisor_info.empty: return 0.5
    status = company_advisor_info.iloc[0]['advisor_status']
    # risk_map = {'Terminated': 0.9, 'Under Investigation': 0.8, 'Resigned': 0.7, 'Not Found': 0.5, 'Active': 0.1}
    risk_map = {'Revoked': 0.9, 'Not Found': 0.5, 'Active': 0.1}
    return risk_map.get(status, 0.5)

def validate_social_media_post(post_text, company, date_str, company_cat, advisor_name):
    """Validates a post using Model 1, contradiction, and advisor risk."""
    if assets is None: return {"error": "Models and data not loaded."}
    model_1 = assets['model_1']
    post_date = pd.to_datetime(date_str)
    results={}
    
    print(f"\n--- Validating Post for '{company}' on {date_str} ---")
    print(f"Post Text: \"{post_text}\"")
    
    # Step 1: Get Market & Financial Risk from Model 1
    # model_1_input = create_model_1_features(company, post_date)
    # model_1_input = pd.read_csv('test_data\model_1_X_test_data.csv')
    model_1_input = assets['test_data'][assets['test_data']['company_cat'] == company_cat]
    if model_1_input is None:
        market_financial_risk = 0.5 # Neutral score if no data
    else:
        market_financial_risk = model_1.predict_proba(model_1_input[(model_1_input['company_cat']==company_cat)])[:, 1][-1]
    results['market_financial_risk'] = market_financial_risk
    print(f"Market/Financial Risk (Model 1): {np.median(market_financial_risk):.2f}")

    # Step 2: Get Contradiction Score
    contradiction_score = get_contradiction_score(post_text, company, post_date)
    results['contradiction_score'] = contradiction_score
    print(f"Contradiction with Press Release: {contradiction_score:.2f}")

    # Step 3: Get Advisor Risk Score
    advisor_risk = get_advisor_risk(company, advisor_name)
    results["advisor_risk"] = advisor_risk
    print(f"Advisor Risk Score: {advisor_risk:.2f}")
    
    # Step 4: Calculate Final Genuinity Score with all three components
    w1 = 0.4 # Weight for market/financial risk
    w2 = 0.3 # Weight for contradiction
    w3 = 0.3 # Weight for advisor risk
    
    genuinity_score = (w1 * market_financial_risk) + (w2 * contradiction_score) + (w3 * advisor_risk)
    results["genuinity_score"] = genuinity_score
    
    print(f"--- FINAL GENUINITY SCORE: {genuinity_score:.2f} ---")
    
    # if genuinity_score > 0.65:
    #     print("Result: High likelihood the post is a genuine warning.")
    # elif genuinity_score > 0.4:
    #     print("Result: Moderate likelihood. Worth monitoring.")
    # else:
    #     print("Result: Low likelihood. May be unsubstantiated rumor.")
    if results["genuinity_score"] > 0.65:
        results["final_result_text"] = "❌ The Post Highly Likely a Fraud attempt."
    elif results["genuinity_score"] > 0.4:
        results["final_result_text"] = "⚠️ Moderate likelihood. Worth monitoring."
    else:
        results["final_result_text"] = "✅ Low likelihood, ."
        
    return results
        
    # return genuinity_score