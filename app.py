import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# Configure page settings
st.set_page_config(page_title="COVID-19 Clustering", layout="wide")

# Custom CSS for iOS/iPadOS style, black/red theme, and gradient background
st.markdown("""
    <style>
    /* Gradient Background: Black to Red in bottom right */
    .stApp {
        background: linear-gradient(to bottom right, #000000 40%, #a10000 100%);
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* iOS/iPadOS style glassmorphism cards */
    .block-container {
        background: rgba(20, 20, 20, 0.4);
        border-radius: 30px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 3rem;
        border: 1px solid rgba(255, 50, 50, 0.2);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    }

    h1, h2, h3, p, label {
        color: #ffffff !important;
    }

    /* Styling interactive elements to match the red theme */
    div[data-baseweb="select"] > div {
        background-color: rgba(0,0,0,0.5);
        border-radius: 12px;
        border: 1px solid #ff3b30;
        color: white;
    }
    
    div[data-baseweb="slider"] div {
        background-color: #ff3b30 !important; 
    }
    
    /* Team citation text block */
    .team-cite {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        text-align: center;
        font-size: 16px;
        color: #ffcccc;
        margin-top: 10px;
        margin-bottom: 30px;
        padding: 15px;
        border: 1px solid rgba(255,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Header and Team Citation
st.title("🦠 COVID-19 Clustering Analysis")

st.markdown(
    '''<div class="team-cite">
    <b>Team Name:</b> TI1<br>
    <b>Team Members:</b> Derrick, Devisa Angella, Hillary Excelcia Glory Hallatu
    </div>''', 
    unsafe_allow_html=True
)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('covid.csv')
        cols_to_use = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return df, cols_to_use
    except FileNotFoundError:
        import numpy as np
        st.warning("`covid.csv` not found. Generating sample COVID-19 data to demonstrate the UI.")
        df = pd.DataFrame({
            'Confirmed Cases': np.random.randint(1000, 100000, 150),
            'Deaths': np.random.randint(10, 5000, 150),
            'Recovered': np.random.randint(500, 80000, 150),
            'Active Cases': np.random.randint(100, 20000, 150)
        })
        cols_to_use = ['Confirmed Cases', 'Deaths', 'Recovered', 'Active Cases']
        return df, cols_to_use

df, numeric_cols = load_data()

if len(numeric_cols) >= 2:
    st.subheader("Configure Clustering Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("Select X-axis Feature", numeric_cols, index=0)
        feature_y = st.selectbox("Select Y-axis Feature", numeric_cols, index=1)
    
    with col2:
        k_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=3)
        
    # Drop rows with missing values in the selected columns to prevent NaN errors
    df_clean = df.dropna(subset=[feature_x, feature_y]).copy()

    if len(df_clean) > 0:
        # Perform KMeans Clustering
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        df_clean['Cluster'] = kmeans.fit_predict(df_clean[[feature_x, feature_y]])
        df_clean['Cluster'] = df_clean['Cluster'].astype(str)

        # Plotly Scatter Plot
        fig = px.scatter(
            df_clean, 
            x=feature_x, 
            y=feature_y, 
            color='Cluster',
            title=f"K-Means Clustering Results (K={k_clusters})",
            color_discrete_sequence=px.colors.sequential.Reds_r,
            template="plotly_dark"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color="white"),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Cleaned Data Reference"):
            st.dataframe(df_clean, use_container_width=True)
    else:
        st.error("After removing missing values for the selected columns, there is no data left to cluster. Please select different features.")
else:
    st.error("The dataset does not have enough numerical columns to perform clustering.")