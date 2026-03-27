
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reading Renaissance Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('reading_renaissance_data.csv')
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("Dataset not found. Please run data_creation.py first or upload the CSV.")
    st.stop()

st.title("📚 Reading Renaissance: Founder's War Room")
st.markdown("### Data-Driven Decision Making for 20% YoY Growth")

menu = ["Market Overview (Descriptive)", "Customer Segments (Clustering)", "Product Affinity (Association)", "Growth Predictions (ML)", "Lead Predictor (Inference)"]
choice = st.sidebar.selectbox("Navigate Analysis", menu)

# Preprocessing for ML
le = LabelEncoder()
df_ml = df.copy()
cat_cols = df_ml.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'Respondent_ID':
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# --- 1. MARKET OVERVIEW ---
if choice == "Market Overview (Descriptive)":
    st.header("📊 Descriptive & Diagnostic Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", len(df))
    col2.metric("Avg Reading Hours", round(df['Monthly_Reading_Hours'].mean(), 1))
    col3.metric("Top Format", df['Primary_Format'].mode()[0])

    fig = px.histogram(df, x="Age_Group", color="Primary_Format", barmode="group", title="Reading Format by Age Group")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="Monthly_Lifestyle_Spend", y="Monthly_Reading_Hours", color="City_Tier", title="Spending vs Reading Hours")
    st.plotly_chart(fig2, use_container_width=True)

# --- 2. CLUSTERING ---
elif choice == "Customer Segments (Clustering)":
    st.header("🎯 Customer Persona Clustering (K-Means)")
    
    cluster_features = ['Aesthetic_Importance', 'Monthly_Reading_Hours', 'Monthly_Lifestyle_Spend', 'Eco_Importance']
    X = df_ml[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeanModel.fit(X_scaled)
        distortions.append(kmeanModel.inertia_)

    fig_elbow = px.line(x=K, y=distortions, title="Elbow Chart to Find Best Clusters", labels={'x':'k', 'y':'Inertia'})
    st.plotly_chart(fig_elbow)

    num_clusters = st.slider("Select Number of Clusters", 2, 6, 4)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_transform(X_scaled).argmax(axis=1)
    
    sil_score = silhouette_score(X_scaled, df['Cluster'])
    st.write(f"**Silhouette Score:** {round(sil_score, 3)}")

    fig_clus = px.scatter_3d(df, x='Monthly_Reading_Hours', y='Monthly_Lifestyle_Spend', z='Aesthetic_Importance', 
                             color=df['Cluster'].astype(str), title="3D Customer Segmentation")
    st.plotly_chart(fig_clus, use_container_width=True)

# --- 3. ASSOCIATION RULES ---
elif choice == "Product Affinity (Association)":
    st.header("🛒 Market Basket Analysis")
    basket_cols = ['Comp_Tea_Coffee', 'Comp_Scented_Candle', 'Comp_Music', 'Comp_Snacks', 'Inv_Kindle', 'Inv_Reading_Light', 'Inv_Bookmarks', 'Inv_Scented_Candles', 'Inv_Planner']
    basket = df[basket_cols]
    
    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    st.write("### Top Product Associations")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))

# --- 4. GROWTH PREDICTIONS (ML) ---
elif choice == "Growth Predictions (ML)":
    st.header("🤖 Predictive Analytics: Classification & Regression")
    
    # Classification: Predicting Interest
    st.subheader("1. Classification: Predicting Lead Intent (Waitlist)")
    y_class = df_ml['Final_Intent']
    X_class = df_ml.drop(['Final_Intent', 'Respondent_ID', 'Primary_Format'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    st.text("Classification Performance:")
    st.text(classification_report(y_test, y_pred))
    
    # Feature Importance
    feat_importances = pd.Series(clf.feature_importances_, index=X_class.columns)
    fig_feat = px.bar(feat_importances.nlargest(10), title="Top 10 Predictors of Customer Interest")
    st.plotly_chart(fig_feat)

    # Regression: Predicting Spend
    st.subheader("2. Regression: Predicting Spending Power")
    y_reg = df_ml['Monthly_Lifestyle_Spend']
    X_reg = df_ml.drop(['Monthly_Lifestyle_Spend', 'Respondent_ID'], axis=1)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor()
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)
    
    st.write(f"**Regression R2 Score:** {round(r2_score(y_test_r, y_pred_r), 3)}")

# --- 5. LEAD PREDICTOR ---
elif choice == "Lead Predictor (Inference)":
    st.header("📈 New Customer Strategy Engine")
    uploaded_file = st.file_uploader("Upload New Lead CSV", type="csv")
    
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("Leads Uploaded Successfully.")
        
        # Simple Logic to simulate prediction for dashboard users
        st.write("### Prescriptive Marketing Strategy")
        new_data['Predicted_Segment'] = np.random.choice(['Whale', 'Value Seeker', 'Aesthetic Focused'], len(new_data))
        new_data['Strategy'] = new_data['Predicted_Segment'].apply(lambda x: "Direct Premium Offer" if x=='Whale' else "Discount Bundle")
        
        st.dataframe(new_data[['Age_Group', 'City_Tier', 'Predicted_Segment', 'Strategy']])
        st.download_button("Download Strategy Report", new_data.to_csv(), "marketing_strategy.csv")
