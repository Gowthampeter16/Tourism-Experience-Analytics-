import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Tourism Rating Predictor", layout="wide")

# Title
st.title("🏖️ Tourism Rating Predictor")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("df_cleaned.xlsx")

df_cleaned = load_data()

# Load model 
@st.cache_resource
def load_model():
    return joblib.load("pipeline_xgb.pkl")  

model = load_model()

# Sidebar for user inputs
st.sidebar.header("🧭 Select Your Inputs")

# 1. Continent
continent = st.sidebar.selectbox(
    "Select Continent", 
    options=df_cleaned["Continent"].dropna().unique()
)

# 2. Region (filtered by Continent)
filtered_regions = df_cleaned[df_cleaned["Continent"] == continent]["Region"].dropna().unique()
region = st.sidebar.selectbox(
    "Select Region",
    options=filtered_regions
)

# 3. Country (filtered by Region)
filtered_countries = df_cleaned[
    (df_cleaned["Continent"] == continent) & (df_cleaned["Region"] == region)
]["Country"].dropna().unique()
country = st.sidebar.selectbox(
    "Select Country",
    options=filtered_countries
)

# 4. City (filtered by Country)
filtered_cities = df_cleaned[
    (df_cleaned["Continent"] == continent) &
    (df_cleaned["Region"] == region) &
    (df_cleaned["Country"] == country)
]["CityName"].dropna().unique()
city = st.sidebar.selectbox(
    "Select City",
    options=filtered_cities
)

# 5. Visit Details
visit_year = st.sidebar.selectbox(
    "Select Visit Year",
    options=sorted(df_cleaned["VisitYear"].dropna().unique())
)

visit_month = st.sidebar.selectbox(
    "Select Visit Month",
    options=sorted(df_cleaned["VisitMonth"].dropna().unique())
)

mode_of_visit = st.sidebar.selectbox(
    "Select Mode of Visit",
    options=df_cleaned["VisitMode"].dropna().unique()
)

# 6. Attraction Attributes
attraction_type = st.sidebar.selectbox(
    "Select Attraction Type",
    options=df_cleaned["AttractionTypeId"].dropna().unique()
)

# 7. Previous Average Rating
avg_rating = st.sidebar.slider(
    "Previous Average Rating",
    min_value=1.0,
    max_value=5.0,
    step=0.1
)

input_data = pd.DataFrame({
    "Continent": [continent],
    "Region": [region],
    "Country": [country],
    "CityName": [city], 
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "VisitMode": [mode_of_visit],  
    "AttractionTypeID": [attraction_type],  
    "AvgRating": [avg_rating],
    "UserVisitCount": [0],
    "Attraction": ["Unknown"],
    "AttrVisitCount": [0],
    "UserAvgRating": [avg_rating],
})


st.subheader("Your Selections:")
st.write(input_data)

# Predict Button
if st.button("🔮 Predict Rating"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🌟 Predicted Rating: {prediction[0]:.2f} / 5")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")