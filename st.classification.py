import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_excel('df_cleaned.xlsx')

# Load model
model = joblib.load('knn_pipeline_model.pkl')

st.title("🌍 Tourism Visit Mode Prediction")

# --- Dependent dropdowns ---

# 1. Continent
continent = st.selectbox('Select Continent', sorted(df['Continent'].unique()))

# 2. Region (dependent on Continent)
region_options = df[df['Continent'] == continent]['Region'].unique()
region = st.selectbox('Select Region', sorted(region_options))

# 3. Country (dependent on Region)
country_options = df[(df['Continent'] == continent) & (df['Region'] == region)]['Country'].unique()
country = st.selectbox('Select Country', sorted(country_options))

# 4. CityName (dependent on Country)
city_options = df[(df['Continent'] == continent) & (df['Region'] == region) & (df['Country'] == country)]['CityName'].unique()
city = st.selectbox('Select CityName', sorted(city_options))

# 5. Attraction (dependent on CityName)
attraction_options = df[
    (df['Continent'] == continent) &
    (df['Region'] == region) &
    (df['Country'] == country) &
    (df['CityName'] == city)
]['Attraction'].unique()
attraction = st.selectbox('Select Attraction', sorted(attraction_options))

# 6. Auto-fill AttractionTypeId
attraction_type_id = df[df['Attraction'] == attraction]['AttractionTypeId'].values[0]
st.write(f"Auto-filled AttractionTypeId: **{attraction_type_id}**")

# 7. Month and Year
visit_month = st.selectbox('Select VisitMonth', sorted(df['VisitMonth'].unique()))
visit_year = st.selectbox('Select VisitYear', sorted(df['VisitYear'].unique()))

# --- Prediction ---
if st.button('Predict Visit Mode'):
    input_data = pd.DataFrame({
        'Continent': [continent],
        'Region': [region],
        'Country': [country],
        'CityName': [city],
        'Attraction': [attraction],
        'AttractionTypeId': [attraction_type_id],
        'UserAvgRating': [4.0],  # Dummy value
        'AttrVisitCount': [10],  # Dummy value
        'VisitMonth': [visit_month],
        'VisitYear': [visit_year],
        'VisitMode_Region': ['Unknown'],  # Dummy value
        'UserId': [0]  # Dummy value
    })

    prediction = model.predict(input_data)
    st.success(f'🎯 Predicted Visit Mode: **{prediction[0]}**')