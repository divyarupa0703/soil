from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('soil/soil_health_model.pkl')

@app.route('/')
def home():
    return render_template('result.html')

# Define preprocessing function
def preprocess_input(data):
    # Ensure the input data has the correct keys
    df = pd.DataFrame([data])
    
    # Rename columns to match expected format
    df.rename(columns={
        'Organic_Matter': 'Organic_Matter_%',
        'CEC': 'Cation_Exchange_Capacity_meq/100g',
        'EC': 'Electrical_Conductivity_dS/m',
        'Bulk_Density': 'Bulk_Density_g/cm³'
    }, inplace=True)
    
    # Add categories based on thresholds
    df['pH_Category'] = df['pH'].apply(lambda x: 'Acidic' if x < 6 else ('Neutral' if x == 6 else 'Alkaline'))
    df['OM_Category'] = df['Organic_Matter_%'].apply(lambda x: 'Low' if x < 3 else ('Medium' if x <= 5 else 'High'))
    df['CEC_Category'] = df['Cation_Exchange_Capacity_meq/100g'].apply(lambda x: 'Low' if x < 10 else ('Medium' if x <= 20 else 'High'))
    df['EC_Category'] = df['Electrical_Conductivity_dS/m'].apply(lambda x: 'Low' if x < 1 else ('Moderate' if x <= 2 else 'High'))
    df['Bulk_Density_Category'] = df['Bulk_Density_g/cm³'].apply(lambda x: 'Low' if x < 1.2 else ('Medium' if x <= 1.6 else 'High'))
    
    # Select only relevant columns
    df = df[['pH', 'Organic_Matter_%', 'Cation_Exchange_Capacity_meq/100g', 
             'Electrical_Conductivity_dS/m', 'Bulk_Density_g/cm³']]
    
    return df

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Debug: Print received data
        print(f"Received Data: {data}")

        # Preprocess input data
        df = preprocess_input(data)
        
        # Debug: Print DataFrame columns
        print(f"Input DataFrame Columns: {df.columns}")
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return prediction
        response = {'Soil_Health_Category': prediction[0]}
        return jsonify(response)
    
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 400

@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.json
        
        # Debug: Print received data
        print(f"Received Data for Recommendations: {data}")
        
        recommendations = provide_recommendations(data)
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 400

def provide_recommendations(data):
    recommendations = []
    
    # pH Recommendations
    if data['pH'] < 5.5:
        recommendations.append("pH is very low. Consider applying lime to significantly increase soil pH.")
    elif 5.5 <= data['pH'] < 6:
        recommendations.append("pH is slightly acidic. Applying lime can help neutralize the soil pH.")
    elif 6 < data['pH'] <= 7:
        recommendations.append("pH is neutral, which is ideal for most crops.")
    elif 7 < data['pH'] <= 8:
        recommendations.append("pH is slightly alkaline. Consider applying sulfur or other acidifying agents to lower the pH if necessary.")
    else:
        recommendations.append("pH is very high. Strong acidifying agents may be required to lower the pH.")

    # Organic Matter Recommendations
    if data['Organic_Matter'] < 2:
        recommendations.append("Organic matter is very low. Increase organic matter significantly by adding compost or well-rotted manure.")
    elif 2 <= data['Organic_Matter'] < 3:
        recommendations.append("Organic matter is low. Add compost or organic fertilizers to improve soil fertility.")
    elif 3 <= data['Organic_Matter'] <= 5:
        recommendations.append("Organic matter is at a moderate level. Maintain this level by regularly adding organic material.")
    else:
        recommendations.append("Organic matter is high. Continue with your current practices to maintain soil health.")

    # Cation Exchange Capacity (CEC) Recommendations
    if data['CEC'] < 5:
        recommendations.append("CEC is very low. Consider adding significant amounts of organic matter or using specific fertilizers to improve nutrient retention.")
    elif 5 <= data['CEC'] < 10:
        recommendations.append("CEC is low. Increase organic matter and use fertilizers to enhance nutrient holding capacity.")
    elif 10 <= data['CEC'] <= 20:
        recommendations.append("CEC is moderate. Regular fertilization should be sufficient to meet nutrient needs.")
    else:
        recommendations.append("CEC is high. Your soil has good nutrient holding capacity. Maintain this level with balanced fertilization.")

    # Electrical Conductivity (EC) Recommendations
    if data['EC'] < 0.5:
        recommendations.append("EC is very low. Consider applying fertilizers to increase soil salinity to a moderate level.")
    elif 0.5 <= data['EC'] < 1:
        recommendations.append("EC is low. Soil salinity is acceptable, but monitor levels to ensure it does not decrease further.")
    elif 1 <= data['EC'] <= 2:
        recommendations.append("EC is moderate. Manage irrigation practices to avoid salt buildup.")
    else:
        recommendations.append("EC is high. High salinity may affect plant growth. Consider leaching the soil or using salinity-tolerant plants.")

    # Bulk Density Recommendations
    if data['Bulk_Density'] < 1.0:
        recommendations.append("Bulk density is very low. Ensure good soil structure and avoid excessive tillage.")
    elif 1.0 <= data['Bulk_Density'] < 1.2:
        recommendations.append("Bulk density is low. Continue to maintain soil structure and avoid compaction.")
    elif 1.2 <= data['Bulk_Density'] < 1.6:
        recommendations.append("Bulk density is moderate. Monitor soil conditions and avoid practices that lead to further compaction.")
    else:
        recommendations.append("Bulk density is high. Consider tilling the soil, adding organic amendments, or using cover crops to reduce compaction.")

    return recommendations


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
