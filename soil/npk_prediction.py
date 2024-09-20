from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load the saved model
preprocessor_file = 'soil/soil_preprocessor.pkl'
model_file = 'soil/multioutput_regressor.pkl'
preprocessor = joblib.load(preprocessor_file)
multioutput_regressor = joblib.load(model_file)

# Create a Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def index():
    return render_template('index.html')

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert all values to appropriate data types
        for key in data:
            if key in ['Location_Latitude', 'Location_Longitude', 'Depth_cm', 'pH', 'Organic_Matter_%',
                       'Moisture_Content_%', 'Bulk_Density_g/cm³', 'Cation_Exchange_Capacity_meq/100g',
                       'Electrical_Conductivity_dS/m', 'Porosity_%', 'Water_Holding_Capacity_%']:
                data[key] = float(data[key])
        
        # Create a DataFrame
        df = pd.DataFrame([data])
        print(f"Input DataFrame:\n{df}")

        # Preprocess the new data
        df_transformed = preprocessor.transform(df)
        print(f"Transformed Data:\n{df_transformed}")

        # Make prediction
        prediction = multioutput_regressor.predict(df_transformed)
        print(f"Prediction:\n{prediction}")

        # Return predictions
        response = {
            'Nitrogen': prediction[0][0],
            'Phosphorus': prediction[0][1],
            'Potassium': prediction[0][2]
        }
        return jsonify(response)
    
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
