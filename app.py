from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

try:
    model = load_model('model.h5')
    with open("label_encoder_gender.pkl", 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open("one_hot_encoder_geo.pkl", 'rb') as file:
        one_hot_encoder_geo = pickle.load(file)
    with open("scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            
            input_data = {
                'CreditScore': int(request.form['CreditScore']),
                'Geography': request.form['Geography'],
                'Gender': request.form['Gender'],
                'Age': int(request.form['Age']),
                'Tenure': int(request.form['Tenure']),
                'Balance': float(request.form['Balance']),
                'NumOfProducts': int(request.form['NumOfProducts']),
                'HasCrCard': int(request.form['HasCrCard']),
                'IsActiveMember': int(request.form['IsActiveMember']),
                'EstimatedSalary': float(request.form['EstimatedSalary'])
            }

            
            geo_encoded = one_hot_encoder_geo.transform([[input_data['Geography']]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
            
            input_df = pd.DataFrame([input_data])
            input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
            input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
            
            input_scaled = scaler.transform(input_df)
            
           
            prediction = model.predict(input_scaled)
            prediction_proba = prediction[0][0]

            if prediction_proba > 0.5:
                result = 'The customer is likely to churn.'
            else:
                result = 'The customer is not likely to churn.'

            return render_template('result.html', prediction_text=result)

        except Exception as e:  
            return f"An error occurred during prediction: {str(e)}"

    return render_template('form.html')  

if __name__ == '__main__':
    app.run(debug=True)
