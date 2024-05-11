from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_status = ""  # Initialize with an empty string

    if request.method == 'POST':
        # Get the form data
        age = int(request.form['age'])
        sex = request.form['sex']
        present_city = request.form['present_city']
        present_state = request.form['present_state']

        # Convert inputs to model format
        sex_female = 1 if sex == 'FEMALE' else 0
        sex_male = 1 if sex == 'MALE' else 0
        city_bengaluru = 1 if present_city == 'Bengaluru City' else 0
        state_karnataka = 1 if present_state == 'Karnataka' else 0

        # Create a data frame for the input data
        input_data = pd.DataFrame({
            'age': [age],
            'Sex_FEMALE': [sex_female],
            'Sex_MALE': [sex_male],
            'PresentCity_Bengaluru City': [city_bengaluru],
            'PresentState_Karnataka': [state_karnataka]
        })

        # Make a prediction
        prediction = model.predict(input_data)
        predicted_status = prediction[0]

    return render_template('index.html', predicted_status=predicted_status)

if __name__ == '__main__':
    app.run(debug=True)
