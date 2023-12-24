import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

# Create a Flask app
app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the saved scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Ensure that the JSON request contains all the required features
        required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Preprocess input data by scaling
        scaled_features = scaler.transform([[data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
                                             data['fbs'], data['restecg'], data['thalach'], data['exang'],
                                             data['oldpeak'], data['slope'], data['ca'], data['thal']]])

        # Make a prediction
        prediction = model.predict(scaled_features)

        # Convert the prediction to a regular Python int
        prediction = int(prediction[0])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
