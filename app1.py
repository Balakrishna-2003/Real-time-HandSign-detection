# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Label mapping
label_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'
}
@app.route('/', methods=['GET'])
def hello():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("hello")
        print(data)
        # Expecting 'landmarks' to be a flat list: [x0, y0, x1, y1, ..., x20, y20]
        landmarks = data.get('landmarks', [])
        
        # print("hellow ",landmarks.type)
        if not landmarks or len(landmarks) != 42:
            return jsonify({'error': 'Invalid landmark data. Expected 42 values (21 x,y pairs).'}), 400

        # Convert to numpy array and reshape
        input_data = np.array(landmarks).reshape(1, -1)

        # Predict[np.asarray(input_data)]
        prediction = model.predict([np.asarray(landmarks)])
        predicted_label = int(prediction[0])
        predicted_character = label_dict.get(predicted_label, "Unknown")

        return jsonify({'prediction': predicted_character})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
