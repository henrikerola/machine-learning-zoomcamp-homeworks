from flask import Flask, request, jsonify
import pickle

# Path to the pickled file
dv_path = './dv.bin'
model_path = './model2.bin'

# Load the DictVectorizer
with open(dv_path, 'rb') as f:
    dv = pickle.load(f)

# Load the LogisticRegression model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def example_endpoint():
    data = request.get_json()

    transformed_data = dv.transform(data)

    predictions = model.predict(transformed_data)
    probability = model.predict_proba(transformed_data)[:, 1][0]
    
    return jsonify({
        'propability': probability
    })