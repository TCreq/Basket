from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np



# modele
model = load("Depots/TestTechnique/modele.joblib")



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        # return jsonify({"error": str(e)})
        pass



if __name__ == '__main__':
    app.run(debug=True)




