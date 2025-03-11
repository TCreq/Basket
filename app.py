from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np



# modele
model = load("modele.joblib")

# model.predict(np.array([1.        , 0.80684597, 0.60638298, 0.68627451, 0.72222222,
#        0.66485753, 0.        , 0.01538462, 0.2       , 0.38961039,
#        0.43137255, 0.687     , 0.69811321, 0.4375    , 0.57553957,
#        0.22641509, 0.48      , 0.12820513, 0.68181818]).reshape(1,-1))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=False
    ) # J'ai préférer garder une API très simple pour la gestion de mon temps et rester pertinent pour la consigne.




