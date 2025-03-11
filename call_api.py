import requests
import pandas as pd
import json
import numpy as np


X_test = pd.DataFrame(np.load("X_test.npy"))

# API Flask
API_URL = "http://127.0.0.1:5000/predict"


# Boucle sur chaque ligne de X_test
predictions = []
for _, row in X_test.iterrows():
    features = list(row)
    print(features)

    # On appelle l'API avec une requete POST, il faut fournir dans le json les variables (sous forme de liste, avec pour clé : "features")
    response = requests.post(API_URL, json={"features":features})  # requête POST
    
    if response.status_code == 200:
        pred = response.json().get("prediction")  # Récupère la prédiction
        predictions.append(pred)
        print(f"Entrée: {data} → Prédiction: {pred}")
    else:
        print(f"Erreur pour {data}: {response.text}")





##########################
# Lignes de commandes pour tester en local

# Linux (pas testé directement car je n'ai pas Linux sur ma Machine)
# curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{\"features\":[1.,0.80684597,0.60638298,0.68627451,0.72222222,0.66485753,0.,0.01538462,0.2,0.38961039,0.43137255,0.687,0.69811321,0.4375,0.57553957,0.22641509,0.48,0.12820513,0.68181818]}'

# Powershell
# Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"features":[1.0,0.80684597,0.60638298,0.68627451,0.72222222,0.66485753,0.0,0.01538462,0.2,0.38961039,0.43137255,0.687,0.69811321,0.4375,0.57553957,0.22641509,0.48,0.12820513,0.68181818]}'
