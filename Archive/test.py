import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import statsmodels.api as sm
from joblib import dump, load


def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=3
    print(confusion_mat)
    print(recall)
    return recall #pour la recherche


# Load dataset
df = pd.read_csv(".\\nba_logreg.csv")

# extract names, labels, features names and values
names = df['Name'].values.tolist() # players names
labels = df['TARGET_5Yrs'].values # labels
paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values
# type(df_vals)

# replacing Nan values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x]=0.0

# normalize dataset
X = MinMaxScaler().fit_transform(df_vals)

#example of scoring with support vector classifier
score_classifier(X,SVC(),labels)



###############################################################

# TODO build a training set and choose a classifier which maximize recall score returned by the score_classifier function
# X, labels

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)


###############################################################
# Liste des hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

best_score = -float("inf")
best_params = None

# Exploration de paramètres
for n in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        for min_samples in param_grid['min_samples_split']:
            classifier = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_split=min_samples, random_state=42)
            
            score = score_classifier(X_train, classifier, y_train) 
            
            # Affichage du score pour chaque combinaison
            print(f"n_estimators={n}, max_depth={depth}, min_samples_split={min_samples} --> Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = {'n_estimators': n, 'max_depth': depth, 'min_samples_split': min_samples}

# Résultat final
print("\nMeilleurs paramètres :", best_params)
# n_estimators=50, max_depth=None, min_samples_split=5
print("Meilleur score :", best_score)
# 0.7972
score_classifier(X,RandomForestClassifier(random_state=42),labels)
# 0.7951
classifier = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=5, random_state=42)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)
##############################################################
# Liste des hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],      # Nombre d'arbres
    'max_depth': [3, 6, 10],             # Profondeur max des arbres
    'learning_rate': [0.01, 0.1, 0.2],   # Taux d'apprentissage
    'subsample': [0.8, 1.0]              # Pourcentage d'échantillons utilisés par arbre
}
best_score = -float("inf")
best_params = None

# Exploration de paramètres
for n in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                classifier = XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=subsample, random_state=42, eval_metric='logloss')
                
                score = score_classifier(X_train, classifier, y_train) 
                
                # Affichage du score pour chaque combinaison
                print(f"n_estimators={n}, max_depth={depth}, learning_rate={lr}, subsample={subsample} --> Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n, 'max_depth': depth, 'learning_rate': lr, 'subsample': subsample}

# Résultat final
print("\nMeilleurs paramètres :", best_params)
# n_estimators=50, max_depth=3, learning_rate=0.01, subsample=0.8
print("Meilleur score :", best_score)
# 0.9686
score_classifier(X,XGBClassifier(random_state=42),labels)
# 0.7652
classifier = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.01, subsample=0.8, random_state=42)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)
##############################################################
# Liste des hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],      # Nombre d'arbres
    'max_depth': [3, 6, 10],             # Profondeur max des arbres
    'learning_rate': [0.01, 0.1, 0.2],   # Taux d'apprentissage
    'subsample': [0.8, 1.0]              # Pourcentage d'échantillons utilisés par arbre
}
best_score = -float("inf")
best_params = None

# Exploration de paramètres
for n in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                classifier = GradientBoostingClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=subsample, random_state=42)
                
                score = score_classifier(X_train, classifier, y_train) 
                
                # Affichage du score pour chaque combinaison
                print(f"n_estimators={n}, max_depth={depth}, learning_rate={lr}, subsample={subsample} --> Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n, 'max_depth': depth, 'learning_rate': lr, 'subsample': subsample}

# Résultat final
print("\nMeilleurs paramètres :", best_params)
# n_estimators=50, max_depth=3, learning_rate=0.01, subsample:0.8
print("Meilleur score :", best_score)
# 0.9568
score_classifier(X,GradientBoostingClassifier(random_state=42),labels)
# 0.7916
classifier = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.01, subsample=0.8, random_state=42)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)


##########################################################
#example of scoring with support vector classifier
#score_classifier(X,SVC(),labels)
classifier = SVC(random_state=42)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)

##########################################################
#comparaison avec le dummy classifier
classifier = DummyClassifier(strategy="most_frequent")
classifier.fit(X_train,y_train)
# score_classifier(X,classifier,labels)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)

##########################################################
#Regression Logistique
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
# score_classifier(X,classifier,labels)
y_pred=classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(accuracy)
print(recall)
print(f1)


#logreg_stats = sm.Logit(y_train, X_train).fit()
print(paramset)
X_train1 = pd.DataFrame(X_train, columns=paramset)
classifier = sm.Logit(y_train,X_train1)
stats = classifier.fit()
print(stats.summary())















dump(classifier, "modele.joblib")

model = load("modele.joblib")
