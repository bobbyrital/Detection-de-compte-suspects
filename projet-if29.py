import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pprint import pprint
from datetime import datetime
from sklearn.cluster import DBSCAN,KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import seaborn as sns
import time

x = ""
while x!="oui" and x!= "non":
    print("Effectuer la requete sur mongoDB(durée estimée de 10min)? (oui/non)")
    print("Si oui, il est surement nécessaire de modifier le nom de la base de donnée et de la collection dans le code.")
    x = input()

if x == "oui":

    # À modifier en changeant le nom de la base de données et de la collection
    tweets = MongoClient().tw_wc.tweets
    print("Affichage d'un tweet de la base de données")
    pprint(tweets.find_one())

    pipeline = [
        {
            '$project': {
                'id': '$user.id', 
                'name': '$user.screen_name', 
                'age': {
                    '$trunc': {
                        '$divide': [
                            {
                                '$subtract': [
                                    datetime.utcnow(), {
                                        '$dateFromString': {
                                            'dateString': '$user.created_at'
                                        }
                                    }
                                ]
                            }, 1000 * 60 * 60
                        ]
                    }
                }, 
                'text': 1, 
                'suspect': {
                    '$toInt': {
                        '$regexMatch': {
                            'input': '$text', 
                            'regex': "(download|stream|streaming|free |live |online|bet |blockchain|bitcoin|casino|cash|money|rich|freepick|code|token|clx|fpx|only in|coin|btc|trading|crypto|payement|mining|miner|generator|robot|\\$\\$|€€|invest|investment|pay me)"
                        }
                    }
                }, 
                'entities': 1, 
                'friends_count': '$user.friends_count', 
                'urls_count': {
                    '$toInt': {
                        '$gt': [
                            {
                                '$size': '$entities.urls'
                            }, 1
                        ]
                    }
                }
            }
        }, {
            '$group': {
                '_id': '$id', 
                'nom': {
                    '$max': '$name'
                }, 
                'age': {
                    '$min': '$age'
                }, 
                'mentions_avg': {
                    '$avg': {
                        '$size': '$entities.user_mentions'
                    }
                }, 
                'hashtag_avg': {
                    '$avg': {
                        '$size': '$entities.hashtags'
                    }
                }, 
                'tweets_urls': {
                    '$sum': '$urls_count'
                }, 
                'tweets_count': {
                    '$sum': 1
                }, 
                'tweets_suspects': {
                    '$sum': '$suspect'
                }, 
                'tweets_differents': {
                    '$addToSet': '$text'
                }, 
                'all_tweets': {
                    '$push': '$text'
                }, 
                'friends_count': {
                    '$max': '$friends_count'
                }
            }
        }, {
            '$addFields': {
                'spam': {
                    '$divide': [
                        {
                            '$subtract': [
                                {
                                    '$size': '$all_tweets'
                                }, {
                                    '$size': '$tweets_differents'
                                }
                            ]
                        }, '$tweets_count'
                    ]
                }, 
                'danger': {
                    '$divide': [
                        {
                            '$add': [
                                '$tweets_suspects', '$tweets_urls'
                            ]
                        }, '$tweets_count'
                    ]
                }, 
                'aggressiveness': {
                    '$divide': [
                        {
                            '$sum': [
                                {
                                    '$divide': [
                                        '$tweets_count', '$age'
                                    ]
                                }, {
                                    '$divide': [
                                        '$friends_count', '$age'
                                    ]
                                }
                            ]
                        }, 350
                    ]
                }, 
                'visibility': {
                    '$divide': [
                        {
                            '$sum': [
                                '$hashtag_avg', '$mentions_avg'
                            ]
                        }, 12.174
                    ]
                }
            }
        }, {
            '$project': {
                'all_tweets': 0, 
                'tweets_count': 0, 
                'friends_count': 0, 
                'age': 0, 
                'mentions_avg': 0, 
                'hashtag_avg': 0, 
                'tweets_urls': 0, 
                'tweets_suspects': 0
            }
        }
    ]

    t1 = time.time()
    #Lancement de la requete
    data = list(tweets.aggregate(pipeline,allowDiskUse=True))
    t2 = time.time()
    print(t2-t1)

    #Conversion de pymongo à pandas
    data = pd.io.json.json_normalize(data)
    print("Exemple de données:\n",data.head())


def plot3d(data,categorie,titre,n_lignes=None):
    fig = plt.figure()
    fig.suptitle(titre)
    ax = fig.add_subplot(111, projection='3d')
    if n_lignes==None:
        ax.scatter(data[:,0],data[:,3],data[:,1],c=categorie[:],cmap='bwr',alpha=1)
    else:
        ax.scatter(data[:n_lignes,0],data[:n_lignes,3],data[:n_lignes,1],c=categorie[:n_lignes],cmap='bwr',alpha=1)
    ax.set_xlabel('aggressiveness')
    ax.set_ylabel('spam')
    ax.set_zlabel('danger')
    ax.invert_yaxis()
    plt.show()

####################### Jeu de données des 2000 tweets classifié à la main #######################

print("Dataset de 2000 tweets:")
#Chargment du jeu de données avec 2000 tweets
data2000 = pd.read_csv("dataset2000.csv")

# Division des attributs pour supprimer ceux inutile pour l'entrainement
id_ = data2000[["_id","tweets_differents","nom"]]
categorie = data2000.suspect

data2000 = data2000.drop(["tweets_differents","_id","nom"],axis=1)

sns.pairplot(data2000,hue="suspect",corner=True)
plt.show()

data2000 = data2000.drop("suspect",axis=1)

# Normalisation des données
data2000_np = StandardScaler().fit_transform(data2000).astype(np.float32)

pca = PCA()
pca.fit(data2000_np)
print("\nPourcentage expliqué de chaque attribut de l'acp:\n",pca.explained_variance_ratio_)

plot3d(data2000_np,categorie,"Données classifiées à la main")

######### Entrainement et évaluation des modèles de clusterings #########

#Kmeans
t = time.time()
#Entrainement du modèle
cat = KMeans(2,random_state=111).fit_predict(data2000_np)
print("\nKmeans:")
print("Temps:",time.time()-t)
print("Matrice de confusion:\n",confusion_matrix(categorie,cat))
plot3d(data2000_np,cat,"Kmeans")

t = time.time()
#Entrainement du modèle
cat = DBSCAN(1.5,30).fit_predict(data2000_np)*-1
print("\nDBSCAN:")
print("Temps:",time.time()-t)
print("Matrice de confusion:\n",confusion_matrix(categorie,cat))
plot3d(data2000_np,cat,"DBSCAN")

######### Entrainement et évaluation des modèles de détections d'anomalies #########
for model,nom in zip([LocalOutlierFactor(),EllipticEnvelope(contamination=0.03,random_state=42),IsolationForest(contamination=0.03,random_state=42)],["Local Outlier Factor","Elliptic Envelope","Isolation Forest"]):
    t = time.time()
    #Entrainement du modèle
    cat = model.fit_predict(data2000_np)*-1
    cat = (cat+1)/2
    print("\n"+nom+":")
    print("Temps:",time.time()-t)
    print("Matrice de confusion:\n",confusion_matrix(categorie,cat))
    plot3d(data2000_np,cat,nom)

#Identification des faux positifs
suspects = id_.iloc[np.where((cat==1)&(categorie==0))]
print("\nExemple de faux positifs suspects:")
for i in [20,21,28]:
    print("\n"+suspects.tweets_differents.iloc[i])


######### Entrainement et évaluation du modèle supervisé #########

#Séparation en jeu d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(data2000_np, cat, test_size=0.30)

t = time.time()
#Entrainement du modèle
svm = SVC(random_state=42)
svm.fit(X_train[:],y_train[:])
print("\nSVM:")
print("Temps",time.time()-t)

prd = svm.predict(X_test)
print("Matrice de confusion en comparaison avec le modèle non-supervisé:\n",confusion_matrix(y_test,prd))

prd2 = svm.predict(data2000_np)
print("\nMatrice de confusion en comparaison avec les données manuelles:\n",confusion_matrix(categorie,prd2))
plot3d(data2000_np,prd2,"SVM")



####################### Jeu de données des 4.6 millions de tweets #######################

print("\nDataset de 4.6 millions de tweets")

# Chargment de l'ensemble des données
data = pd.read_csv("data.csv")
id_ = data[["_id","nom"]]
data = data.drop(["_id","nom"],axis=1)
data_np = StandardScaler().fit_transform(data).astype(np.float32)

x = ""
while x!="oui" and x!= "non":
    print("Réentrainer l'Isolation Forest (durée estimée de 1min20)? (oui/non)")
    x = input()

if x == "oui":
    print("\nEntrainement de l'Isolation Forest (temps estimé de 1min20):")
    t = time.time()
    #Entrainement du modèle
    cat = model.fit_predict(data2000_np)*-1
    cat = (cat+1)/2
    print("\n"+IsolationForest+":")
    print("Temps:",time.time()-t)
else:
    cat = pd.read_csv("IF_pred.csv").suspect

#Affichage de 10 000 points sur 1.8 millions pour réduire le temps d'affichage
plot3d(data_np,cat,"Isolation Forest",10000)

x = ""
while x!="oui" and x!= "non":
    print("Réentrainer le SVM (durée estimée de 30min)? (oui/non)")
    x = input()

if x == "oui":
    print("\nEntrainement du SVM (temps estimé de 30min)")
    #Séparation en jeu d'entrainement et de test (seulement 2% car il y a beaucoup de données)
    X_train, X_test, y_train, y_test = train_test_split(data_np, cat, test_size=0.02)
    t = time.time()
    svm = SVC()
    #Entrainement du modèle
    svm.fit(X_train[:],y_train[:])
    print("Temps",time.time()-t)
    prd = svm.predict(X_test)
    print("Matrice de confusion du Test set")
    print(confusion_matrix(y_test,prd))
    prd2 = svm.predict(data_np)

else:
    prd2 = pd.read_csv("SVM_pred.csv").suspect


print("\nMatrice de confusion en comparaison avec le modèle non-supervisé")
print(confusion_matrix(cat,prd2))
#Affichage de 10 000 points sur 1.8 millions pour réduire le temps d'affichage
plot3d(data_np,prd2,"SVM",10000)
