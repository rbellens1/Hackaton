import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import statsmodels.api as sm
#statsmodels.stats.weightstats.DescrStats
import os




np.random.seed(0)  # Pour la reproductibilité

X1 = np.random.rand(10) * 10  # Première colonne (valeurs aléatoires de 0 à 10)
X2 = np.random.rand(10) * 5    # Deuxième colonne (valeurs aléatoires de 0 à 5)
data = np.column_stack((X1, X2))  # Combiner en une matrice 10x2

# Créer une relation linéaire avec un peu de bruit
Y = 1.5 * X1 + 2.0 * X2 + 5 + np.random.normal(0, 1, data.shape[0])

n    = data.shape[0]  #nombres de data

def Regression(data, Y, n):
    """
    inputs:
    data: numpy array de longueur 7*n contenat les données
    Y: numpy array de longueur 7*1 contenant les valeurs de PM2.5
    n : nombre de données pour chaque variable
    
    return:
    Y numpy array contenant la meilleure aproximation de y en régression linéaire
    data_cst : tableau numpy étant la matrice data + la constante
    Results.summary() : résultats de la régression
    beta_array : array numpy contenant les valeurs des betas
    
    """
    data = sm.add_constant(data)
    Results = sm.OLS(Y, data).fit()
    beta_array = Results.params
    Y_estimator = Results.predict(data)
    return (Results.summary(), Y_estimator, data, beta_array)

summary, Y_estimator, data_cst, Beta = Regression(data, Y, n)


Y_mean = 0
for i in Y:
    Y_mean += i/len(Y)

def Goodness_of_fit(Y, Y_estimator, Y_mean):
    
    SSE = np.dot(Y-Y_estimator,Y-Y_estimator)
    SSR = np.dot(Y_estimator-Y_mean, Y_estimator-Y_mean)
    R2 = SSE/SSR
    return R2

print(Goodness_of_fit(Y, Y_estimator, Y_mean))


X = data_cst[:,1]
Y_1 = X*Beta[1]


plt.scatter(X, Y, color='blue', label='Données d\'origine')  # Points d'origine
plt.plot(X, Y_1, color='red', label='Droite de régression')  # Droite de régression
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression Linéaire')
plt.legend()
plt.grid()
plt.show()


  