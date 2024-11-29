"""
CELL N°1.1 : IMPORTING ALL THE NECESSARY PACKAGES

@pre:  /
@post: The necessary packages should be loaded.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import *
from sklearn import datasets #
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


"""
CELL N°1.2 : IMPORTING THE DATASET

@pre:  /
@post: The object `df` should contain a Pandas DataFrame corresponding to the file `diabetes_dataset.csv`
"""
df = pd.read_csv("diabetes_dataset.csv")

#df.info()
#df.describe()


"""
CELL N°1.3 : IS THE DATASET BALANCED?

@pre:  `df` contains the dataset
@post: Plot the diabetic/non-diabetic distribution in a pie chart
"""

diabetes_counts = df['Diabetes'].value_counts()

# Création du camenbert
labels = ['Non-Diabétique', 'Diabétique']
colors = ['chartreuse', 'red']
plt.figure(figsize=(6, 6))
plt.pie(diabetes_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Répartition des individus selon le diabète')
plt.show()


"""
CELL N°1.4 : SCALE THE DATASET

@pre:  A pandas.DataFrame `df` containing the dataset
@post: A pandas.DataFrame `df` containing the standardized dataset (except classification columns (Diabetes))

"""
def scale_dataset(df): 
    features = df.drop(columns =["Diabetes"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    scaled_df["Diabetes"] = df ["Diabetes"]
    scaled_df = scaled_df[["Diabetes"] + [col for col in scaled_df.columns if col != "Diabetes"]]
    return scaled_df

df = scale_dataset(df)
#df.info()
#df.describe()


"""
CELL N°2.1 : CORRELATION MATRIX

@pre:  `df` contains the diabetes dataset
@post: `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the full dataset
"""

### Calculate correlation matrix

# Calculate "pearson" correlation matrix
corr_matrix = df.corr(method='pearson') 
#corr_matrix.info()
#corr_matrix.describe()


### Plot with seaborn heatmap

# Plot & costumisation
sns.heatmap(corr_matrix, cmap="Blues", annot=False, square=True, )


#Tutorial: https://www.geeksforgeeks.org/create-a-correlation-matrix-using-python/


"""
CELL N°2.2 : ANALYZE THE CORRELATION WITH DIABETE

@pre:  `corr_matrix` is a Pandas DataFrame that contains the correlation matrix of the training set
@post: `sorted_features` contains a list of features (columns of `df`) 
       sorted according to their correlation with `Diabetes` 
"""

def sort_features(corr_matrix):
    # Sort the correlation matrix with respect to the target variable
    sorted_corr_matrix = corr_matrix["Diabetes"].abs().sort_values(ascending=False)
    #print(sorted_corr_matrix)
    return(sorted_corr_matrix)

#Tutorial: https://www.geeksforgeeks.org/sort-correlation-matrix-in-python/

sorted_features = sort_features(corr_matrix)


"""
CELL N°3.1 : LINEAR REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

def linear_regressor(X_train, y_train, threshold = 0.5):

    model = LinearRegression().fit(X_train, y_train)
    
    return lambda X_test : (model.predict(X_test)>=threshold).astype(int)


"""
CELL N°3.2 : LOGISTIC REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post:  Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

def logistic_regressor(X_train, y_train, threshold = 0.5):
    model = LogisticRegression().fit(X_train, y_train)
    
    return lambda X_test : (model.predict_proba(X_test)[:,1] > threshold).astype(int)


"""
CELL N°3.3 : KNN REGRESSOR

@pre:  `X_train` and `y_train` contain the training set of `df` and a threshold that is a numerical value (float) by default 0.5.
@post: Lambda function that takes `X_test` and returns a 1D array of binary predictions (0 or 1) according to the given threshold.
"""

def knn_regressor(X_train, y_train, threshold = 0.5, n_neighbors = 10):
    model = KNeighborsRegressor(n_neighbors).fit(X_train, y_train)
    
    return lambda X_test : (model.predict(X_test)>=threshold).astype(int)


"""
CELL N°4.1 : PRECISION SCORE

@pre:  /
@post: `precision(y_test, y_pred)` returns the prediction metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""

def precision(y_test, y_pred):
    predict_true=0
    real_true=0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]==1:
            real_true+=1
        if y_pred[i]==1:
              predict_true+=1
    if predict_true==0: return 0
    return real_true/predict_true


"""
CELL N°4.2 : RECALL SCORE

@pre:  /
@post: `recall(y_test, y_pred)` returns the recall metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""

def recall(y_test, y_pred):
    real_true=0
    all_true=0
    for k in range(len(y_test)):
        if y_test[k]==y_pred[k]==1:
            real_true+=1
        if y_test[k]==1:
            all_true+=1
    if all_true==0: return 0
    return real_true/all_true


"""
CELL N°4.3 : F1 SCORE

@pre:  /
@post: `f1_score(y_test, y_pred)` returns the F1 score metric based on the predicted labels `y_pred`
       and the true labels `y_test`. 
"""

def f1_score(y_test, y_pred):
    r=recall(y_test,y_pred)
    p=precision(y_test,y_pred)
    if(r==p==0): return 0
    return 2/((r)**(-1)+(p)**(-1))


"""
CELL N°5.1 : K-FOLD PREPARATION

@pre:  `df` contains the scaled dataset.
@post: The following specifications should be satisfied: 
            - `kf` should contain a `KFold` object with 3 splits, shuffled and with 1109 seed. 
            - `X` should contain a pd.DataFrame with all the features (all columns except `Diabetes`)
            - `y` should contain a pd.DataFrame with all the labels (only the column `Diabetes`)
"""

kf = KFold(n_splits = 3,shuffle=True,random_state=1109)

X = df.drop(columns=["Diabetes"]) #remove the column "Diabetes"
y = df["Diabetes"] #keep only the column "Diabetes"


"""
CELL N°5.2 : FIND THE RIGHT COMBINATION LENGTH/REGRESSOR

@pre:  `kf`, `X` and `y` are defined such as in the @post of CELL 5.1. 
@post: `result` is such that `result[reg][i]` contains the average of the validations for regressor `reg`, 
       when keeping the `i` most correlated features
"""

def validation(regressor, X_test, y_test):
    y_pred = regressor(X_test)
    return (recall(y_test, y_pred), precision(y_test, y_pred), f1_score(y_test, y_pred))

threshold = 0.5
result = {
    "linear":{}, 
    "logistic":{}, 
    "knn":{}
}

y_arr = np.array(y)
corr=df.corr()["Diabetes"].drop("Diabetes").abs() #correlations between "Diabetes" and the other features

for i in np.arange(1,X.shape[1]+1,dtype=int): #faire pour different nombre de collonne choisie (attention pas faire i=0)
    
    print(i) #to know where the program is
    
    corLargest=corr.nlargest(n=i) #keep only i features with the biggest correlation
    iMostCor=df[corLargest.index.tolist()] #i features most correlated
    iMostCorArr=np.array(iMostCor)
    
    val=np.zeros((3,3,3)) #help to compute the mean
    j=0
    
    for train_index, test_index in kf.split(iMostCor):
        
        X_train, X_test = iMostCorArr[train_index, :], iMostCorArr[test_index, :]
        y_train, y_test = y_arr[train_index].ravel(), y_arr[test_index].ravel()
        
        #create the regressors
        linReg=linear_regressor(X_train,y_train,threshold)
        logReg=logistic_regressor(X_train,y_train,threshold)
        knnReg=knn_regressor(X_train,y_train,threshold)
        
        val[0,j]=validation(linReg, X_test, y_test)
        val[1,j]=validation(logReg, X_test, y_test)
        val[2,j]=validation(knnReg, X_test, y_test)
        
        j+=1

    result["linear"][i]=np.sum(val[0], axis=0)/3
    result["logistic"][i]=np.sum(val[1], axis=0)/3
    result["knn"][i]=np.sum(val[2], axis=0)/3


"""
CELL N°5.3 : VISUALIZE THE SCORES

@pre:  `result` contains the average of the validations for regressor `reg`, when keeping the `i` most correlated features
@post: plot of the scores for each condition
"""

print(result)

from helper import plot_result
plot_result(result, threshold, to_show = "recall")
plot_result(result, threshold, to_show = "f1_score")


"""
CELL N°6.1 : VISUALIZE YOUR RESULTS

@pre:  /
@post: /
"""

X = df.drop(columns=["Diabetes"]) 
y = df["Diabetes"]                


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1109)

model = LogisticRegression()
model.fit(X_train, y_train)


beta = model.coef_.flatten()   
intercept = model.intercept_[0]  

beta_hat_dot_X_test = np.dot(X_test, beta) + intercept 
y_test = np.array(y_test)  


threshold = 0.22 # A modifier
predicted = beta_hat_dot_X_test >= threshold  


tp = (predicted == 1) & (y_test == 1)  
tn = (predicted == 0) & (y_test == 0)  
fp = (predicted == 1) & (y_test == 0)  
fn = (predicted == 0) & (y_test == 1) 


sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

plt.scatter(beta_hat_dot_X_test[tp], y_test[tp], color='green', alpha=0.7, edgecolor='k', label="True Positives (TP)")
plt.scatter(beta_hat_dot_X_test[tn], y_test[tn], color='blue', alpha=0.7, edgecolor='k', label="True Negatives (TN)")
plt.scatter(beta_hat_dot_X_test[fp], y_test[fp], color='orange', alpha=0.7, edgecolor='k', label="False Positives (FP)")
plt.scatter(beta_hat_dot_X_test[fn], y_test[fn], color='red', alpha=0.7, edgecolor='k', label="False Negatives (FN)")

plt.fill_betweenx(
    y=[min(y_test) - 0.5, max(y_test) + 0.5],
    x1=min(beta_hat_dot_X_test) - 0.5,
    x2=threshold,
    color="lightblue",
    alpha=0.2,
    label="Negative Region"
)
plt.fill_betweenx(
    y=[min(y_test) - 0.5, max(y_test) + 0.5],
    x1=threshold,
    x2=max(beta_hat_dot_X_test) + 0.5,
    color="lightcoral",
    alpha=0.2,
    label="Positive Region"
)

plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label="Threshold ($\\tau = 0.22$)")


plt.title(r"Scatter Plot for Logistic Regression Predictions", fontsize=16)
plt.xlabel(r"$\hat{\beta}^\top x_i$", fontsize=14)
plt.ylabel("Diabetes Prediction", fontsize=14)
plt.legend(fontsize=12, loc="upper left")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()