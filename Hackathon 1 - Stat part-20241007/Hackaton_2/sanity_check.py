import ast
import os, sys

import pandas as pd
import numpy as np

# Argument handling
class IllegalArgumentError(ValueError):
    pass
if len(sys.argv) != 2:
    raise IllegalArgumentError("You must provide only one argument with the name of your .py file.")
_, input_filename = sys.argv
if input_filename.split(".")[-1] != "py":
    raise IllegalArgumentError("Your input file is not a .py file.")

# Create tmp file
tmp_filename = "tmp.py"
with open(input_filename, 'r') as f: 
    tree = ast.parse(f.read(), filename=input_filename)
    relevant = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
try: 
    with open(tmp_filename, 'w') as f:
        for r in relevant: 
            if "plot_result" in ast.unparse(r):
                continue
            f.write(ast.unparse(r) + "\n\n") 
except Exception as e:
    os.remove(tmp_filename)
    raise e

try: 
    exec(open(tmp_filename).read())
except Exception as e:
    os.remove(tmp_filename)
    raise e
try: 
    df = pd.read_csv("diabetes_dataset.csv")
except Exception as e:
    os.remove(tmp_filename)
    raise e

########################## 
# Test 1 "scale_dataset" #
########################## 
try: 
    df_scaled = scale_dataset(df)
except Exception as e: 
    os.remove(tmp_filename)
    raise e
if not isinstance(df_scaled, pd.DataFrame):
    os.remove(tmp_filename)
    raise ValueError("Function `scale_dataset` must return a pd.DataFrame.")
print("[V] Test 1 `scale_dataset` passed")

########################## 
# Test 2 "sort_features" #
########################## 
corr_matrix = df.corr(method='pearson')
try: 
    features_list = sort_features(corr_matrix)
except Exception as e:
    os.remove(tmp_filename)
    raise e
if not isinstance(features_list, list):
    os.remove(tmp_filename)
    raise ValueError("Function `features_list` must return a list.")
print("[V] Test 2 `sort_features` passed")

#######################################################################
# Test 3 "linear_regressor", "logistic_regressor" and "knn_regressor" #
#######################################################################
y = df_scaled['Diabetes']
y = y.sample(frac = 0.01, random_state=1109)
X = df_scaled.drop(['Diabetes'],axis=1)
X = X.sample(frac = 0.01, random_state=1109)

try: 
    reg_functions = [linear_regressor, logistic_regressor, knn_regressor]
except Exception as e:
    os.remove(tmp_filename)
    raise e

for reg in reg_functions:
    try: 
        f = reg(X, y, threshold = 0.2)
    except Exception as e:
        os.remove(tmp_filename)
        raise e
    if not callable(f):
        os.remove(tmp_filename)
        raise ValueError("Function `{}` must return a callable function f(X_test).".format(reg.__name__))
    
    try: 
        f_ret = f(X)    
    except Exception as e: 
        os.remove(tmp_filename)
        raise e
    if not isinstance(f_ret, np.ndarray):
        raise ValueError("The output of `{}` must return a np.ndarray.".format(reg.__name__))
print("[V] Test 3 `linear_regressor`, `logistic_regressor`, `knn_regressor` passed")

###############################################
# Test 4 "recall", "precision" and "f1_score" #
###############################################
y_test = np.array([1., 1., 1.])
y_pred = np.array([0., 1., 0.])
try: 
    val_functions = [recall, precision, f1_score]
except Exception as e:
    os.remove(tmp_filename)
    raise e

for val in val_functions:
    try: 
        value = val(y_test, y_pred)
    except Exception as e:
        os.remove(tmp_filename)
        raise e
    if not isinstance(value, float):
        os.remove(tmp_filename)
        raise ValueError("Function `{}` must return a float.".format(val.__name__))
print("[V] Test 4 `recall`, `precision` and `f1_score`")

print("All tests passed.")
os.remove(tmp_filename)
