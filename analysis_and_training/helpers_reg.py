## basic package
import pandas as pd
import numpy as np

## Visualisation packagers
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

## pkg data preparation
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

## pkg evaluation metric
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error


# REGRESSION

def model_select_reg(features, target, models):
    import numpy as np
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)
    MAE=[]
    MSE=[]
    R_Square=[]
    MSLE = []
    
    for model in models:
        clf=model[1]
        clf.fit(X_train, y_train)
        pred=clf.predict(X_test)
        mae = mean_absolute_error(y_test,pred)
        mse = mean_squared_error(y_test,pred)
        r_Square = r2_score(y_test,pred)
        msle = mean_squared_log_error(y_test,pred)
        
        MAE.append(mae)
        MSE.append(mse)
        R_Square.append(r_Square)
        MSLE.append(msle)

       
    res = pd.DataFrame({"MAE":MAE, "MSE":MSE,  "R_Square":R_Square, "MSLE":MSLE})
    return res


def regression_cv(X, y, algo_name, algorithm, gridsearchParams, cv):
    """
    Regression with cross-validation method
    """
    import numpy as np
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
    grid = GridSearchCV(
    estimator=algorithm,
    param_grid=gridsearchParams,
    cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    
    grid_model = grid.fit(X_train, y_train)
    best_params = grid_model.best_params_
    pred = grid_model.predict(X_test)
    predt = grid_model.predict(X_train)
    
    # Evaluation
    MAE = mean_absolute_error(y_test,pred)
    MSE = mean_squared_error(y_test,pred)
    R_Square = r2_score(y_test,pred)
    MSLE = mean_squared_log_error(y_test,pred)
    
    # metrcs = grid_result.gr
    # Save Model
    filename = 'models/'+algo_name+str(round(R_Square*100,2))+'.sav'
    pickle.dump(grid_model, open(filename, 'wb'))
    filename1 = 'models/'+algo_name+str(round(R_Square*100,2))+'.pkl'
    pickle.dump(grid_model, open(filename1,'wb'))
   # model=pickle.load(open('model.pkl','rb'))
    
    
    print('Best Params : ', best_params, '\n', '='*50)
    
    print("Model Evaluation :")
    print('Mean Absolute Error    :',MSE)
    print('Mean Squared Error     :',MSE)
    print('Mean Squared Log Error :',MSLE)
    print('R-Square               :',R_Square, '\n', '='*50)
 
    print("prediction: ")
    y_test =y_test.tolist()
    df2 = pd.DataFrame({'Actual':y_test,'Predicted':np.round(pred,2)})
    print(df2[:7].T)
    
def regression_ho(X, y, algo_name,algorithm):
    """
    Regression holdout method"""
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    model=algorithm.fit(X_train,y_train)
    pred=model.predict(X_test)
    
    # Evaluation
    MAE = mean_absolute_error(y_test,pred)
    MSE = mean_squared_error(y_test,pred)
    MSLE = mean_squared_log_error(y_test,pred)
    R_Square = r2_score(y_test,pred)
    
    # Save Model
    filename = 'models/'+algo_name+str(round(R_Square*100,2))+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    filename1 = 'models/'+algo_name+str(round(R_Square*100,2))+'.pkl'
    pickle.dump(model, open(filename1,'wb'))
   # model=pickle.load(open('model.pkl','rb'))
    
    print(f"{algo_name} Model Successfully Trainde and Saved as {filename}")
    print("****"*15)
    
    # Display metric
    print(f"{algo_name} Model Evaluation :")   
    print("Model Evaluation :")
    print('Mean Absolute Error    :',MSE)
    print('Mean Squared Error     :',MSE)
    print('Mean Squared Log Error :',MSLE)
    print('R-Square               :',R_Square, '\n', '='*50)
 
    print("prediction: ")
    y_test =y_test.tolist()
    df2 = pd.DataFrame({'Actual':y_test,'Predicted':np.round(pred,2)})
    print(df2[:7].T)
    
    # HYPERPARAMETER TUNING

def hyperparam_tuning(X_train, y_train, algo_name, algorithm, hyperparams, cv):
    grid = GridSearchCV(
    estimator=algorithm,
    param_grid=hyperparams,
    cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_model = grid.fit(X_train, y_train)
    best_params = grid_model.best_params_
    #print(f"{algo_name} best parameters are : {best_params}")
    return algo_name, best_params, cv


def hyperparam_tuning1(X,y, algos):
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], 
                           config['params'], 
                           cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


#-PREDICTION

def prediction(loaded_model):
    '''
        Accept features values from the user and classify on base of those features 
        if the iris flower spece (setosa, virginica, versicolor) 
    '''
    print("============WELCOME TO=============")
    print("        Diabetes Prediction     ")
    print("===================================")
    print(" ")
    # Load Model
    #model = pickle.load(open(model_path, "rb"))
    
    print("Please Enter Predictor variables")
    print("--------------------------------")
    l=[]
    l.append(float(input('Enter Pregnancies : ')))
    l.append(float(input('Enter Glucose  : ')))
    l.append(float(input('Enter BloodPressure : ')))
    l.append(float(input('Enter SkinThickness  : ')))
    l.append(float(input('Enter Insulin : ')))
    l.append(float(input('Enter BMI  : ')))
    l.append(float(input('Enter DiabetesPedigreeFunction : ')))
    l.append(float(input('Enter Age  : ')))
    print("--------------------------------")

    print("")      
    #arr = pd.DataFrame([l])
    #arr = np.asarray([l])
    model = loaded_model     
    pred_result = model.predict([l])
    if pred_result[0] == 0:
        result = "This Patient is Not Prediabetic"
    else:
        result = "This Patient is Prediabetic"
    print("===================================")
    print(f'Result : {result}')
    print("============THANK YOU=============")
    
def prediction2(loaded_model, features):
    #model = pickle.load(open(model_path, "rb"))
    model = loaded_model
    pred_result = model.predict([features])
    if pred_result[0] == 0:
       print("This Patient is Not Prediabetic")
    else:
        print("This Patient is Prediabetic")
