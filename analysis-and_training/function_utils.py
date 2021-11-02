## basic package
import pandas as pd
import numpy as np

## Visualisation packagers
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

## pkg data preparation
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

## pkg evaluation metric
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error

import pickle

##--CLASSIFICATION CROSS-VALIDATION METHOD--####################################

def clf_m_train_cv(X, y, algo_name, algorithm, gridsearchParams, cv):
    import numpy as np
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
    grid = GridSearchCV(
    estimator=algorithm,
    param_grid=gridsearchParams,
    cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_model = grid.fit(X_train, y_train)
    best_params = grid_model.best_params_
    pred = grid_model.predict(X_test)
    predt = grid_model.predict(X_train)
    
    clf_report = classification_report(y_test,pred)
    train_acc = accuracy_score(y_train, predt)
    test_acc = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    
    # metrcs = grid_result.gr
    ## Save model
    # metrcs = grid_result.gr
    #pickle.dump(grid_result, open(algo_name, 'wb'))
    filename = 'models/'+algo_name+str(round(train_acc*100,2))+'.sav'
    pickle.dump(grid_model, open(filename, 'wb'))
    filename1 = 'models/'+algo_name+str(round(train_acc*100,2))+'.pkl'
    pickle.dump(grid_model, open(filename1,'wb'))
    ## load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    
    
    print('Best Params : ', best_params, '\n', '='*50)
    print('Classification Report :', clf_report, '\n', '='*50)
    print('Accuracy Score train : ' + str(train_acc))
    print('Accuracy Score test  : ' + str(test_acc), '\n', '='*50)
    print('Confusion Matrix : \n', conf_mat, '\n', '='*50)
    
    print("Prediction :")
    y_test =y_test.tolist()
    df2 = pd.DataFrame({'Actual':y_test,'Predicted':pred})
    return df2
    
    
##--CLASSIFICATION HOLDOU METHOD--################################################

def clf_m_tain(X, y,algo_name,  algorithm):
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
    #algo = algo
    model = algorithm.fit(X_train,y_train)
    #estimator=algorithm,
    pred = model.predict(X_test)
    predt = model.predict(X_train)
    
    clf_report = classification_report(y_test,pred)
    train_acc = accuracy_score(y_train, predt)
    test_acc = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    
    ## Save model
    # metrcs = grid_result.gr
    #pickle.dump(grid_result, open(algo_name, 'wb'))
    filename = 'models/'+algo_name+str(round(train_acc*100,2))+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    filename1 = 'models/'+algo_name+str(round(train_acc*100,2))+'.pkl'
    pickle.dump(model, open(filename1,'wb'))
    ## load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
   
    #print('Best Params : ', best_params, '\n', '='*50)
    print('Classification Report :', clf_report, '\n', '='*50)
    print('Accuracy Score train : ' + str(train_acc))
    print('Accuracy Score test  : ' + str(test_acc), '\n', '='*50)
    print('Confusion Matrix : \n', conf_mat, '\n', '='*50)
    
    print("Prediction :")
    y_test =y_test.tolist()
    df2 = pd.DataFrame({'Actual':y_test,'Predicted':pred})
    #print(df2[:7].T)
    return loaded_model, df2

##--REGRESSION CROSS-VALIDATION METHOD--###########################################
def reg_m_train_cv(X, y, algo_name, algorithm, gridsearchParams, cv):
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
    
 ##--REGRESSION HOLDOUT METHOD--##################################################

def reg_m_train(X, y, algo_name,algorithm):
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
    
##--FEATURE IMPORTANCE--###########################################################
   
def feature_imp(features, model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature Ranking :" )
    features_name = []
    rankins_score = []
    for f in range(features.shape[1]):
       # print(" %s =>  (%f)  " %(list(X)[f], importances[indices[f]]))
        features_name.append(list(features)[f])
        rankins_score.append(importances[indices[f]])
    fimp = pd.DataFrame({'Features Name' : features_name,'Ranking Score' : rankins_score})
    fimp['%'] = round(fimp['Ranking Score']*100,2)
    
    plt.rcParams['figure.figsize'] = 5,5
    sns.set_style('whitegrid')
    ax = sns.barplot(x='Ranking Score', y='Features Name', data = fimp)
    ax.set(xlabel='Gini Importance')
    ax.set(title='Feature Importance')
    # plt.show()
    return fimp

###--PREDICTION--################################################################

def prediction(model):
    '''
        Accept features values from the user and classify on base of those features 
        if the iris flower spece (setosa, virginica, versicolor) 
    '''
    print("============WELCOME TO=============")
    print("    Iris Flower Classification     ")
    print("===================================")
    print(" ")
    # Load Model
    model = pickle.load(open(model, "rb"))
    
    print("Please Enter Predictor variables")
    print("--------------------------------")
    l=[]
    l.append(float(input('Enter sepal length : ')))
    l.append(float(input('Enter sepal width  : ')))
    l.append(float(input('Enter petal length : ')))
    l.append(float(input('Enter petal width  : ')))
    print("--------------------------------")
    print("")      
    #arr = pd.DataFrame([l])
    #arr = np.asarray([l])
         
    pred_result = model.predict([l])
    print("===================================")
    print(f'Iris Classified as : {pred_result}')
    print("============THANK YOU=============")
    
    
    
    