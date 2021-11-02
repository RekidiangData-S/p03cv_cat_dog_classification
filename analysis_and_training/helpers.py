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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import roc_curve, auc

import pickle

# CLASSIFICATION

def model_select_class(features, target, models):
    import numpy as np
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)
    train_accuracy=[]
    test_accuracy=[]
    ROC=[]
   
    algo_names = []
    accs = []
    rocs = []
    precs = []
    recs = []
    f1s = []
    gen_errors =[]
    model_probs = []
    for model in models:
        

        clf=model[1]
        clf.fit(X_train, y_train)
        train_acc=round((clf.score(X_train, y_train))*100,2)
        train_accuracy.append(train_acc)

        y_pred=clf.predict(X_test)
        #find accuracy
        test_acc=round(accuracy_score(y_test,y_pred)*100,2)
        test_accuracy.append(test_acc)

        #find the ROC_AOC curve
        roc=round(roc_auc_score(y_test,y_pred)*100,2)
        ROC.append(roc)
        
        y_true = y_test
        tn = confusion_matrix(y_true, y_pred)[0, 0]
        fp = confusion_matrix(y_true, y_pred)[0, 1]
        fn = confusion_matrix(y_true, y_pred)[1, 0]
        tp = confusion_matrix(y_true, y_pred)[1, 1]

        accu = round(((tp+tn)/(tp+tn+fp+fn))*100,2)
        prec = round((tp/(tp+fp))*100,2)
        rec = round((tp/(tp+fn))*100, 2) #TPR
        f1 = round((2*((prec*rec)/(prec+rec))),2)
        FPR = round((tn/(tn+fp))*100,2)
        roc_ = round((rec/FPR)*100)
        #gen_error = (train_acc - test_acc)
        #print("\nAccuracy {0} ROC {1}".format(acc,roc))
       
        algo_names.append(model[0])
        #train_accuracy.append(train_acc)
        rocs.append(roc)
        precs.append(prec)
        recs.append(rec) 
        f1s.append(f1)
       # gen_errors.append(gen_error)
        #  Calibration Curve (Reliability Curves) 
        #model_prob = clf.fit(X_train, y_train).predict_proba(X_test)
        #model_probs.append(model_prob)
        
    #skplt.metrics.plot_calibration_curve(y_test,    model_probs, algo_names, n_bins=15, figsize=(12,6));
    
    
    res = pd.DataFrame({"Agorithm":algo_names, "Train-Accuracy":train_accuracy,  "ROC":rocs, "Precision":precs,
                        "Recall":recs, "F1-Score":f1s, "Test-Accuracy":test_accuracy})
    res['"GEN. ERROR"'] = res["Train-Accuracy"] - res["Test-Accuracy"]
    
    return res


def classify_cv(X, y, algo_name, algorithm, gridsearchParams, cv):
    """
    Classification with cross-validation method
    """
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

def classify_cv1(features, target, algo_name, algorithm, gridsearchParams, cv):
    
    import numpy as np
    np.random.seed(10)
    algo_name=algo_name
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)
    
    grid = GridSearchCV(
    estimator=algorithm,
    param_grid=gridsearchParams,
    cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_clf = grid.fit(X_train, y_train)
    best_params = grid_clf.best_params_
    y_pred = grid_clf.predict(X_test)
    train_acc = grid_clf.score(X_train, y_train)
    test_acc = grid_clf.score(X_test, y_test)
    roc=round(roc_auc_score(y_test,y_pred)*100,2)
    #redt = grid_clf.predict(X_train)
    
    
    y_true = y_test
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]

    accu = round(((tp+tn)/(tp+tn+fp+fn))*100,2)
    prec = round((tp/(tp+fp))*100,2)
    rec = round((tp/(tp+fn))*100, 2) #TPR
    f1 = round((2*((prec*rec)/(prec+rec))),2)
    FPR = round((tn/(tn+fp))*100,2)
    roc_ = round((rec/FPR)*100)
    gen_error = (train_acc - test_acc)

    res = pd.DataFrame({"Agorithm":algo_name, "Train-Accuracy":train_acc, "ROC":roc, "Precision":prec, 
                        "Recall":rec, "F1-Score":f1, "Test-Accuracy":test_acc, "GEN. ERROR":gen_error}, index=[0])
    return res
    
    
def classify_ho(X, y,algo_name,  algorithm):
    """
    Classification with holdout method
    """
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

def classify_ho1(algo_name, algo_and_hp, features, target):
    import numpy as np
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    algo_names = algo_name
    clf = algo_and_hp
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    roc=round(roc_auc_score(y_test,y_pred)*100,2)
    
    y_pred=clf.predict(X_test)
    
    y_true = y_test
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]

    accu = round(((tp+tn)/(tp+tn+fp+fn))*100,2)
    prec = round((tp/(tp+fp))*100,2)
    rec = round((tp/(tp+fn))*100, 2) #TPR
    f1 = round((2*((prec*rec)/(prec+rec))),2)
    FPR = round((tn/(tn+fp))*100,2)
    gen_error = round((train_acc - test_acc)*100,2)

    res = pd.DataFrame({"Agorithm":algo_names, "Train-Accuracy":train_acc, "ROC":roc, "Precision":prec, 
                        "Recall":rec, "F1-Score":f1, "Test-Accuracy":test_acc, "GEN. ERROR":gen_error}, index=[0])
    return res


    
# FEATURE IMPORTANCE
   
def feature_import(features, model):
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
