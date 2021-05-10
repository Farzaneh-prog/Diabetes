import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV,  KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC

def train_model(X_train, X_test, y_train, y_test, algo):
    # all available models in dictionary
    models = {"DT":DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=14),"rf":RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0),"knn": KNeighborsClassifier(n_neighbors = 4, weights='distance', algorithm='auto',leaf_size=2, p=2, metric='minkowski'), "svm":SVC(kernel='poly', probability = True, C = 0.1, gamma='auto')}

    # select model
    model = models[algo]
    
    # train model
    model.fit(X_train,y_train)

    # score model
    accuracy = model.score(X_test, y_test)
    print('Model {} successfully trained with an accuracy of {}% '.format(algo,round(accuracy,4)*100))  
    model_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, model_pred)

    # give back
    return model, conf_mat

def optimize_model(model, X_train,y_train, algo):

    para={"DT":{'criterion':['gini','entropy'],'max_depth':[i for i in range(1,6)],'min_samples_split':[i for i in range(2,20)]},"rf":{'criterion':['gini','entropy'],'n_estimators':[3, 10, 30],'max_depth':[i for i in range(1,6)],'min_samples_split':[i for i in range(2,10)]},"knn":{'n_neighbors':[i for i in range(2,10)],'weights':['uniform','distance'],'leaf_size':[i for i in range(2,30)]},"svm":{'kernel':('linear','poly','sigmoid','rbf'),'C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],'gamma':(1,2,3,'auto')}}

    #parameterraum which shall be tested to optimize hyperparameters
    model_para = para[algo]

    #GridSearchCV object
    model_grd = GridSearchCV(model, model_para, cv=5) 

    #creates differnt classifiers with all the differnet parameters out of our data
    model_grd.fit(X_train,y_train)
    #man könnte hier für Knn Methode auch auf (X,y) trainieren um dem Optimale Hyperparameter zu finden und den gerechnete Accuranz hier ist das richtige Accuranz
 
    #best paramters that were found
    model_best_parameters = model_grd.best_params_  
    print('Model {} successfully optimized with the best parameters of {}.'.format(algo,model_best_parameters))  

    #new model object with best parameters
    model_with_best_parameters = model_grd.best_estimator_

    return model_with_best_parameters

def kfoldevaluate_optimzed_model(algo, model_with_best_parameters, X, y, X_train, X_test, y_train, y_test):

    # train model
    model_with_best_parameters.fit(X_train,y_train)

    # score model
    accuracy = model_with_best_parameters.score(X_test, y_test)
    print('Model {} successfully trained with an accuracy of {}% '.format(model_with_best_parameters,round(accuracy,4)*100))  
    model_pred = model_with_best_parameters.predict(X_test)
    conf_mat = confusion_matrix(y_test, model_pred)

    #k_fold object to optimize the accuracy measurement
    accuracy_k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

    #scores reached with different splits of training/test data 
    accuracy_k_fold_scores = cross_val_score(model_with_best_parameters, X, y, cv=accuracy_k_fold, n_jobs=-1)

    #arithmetic mean of accuracy scores 
    mean_accuracy_best_parameters = np.mean(accuracy_k_fold_scores)

    print('K_fold Accuracy of {} with best parameters is {}% '.format(algo, round(mean_accuracy_best_parameters, 4)*100))

    return model_with_best_parameters, conf_mat

def roc_vorbereitung(model_with_best_parameters, X_test,y_test):
    
    # get probabilities of class membership of test instances
    model_probs = model_with_best_parameters.predict_proba(X_test)

    #get col with positive probabilities
    y_pred_proba = model_probs[:,1]

    # get false positive rate, true positive rate and threshold values
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba, pos_label=1)
    
    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, threshold, roc_auc
