import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
import sys, os
sys.path.append("..")

def pl_ROC(algo, roc_auc, fpr, tpr):

    #define figure size 
    plt.figure(figsize=(12,12))

    #add title
    plt.title('{} ROC Curve'.format(algo))

    # plot and add labels to plot
    plt.plot(fpr, tpr, 'b', label = 'Diabetes data: {} AUC =  {}'.format(algo,(round(roc_auc,4))))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return

def Diabetes_tree(best_model, X):
    plt.figure(figsize=(30,14))
    tree.plot_tree(best_model, filled=True, fontsize=12)
    
    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/diabetes/reports/figures'
    os.chdir(pathr)
    export_graphviz(best_model,out_file=("Diabetes_tree.dot"), feature_names=X.columns[:],class_names=(['Diabetes Negative','Diabetes Positive']), rounded=True, filled=True)
    os.system("dot -Tpng Diabetes_tree.dot -o Diabetes_tree.png") 
    os.system("dot -Tps Diabetes_tree.dot -o Diabetes_tree.ps")

    #change to the start working directory
    os.chdir(path_start)
    return