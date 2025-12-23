import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve,average_precision_score
import os
import pandas as pd,numpy as np
import joblib
from sklearn.pipeline import Pipeline
from pathlib import Path

def plot_confusion(model:Pipeline,x_test,y_test,
                   output='C:/ML_Projects/customer_churn_predictor/plots'):
    output = Path(output)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('confusion metrics')
    os.makedirs(output,exist_ok=True)
    plt.savefig(output/'confusion_metrics.png')
    plt.close()

def plot_presicion_recall(model:Pipeline,x_test,y_test,
                          output='C:/ML_Projects/customer_churn_predictor/plots'):
    output = Path(output)
    try:
        y_pred = model.predict_proba(x_test)[:,1]
    except Exception:
        y_pred = model.predict(x_test)

    precision,recall,_ = precision_recall_curve(y_test,y_pred)
    ap = average_precision_score(y_test,y_pred)

    plt.plot(precision,recall,label=f"avg precision score:{ap:.3f}")
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision Recall Curve')

    os.makedirs(output,exist_ok=True)
    plt.savefig(output/'precision_recall.png')
    plt.close()


def save_top_features_perm(model:Pipeline,x_test,y_test,
     output='C:/ML_Projects/customer_churn_predictor/plots',top_n=15):
    from sklearn.inspection import permutation_importance
    output = Path(output)
    if hasattr(model,'named_steps') and 'preprocessor' in model.named_steps:
        pre = model.named_steps['preprocessor']
        x_trans = pre.transform(x_test)
        clf = model.named_steps['clf']
        
        if hasattr(clf,'named_steps') and 'rf' in clf.named_steps:
            estimator = clf.named_steps[list(clf.named_steps).keys()[-1]]
        else:
            estimator = clf
    else:
        x_trans = x_test.values
        estimator = model

    res = permutation_importance(estimator,x_trans,y_test,n_repeats=10,random_state=42,n_jobs=-1)
    importances = res.importances_mean
    try:
        feature_names = pre.get_feature_names_out()
    except:
        feature_names = [f"f_{i}"for i in range(x_trans.shape[1])]
    imp_df = pd.DataFrame({'features':feature_names,'importances':importances})
    imp_df = imp_df.sort_values(by='importances',ascending=False).head(top_n)

    plt.barh(imp_df['features'][::-1],imp_df['importances'][::-1])
    plt.xlabel('permutation importances')
    plt.ylabel('Importances')
    plt.title('Top Feature Importances')
    os.makedirs(output,exist_ok=True)
    plt.savefig(output/'feature_importances.png')
    plt.close()

    imp_df.to_csv('C:/ML_Projects/customer_churn_predictor/reports/feature_importances.csv')
    return imp_df




