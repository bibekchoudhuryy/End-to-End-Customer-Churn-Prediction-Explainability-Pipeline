import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline

def build_model_candidates(preprocessor,random_state=42):
    models = {}
    models['logisticregresion'] = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('clf',LogisticRegression(max_iter=2000,random_state=random_state,class_weight='balanced'))
    ])

    models['RandomForest'] = imbpipeline(steps=[
        ('preprocessor',preprocessor),
        ('smote',SMOTE(random_state=random_state)),
        ('clf',RandomForestClassifier(random_state=random_state,n_jobs=-1))
    ])

    models['xgb'] = imbpipeline(steps=[
        ('preprocessor',preprocessor),
        ('smote',SMOTE(random_state=random_state)),
        ('clf',XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=random_state))
    ])

    return models

def grid_search(pipeline:Pipeline,param_grid,x,y,cv=3,scoring='avarage_precision'):
    skf = StratifiedKFold(n_splits=cv,shuffle=True,random_state=42)
    gs = GridSearchCV(pipeline,param_grid=param_grid,scoring=scoring,cv=skf,n_jobs=-1,verbose=1)
    gs.fit(x,y)
    return gs