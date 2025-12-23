import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import os

logger = logging.getLogger(__name__)
numeric = ['tenure','MonthlyCharges','TotalCharges']

def load_data(path:str):
    df = pd.read_csv(path)
    logger.info(f"loaded raw data: {df.shape}")
    return df

def basic_clean(df:pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for n in numeric:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n],errors='coerce')
    
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().replace({'':pd.NA,'nan':pd.NA,'NA':pd.NA})
    
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    return df

def feature_engineer(df:pd.DataFrame):
    df = df.copy()
    df['tenure_bucket'] = pd.cut(df['tenure'],bins=[-1,12,24,48,60,1000],
                                 labels=['0-12','12-24','24-48','48-60','60+'])
    return df

def split_save(df:pd.DataFrame,target='Churn',test_size=0.2,random_state=42,
               out_dir="C:/ML_Projects/customer_churn_predictor/data/processed"):
    os.makedirs(out_dir,exist_ok=True)
    from pathlib import Path
    out_dir = Path(out_dir)
    df = df.copy()
    df = df.dropna(subset=[target])
    x = df.drop(columns=[target,'customerID'])
    y = df[target]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,
                                                     random_state=random_state,stratify=y)
    
    x_train.to_csv(out_dir/'x_train.csv',index=False)
    x_test.to_csv(out_dir/'x_test.csv',index=False)
    y_train.to_csv(out_dir/'y_train.csv',index=False)
    y_test.to_csv(out_dir/'y_test.csv',index=False)

    return x_train,x_test,y_train,y_test

def get_num_and_cat_cols(df:pd.DataFrame):
    df = df.copy()
    exclude = ['Churn','customerID']
    cols = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in cols if c in numeric or pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols ]

    return num_cols,cat_cols
