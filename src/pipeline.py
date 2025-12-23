from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def build_preprocessor(numeric_cols,categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('impute',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num',numeric_transformer,numeric_cols),
        ('cat',categorical_transformer,categorical_cols)
    ],remainder='drop')

    return preprocessor

