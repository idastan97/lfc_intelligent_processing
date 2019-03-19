import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('./data.csv')
df = pd.DataFrame(data)

np_data = df.values
cols = list(df)
feature_cols = cols[:-15]
target_cols = cols[-15:]

X = np_data[:, [i for i in range(len(feature_cols))]]

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0, 10, 16]              # The column(s) to be applied on.
         )
    ]
)

X_categ = transformer.fit_transform(X)
X = np.delete(X,[0,10,16],1)
X = np.concatenate((X,X_categ),axis=1)

Y_classes = {
    'offside' : np_data[:, -5],
    'penalty' : np_data[:, -4],
    'corner'  : np_data[:, -3],
    'oppscore': np_data[:, -2],
    'wescore' : np_data[:, -1]
}

Y_zoom = {
            'corner' : np_data[:, -15],
            'save' : np_data[:, -14],
            'freek'  : np_data[:, -13],
            'goal': np_data[:, -12],
            'assist' : np_data[:, -11],
            'foul' : np_data[:, -10],
            'penalty' : np_data[:, -9],
            'offside' : np_data[:, -8],
            'steal' : np_data[:, -7],
            'fight'  : np_data[:, -6]
}

models = {}
for cl in Y_classes:
    knn = KNeighborsClassifier(n_neighbors=25)
    knn.fit(X, Y_classes[cl])
    models[cl] = knn

models_zoom = {}
for cl in Y_zoom:
    knn = KNeighborsRegressor(n_neighbors=25)
    knn.fit(X, Y_zoom[cl])
    models_zoom[cl] = knn