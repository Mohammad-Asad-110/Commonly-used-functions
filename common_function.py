# Splitting data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=110)



# STANDARD SCALER

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)



# MINMAX SCALER

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)


# LABEL ENCODER

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)


# IMPUTING

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0) # can also be set to NaN / np.nan
X_train=imp.fit_transform(X_train)


#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# CALSSIFICATION REPORT

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ACURACY REPORT

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# GRID SEARCH

from sklearn.grid_search import GridSearchCV

params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=model,param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)


# RANDOM SEARCH

from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5), "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,param_distributions=params,cv=4,n_iter=8,random_state=110)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)






