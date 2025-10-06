import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


training_data = pd.read_csv('train.csv')

#y_train = training_data['Kiss_index']
columns_to_drop = ['Ethnicity', 'ID', 'Country_Born', 'Country_Res', 'RelY', 'RelM', 'prec_mean_anual_Born', 'snow_cover_days_Born', 't_mean_Born', 't_mean_coldest_month_Born', 't_mean_warmest_month_Born']
#define columns with too many NaN values
columns_nan = ['HPP_9', 'GDP', 'GINI', 'SexRatio.2017est.', 'SexRatio.15.54.', 'SRA_Partner', 'SRM_Partner', 'SRH_Partner' ]

X_train = training_data.drop(columns_to_drop, axis=1)
X_train = X_train.drop(columns_nan, axis=1)
X_train1 = X_train.dropna() #drop remaining rows with NANs (from 1400 rows to 700)

# define final X and y
y_train1_final = X_train1['Kiss_index']
X_train1_final = X_train1.drop('Kiss_index', axis=1)

#possibly drop relationship year and month - Rellength combines both
#RelY, RelM, prec_mean_anual_Born, snow_cover_days_Born, t_mean_Born, t_mean_coldest_month_Born, t_mean_warmest_month_Born 


lasso_cv = LassoCV(cv=5)  # 5-fold cross-validation
lasso_cv.fit(X_train1_final.values, y_train1_final.values)
selected_features = X_train1_final.columns[lasso_cv.coef_ != 0]
print(selected_features)

# try with larger dataset
columns_nan2 = ['UNRegion', 'RelLength']
X_train2 = X_train.drop(columns_nan2, axis=1)
X_train2 = X_train2.dropna() #drop remaining rows with NANs (from 1400 rows to 900)

y_train2_final = X_train2['Kiss_index']
X_train2_final = X_train2.drop('Kiss_index', axis=1)

lasso_cv2 = LassoCV(cv=5)  # 5-fold cross-validation
lasso_cv2.fit(X_train2_final.values, y_train2_final.values)
selected_features2 = X_train2_final.columns[lasso_cv2.coef_ != 0]
print(selected_features2)

Elastic_model = ElasticNetCV(cv=5, l1_ratio=0.1)
Elastic_model.fit(X_train2_final.values, y_train2_final.values)
selected_features_enet = X_train2_final.columns[Elastic_model.coef_ != 0]
print(selected_features_enet)


# try scaling data
scaler = StandardScaler()
scaled_data_X = scaler.fit_transform(X_train2_final.values)


lasso_cv_scaled = LassoCV(cv=5, max_iter=2000)  # 5-fold cross-validation
lasso_cv_scaled.fit(scaled_data_X, y_train2_final.values)
# selected_features2 = X_train1_final2.columns[lasso_cv2.coef_ != 0]
# print(selected_features2)


# Split data to check how well model performs
X_train_split_scaled = scaler.fit_transform(X_train2_final[:800].values)
y_train_split = y_train2_final[:800].values
X_test_split_scaled = scaler.fit_transform(X_train2_final[800:].values)
y_test_split = y_train2_final[800:].values


lasso_cv_split = LassoCV(cv=5, max_iter=2000)  # 5-fold cross-validation
lasso_cv_split.fit(X_train_split_scaled, y_train_split)
y_pred = lasso_cv_split.predict(X_test_split_scaled)
mse = mean_squared_error(y_test_split, y_pred)

Elastic_model2 = ElasticNetCV(cv=5, l1_ratio=0.2, max_iter=2000)
Elastic_model2.fit(X_train_split_scaled, y_train_split)
# selected_features_enet = X_train2_final.columns[Elastic_model.coef_ != 0]
# print(selected_features_enet)
y_pred_elast = Elastic_model2.predict(X_test_split_scaled)
mse_elast = mean_squared_error(y_test_split, y_pred_elast)

X_train_split = X_train2_final[:800].values
y_train_split = y_train2_final[:800].values
X_test_split = X_train2_final[800:].values
y_test_split = y_train2_final[800:].values


lasso_cv_test = LassoCV(cv=5, max_iter=2000)  # 5-fold cross-validation
lasso_cv_test.fit(X_train_split, y_train_split)
y_pred_test = lasso_cv_test.predict(X_test_split)
mse_test = mean_squared_error(y_test_split, y_pred_test)


# try with differently preprocessed data
X_train1_split_scaled = scaler.fit_transform(X_train1_final[:600].values)
y_train1_split = y_train1_final[:600].values
X_test1_split_scaled = scaler.fit_transform(X_train1_final[600:].values)
y_test1_split = y_train2_final[600:].values

lasso_cv1 = LassoCV(cv=5, max_iter=2000)  # 5-fold cross-validation
lasso_cv1.fit(X_train1_split_scaled, y_train1_split)
y_pred1 = lasso_cv1.predict(X_test1_split_scaled)
# mse1 = mean_squared_error(y_test1_split, y_pred1)










