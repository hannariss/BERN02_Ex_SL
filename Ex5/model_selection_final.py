import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


training_data = pd.read_csv('train.csv')

columns_to_drop = ['Ethnicity', 'ID', 'Country_Born', 'Country_Res', 'RelY', 'RelM', 'prec_mean_anual_Born', 'snow_cover_days_Born', 't_mean_Born', 't_mean_coldest_month_Born', 't_mean_warmest_month_Born']
#define columns with too many NaN values
columns_nan = ['HPP_9', 'GDP', 'GINI', 'SexRatio.2017est.', 'SexRatio.15.54.', 'SRA_Partner', 'SRM_Partner', 'SRH_Partner', 'UNRegion', 'RelLength' ]

X_train = training_data.drop(columns_to_drop, axis=1)
X_train = X_train.drop(columns_nan, axis=1)
X_train = X_train.dropna() #drop remaining rows with NANs (from 1400 rows to 700)

y_train_final = X_train['Kiss_index']
X_train_final = X_train.drop('Kiss_index', axis=1)

#define model
lasso_cv = LassoCV(cv=5)  # 5-fold cross-validation
lasso_cv.fit(X_train_final.values, y_train_final.values)
selected_features = X_train_final.columns[lasso_cv.coef_ != 0]
print(selected_features)

# define second model
Elastic_model2 = ElasticNetCV(cv=5, l1_ratio=0.2, max_iter=2000)
Elastic_model2.fit(X_train_final, y_train_final)

# X_train_split = X_train_final[:800].values
# y_train_split = y_train_final[:800].values
# X_test_split = X_train_final[800:].values
# y_test_split = y_train_final[800:].values

# lasso_cv_test = LassoCV(cv=5, max_iter=2000)  # 5-fold cross-validation
# lasso_cv_test.fit(X_train_split, y_train_split)
# y_pred_test = lasso_cv_test.predict(X_test_split)
# mse_test = mean_squared_error(y_test_split, y_pred_test)


# make predictions on test data
data_test = pd.read_csv('test.csv')

X_test = data_test.drop(columns_to_drop, axis=1)
X_test = X_test.drop(columns_nan, axis=1)
X_test = X_test.dropna()

y_pred = lasso_cv.predict(X_test.values)

y_pred_elast = Elastic_model2.predict(X_test.values)






#save to csv
ID = data_test['ID']
# output = pd.DataFrame({'ID': ID, 'Kiss_index': y_pred})
# output.to_csv('submission1.csv', index=False)


output_elast = pd.DataFrame({'ID': ID, 'Kiss_index': y_pred_elast})
output_elast.to_csv('submission_elast.csv', index=False)
