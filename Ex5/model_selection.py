import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV


training_data = pd.read_csv('train.csv')

#drop inhomogeneous columns
training_data.drop('Ethnicity', axis=1, inplace=True)

y_train = training_data['Kiss_index']
columns_to_drop = ['Kiss_index', 'ID', 'Country_Born', 'Country_Res', 'RelY', 'RelM', 'prec_mean_anual_Born', 'snow_cover_days_Born', 't_mean_Born', 't_mean_coldest_month_Born', 't_mean_warmest_month_Born']
X_train = training_data.drop(columns_to_drop, axis=1)

#possibly drop relationship year and month - Rellength combines both
#RelY, RelM, prec_mean_anual_Born, snow_cover_days_Born, t_mean_Born, t_mean_coldest_month_Born, t_mean_warmest_month_Born 


lasso_cv = LassoCV(cv=5)  # 5-fold cross-validation
lasso_cv.fit(X_train.values.reshape(-1,1), y_train)


