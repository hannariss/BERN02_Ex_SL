import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import func

data = pd.read_csv('pollution_cleaneddata.csv')

MORT = data['MORT']
POOR = data['POOR']

x_0 = np.linspace(min(POOR), max(POOR), 200)
pred, se = func.LOESS(POOR, MORT, 12, x_0)

plt.plot(POOR, MORT, 'o', label='data')
plt.plot(x_0, pred, 'r-', label='LOESS fit')
plt.fill_between(x_0, pred-se, pred+se, color="blue", alpha=0.2, label="+- 1 SE")
plt.title(f'LOESS fit with k=30')
plt.xlabel('POOR')
plt.ylabel('MORT')
plt.legend()
plt.show()

pred, se = func.LOESS(POOR, MORT, 10, x_0=[10, 18, 25])
print(f'MORT predictions for POOR = 10%, 18%, 25% : {pred} \n standard deviations: {se}')

