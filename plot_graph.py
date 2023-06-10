import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
# X = [ 0.1, 0.20, 0.25, 0.45, 0.50, 0.65, 0.75, 0.90, ]
# Y = [ 0.326, 0.256, 0.2168, 0.258, 0.295, 0.2259, 0.2168, 0.3377,]
X = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9]
Y = [0.7814, 0.7982, 0.8085, 0.8191, 0.8320, 0.8948, 0.9145, 1.0]
new_X = np.linspace(0.05, 0.90, 20)
f_cubic = interp1d(X, Y, kind = 'cubic')
plt.plot(new_X[::-1], f_cubic(new_X)[::-1])
plt.xlabel('n / m')
plt.ylabel('Purity')
plt.title('Octet Purity')
plt.show()
# data = {'noise-ratio' : X,  'P+D+PD' : Y}
# df = pd.DataFrame.from_dict(data=data, orient = 'index')
# print(df)
# sns.lineplot(x = "noise-ratio", y = "Profit", data = df)
# plt.title('Octet Processing speed with noise-ratio-region')
# # plt.legend()
# plt.show()
# Year = [2012, 2014, 2016, 2020, 2021, 2022, 2018]
# Profit = [80, 75.8, 74, 65, 99.5, 19, 33.6]

# data_plot = pd.DataFrame({"Year":Year, "Profit":Profit})

# sns.lineplot(x = "Year", y = "Profit", data=data_plot)
# plt.show()