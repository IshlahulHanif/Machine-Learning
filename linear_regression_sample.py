import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#membaca data
USAhousing = pd.read_csv('USA_Housing.csv')

#memeriksa data
print(USAhousing.head())
print(USAhousing.describe())

#kita ingin memprediksi harga rumah menggunakan semua variable yang memiliki pengaruh(korelasi) thd harga rumah
sns.pairplot(USAhousing)
plt.show()
"""
Secara intuitif, terlihat bahwa semua variable memiliki korelasi thd harga rumah
sehingga, semua variable selain 'Price' akan kita lihat korelasinya terhadap 'Price'
"""
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#membuat object linear model
lm = LinearRegression()
lm.fit(X_train,y_train)

#kita ingin melihat koefisien dari masing-masing variable, tujuannya untuk melihat pengaruhnya thd harga rumah
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()
"""
terlihat bahwa perbandingan hasil prediksi dan data sebenarnya memiliki korelasi yang baik
hal ini terlihat dari data prediksi dan data sebenarnya yang membentuk grafik linier naik
"""