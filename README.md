# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
~~~
import pandas as pd
from scipy import stats
import numpy as np
~~~


~~~
df=pd.read_csv("/content/sample_data")
~~~


~~~
df.head()
~~~


<img width="1919" height="1026" alt="Screenshot 2026-08-24 152736" src="https://github.com/user-attachments/assets/9f9cfd51-8ad0-4c46-af9f-88db2d345b16" />



~~~
df.dropna()
~~~


~~~
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
~~~

<img width="1918" height="1007" alt="Screenshot 2026-08-24 152751" src="https://github.com/user-attachments/assets/7458f095-d0d8-43cf-b879-db1c459a9910" />



~~~
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
~~~

<img width="1919" height="965" alt="Screenshot 2026-08-24 152805" src="https://github.com/user-attachments/assets/504261fd-061c-4c08-9b8a-74cb6503ffcc" />



~~~
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
~~~


~~~
df
~~~



<img width="1919" height="1014" alt="Screenshot 2026-08-24 152814" src="https://github.com/user-attachments/assets/06073bab-e8ee-4353-b233-0d47b808d04f" />



~~~
df3=pd.read_csv("/content/bmi.csv")
~~~


~~~
<img width="1918" height="1027" alt="Screenshot 2026-08-24 152824" src="https://github.com/user-attachments/assets/4b6fd4e5-b2b5-4eaa-9bf0-58bef8078f6e" />
~~~


~~~
df3
~~~



<img width="1918" height="1027" alt="Screenshot 2026-08-24 152824" src="https://github.com/user-attachments/assets/3996ab74-678b-47c2-89ee-edb2834c5a11" />


~~~
df4=pd.read_csv("/content/bmi.csv")
~~~


~~~
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
~~~

<img width="1919" height="840" alt="Screenshot 2026-08-24 152838" src="https://github.com/user-attachments/assets/096f83fe-54cc-4164-8199-b0dccd27a1c8" />

~~~
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
~~~



~~~
df=pd.read_csv('/content/titanic_dataset.csv')
~~~

~~~
df.columns
~~~



~~~
df.shape
~~~


<img width="1914" height="938" alt="Screenshot 2026-08-24 152849" src="https://github.com/user-attachments/assets/28641fe6-0379-4383-99b6-64fb05d05864" />

~~~
x=df.drop("Survived",axis=1)
y=df["Survived"]
~~~


~~~
df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
~~~


~~~
df.columns
~~~


~~~
df['Age'].isnull().sum()
~~~


~~~
df['Age'].fillna(method='ffill')
~~~


<img width="1917" height="1010" alt="Screenshot 2026-08-24 152858" src="https://github.com/user-attachments/assets/0e956fa7-afce-4acb-9d90-7cf86cdc72fa" />

~~~
tips.head()
~~~



~~~
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
~~~



~~~
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic:{chi2}")
print(f"p-value:{p}")
~~~
<img width="1674" height="627" alt="Screenshot 2026-08-24 153859" src="https://github.com/user-attachments/assets/d8aaa027-755f-4629-94f6-aff11323a3fe" />

~~~
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,0,1,0]

}
my_pd=pd.DataFrame(data)

X=my_pd[['Feature1','Feature3']]
y=my_pd['Target']

selector=SelectKBest(score_func=mutual_info_classif,k=2)
x_new=selector.fit_transform(X,y)

selected_features_indices=selector.get_support(indices=True)

selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
~~~

<img width="1919" height="868" alt="Screenshot 2026-08-24 152927" src="https://github.com/user-attachments/assets/d94e2824-3392-4111-8945-a4a6bc55fbee" />

<img width="1919" height="692" alt="Screenshot 2026-08-24 152940" src="https://github.com/user-attachments/assets/48b26524-0c9f-46ac-bd18-7dfdee451894" />

<img width="1911" height="895" alt="Screenshot 2026-08-24 152949" src="https://github.com/user-attachments/assets/53d51e45-6ac2-4ec9-9be5-6ad644ec8b63" />


# RESULT:
       # INCLUDE YOUR RESULT HERE
