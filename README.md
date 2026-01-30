<H3>ENTER YOUR NAME: KIRUTIKA K R
<H3>ENTER YOUR REGISTER NO: 212224230128
<H3>EX. NO.1</H3>
<H3>DATE: 30.01.2026
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df= pd.read_csv("Churn_Modelling.csv")
print(df)

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()
print(df['EstimatedSalary'].describe())

scaler=MinMaxScaler()
df1 = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include='number')),
    columns=df.select_dtypes(include='number').columns
)
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print("Lenght of X_test ",len(X_test))
```


## OUTPUT:
### DataSet:
<img width="1138" height="805" alt="image" src="https://github.com/user-attachments/assets/d5f0a576-dad5-4224-bd36-a97cacac08c1" />


### X VALUES:
<img width="735" height="262" alt="image" src="https://github.com/user-attachments/assets/e7bb0066-5fc1-4d7b-9fad-f8e83d9a37f5" />

### Y VALUES:
<img width="502" height="126" alt="image" src="https://github.com/user-attachments/assets/b2e7f032-d7d1-4411-a383-5c7d051c033a" />

### Null Values:
<img width="988" height="750" alt="image" src="https://github.com/user-attachments/assets/d93c6c17-e2ae-43aa-b072-54fa2070c2ba" />

### Duplicate values:
<img width="548" height="321" alt="image" src="https://github.com/user-attachments/assets/22c80fd9-c4cf-48ba-9934-1ae8c81160e6" />

### Describing The DataSet:
<img width="690" height="297" alt="image" src="https://github.com/user-attachments/assets/0960f5f5-a26d-4aba-a0d0-dd4f9f8f291a" />

### TRAINING DATA:
<img width="1095" height="755" alt="image" src="https://github.com/user-attachments/assets/2ba981d9-bd1b-4551-b9ce-73993257d545" />

### TESTING DATA:
<img width="1017" height="518" alt="image" src="https://github.com/user-attachments/assets/918d1bc7-ed4d-44df-8381-c9d39ba14c1b" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


