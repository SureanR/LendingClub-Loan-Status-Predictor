# LendingClub-Loan-Status-Predictor
The model was built to determine whether a borrower will repay their loan. The model was built using historical data on loans given out with information whether or not the borrower defaulted. The model was built using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club.

Project was created using Jupyter Notebook 6.1.4

## Imports and loading the Data
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
```
```
%matplotlib inline  
df = pd.read_csv('../DATA/lending_club_loan_two.csv')  
df.info()  
```
![image](https://user-images.githubusercontent.com/89992872/132103153-6e4c3438-02ef-4b1a-8ec2-6865b6433e59.png)

## Exploratory Data Analysis
```
sns.countplot(x = df['loan_status'])  
```
![image](https://user-images.githubusercontent.com/89992872/132103287-e3bdeb9b-884d-433b-99ca-6a9b944b64f1.png)
```
plt.figure(figsize=(12,4))  
sns.distplot(df['loan_amnt'], kde=False)
```
![image](https://user-images.githubusercontent.com/89992872/132103300-2baa6f4f-2b3d-4471-9c40-2975fc4dedfb.png)
```
plt.figure(figsize=(12,8))  
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)  
```
![image](https://user-images.githubusercontent.com/89992872/132103322-67eeb7f7-219c-4cd8-bd39-004b0506e550.png)

## Exploring the almost perfect correlation with the "installment feature"
```
feat_info('installment')
```
![image](https://user-images.githubusercontent.com/89992872/132103405-732b9d9d-f87f-4119-9d65-330f40828b7b.png)
```
feat_info('loan_amnt')
```
![image](https://user-images.githubusercontent.com/89992872/132103422-f661f028-7844-43ec-b28d-7855cb7f5cfc.png)
```
sns.scatterplot(x='installment',y='loan_amnt',data=df)
```
![image](https://user-images.githubusercontent.com/89992872/132103431-4efed838-729e-40e9-8ad4-e61dee885f61.png)
```
sns.boxplot(x='loan_status', y='loan_amnt', data=df)  
```
![image](https://user-images.githubusercontent.com/89992872/132103448-38b6b687-7351-45ae-9a98-d73b4d55d2cb.png)

## Calculating summary statistics for the loan amount, grouped by the loan_status
```
df.groupby('loan_status')['loan_amnt'].describe()  
```
![image](https://user-images.githubusercontent.com/89992872/132103459-35129ee4-3408-48ba-9788-9c84d932e706.png)

## Exploring Grade and SubGrade Columns that LendingClub attributes to loans
```
sorted(df['grade'].unique())
```
![image](https://user-images.githubusercontent.com/89992872/132103883-1b1beabb-1654-4adc-a593-c24c8097876f.png)
```
sorted(df['sub_grade'].unique())  
```
![image](https://user-images.githubusercontent.com/89992872/132103890-2de32f17-bbd2-438f-aa2f-a9a342620a98.png)
```
sns.countplot(x='grade', data=df, hue='loan_status' )
```
![image](https://user-images.githubusercontent.com/89992872/132103904-8fb03bb2-82c3-4f72-9c8f-7b36b99b33e6.png)

## Countplot per subgrade with and without hue
```
plt.figure(figsize=(12,4))  
subgrade_order=sorted(df['sub_grade'].unique())  
sns.countplot(x='sub_grade', data=df, order=subgrade_order , palette='coolwarm')
```
![image](https://user-images.githubusercontent.com/89992872/132103934-04b40273-ca03-4bea-ab4c-282b5f145bd1.png)
```
plt.figure(figsize=(12,4))  
sns.countplot(x='sub_grade', data=df, order=subgrade_order , palette='coolwarm', hue='loan_status')  
```
![image](https://user-images.githubusercontent.com/89992872/132103944-ba3650e0-e3d7-4e29-ba30-0d62570ab011.png)

## Isolating F and G subgrades as they don't get paid back too often
```
plt.figure(figsize=(12,4))  
f_g=['F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']  
sns.countplot(x='sub_grade', data=df, order=f_g, hue='loan_status')  
```
![image](https://user-images.githubusercontent.com/89992872/132103964-f979835a-fb31-4e51-ad98-b81bbcd16be8.png)

## Creating a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off"
```
df['loan_status'].unique()  
```
![image](https://user-images.githubusercontent.com/89992872/132103991-6df84609-a7b4-427b-aef8-2af376d6f7c2.png)
```
df['loan_repaid']=df['loan_status'].map({'Fully Paid':1,'Charged Off':0})  
df[['loan_repaid','loan_status']]  
```
![image](https://user-images.githubusercontent.com/89992872/132104009-ab2206ff-26a1-4971-8f89-ccd3297fce32.png)

## Creating a bar plot to show the correlation of the numeric features to the new loan_repaid column.
```
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot.bar()  
```
![image](https://user-images.githubusercontent.com/89992872/132104033-0186599c-3621-427a-a23a-5846d1db6966.png)

## Data Preprocessing

## Exploring Missing Data

## Length of dataframe
```
len(df)
```
![image](https://user-images.githubusercontent.com/89992872/132104079-31f8d60f-b9f2-4c24-9f01-06e5e10681e6.png)

## Series that displays the total count of missing values per column
```
df.isnull().sum()  
```
![image](https://user-images.githubusercontent.com/89992872/132104099-772db9a3-fd8d-4672-8956-ac2a1da6eee8.png)

## In percentage
```
100*df.isnull().sum()/len(df)  
```
![image](https://user-images.githubusercontent.com/89992872/132104116-9fd184a4-972f-4817-87e0-3687967b8275.png)

## Examining emp_title and emp_length to see whether it will be okay to drop them
```
feat_info('emp_title')  
print('\n')  
feat_info('emp_length')  
```
![image](https://user-images.githubusercontent.com/89992872/132104153-c15acf57-712b-42e8-ae8e-9fb78dd94256.png)

## Number of unique employment job titles
```
df['emp_title'].nunique()  
```
![image](https://user-images.githubusercontent.com/89992872/132104159-23b26e2d-466b-4994-a070-5528ad8392dd.png)
```
df['emp_title'].value_counts()
```
![image](https://user-images.githubusercontent.com/89992872/132104168-02c8bee7-81c1-4a87-9db3-2fa4371d24c2.png)

## Removing emp_title column as there are too many unique job titles to try to convert this to a dummy variable feature
```
df=df.drop('emp_title', axis=1)  
```

##  Creating a count plot of the emp_length feature column
```
sorted(df['emp_length'].dropna().unique())
```
![image](https://user-images.githubusercontent.com/89992872/132104209-1da9ef11-20c4-4f41-bf48-520db3d1e6fd.png)

```
plt.figure(figsize=(12,4))  
lengths = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years', '10+ years']  
sns.countplot(x='emp_length', data=df, order=lengths)  
```
![image](https://user-images.githubusercontent.com/89992872/132104215-eef86e1d-a55f-4fee-8596-c192aec3db59.png)

##  Countplot with a hue separating Fully Paid vs Charged Off
```
plt.figure(figsize=(12,4))  
lengths = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years', '10+ years']  
sns.countplot(x='emp_length', data=df, order=lengths, hue='loan_status')  
```
![image](https://user-images.githubusercontent.com/89992872/132104233-17255042-39ff-41a5-a85f-0beb5fdc7388.png)

## Exploring what percent of people per employment category didn't pay back their loan
```
df.groupby('loan_status')['emp_length'].value_counts() 
```
![image](https://user-images.githubusercontent.com/89992872/132104500-3478137b-1984-4806-b51a-a03e2dae1a4c.png)
```
emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']  
emp_co  
```
![image](https://user-images.githubusercontent.com/89992872/132104555-4ae0cb5d-f587-4432-8184-cce63bd180d3.png)
```
emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']  
emp_fp  
```
![image](https://user-images.githubusercontent.com/89992872/132104561-d4fdcb93-f193-4af2-872f-580c880e58fc.png)

## Percentage of charged off loans per emp_length
```
emp_len = emp_co/(emp_co+emp_fp)  
emp_len  
```
![image](https://user-images.githubusercontent.com/89992872/132104586-b63545f8-22fa-4ad2-8db8-6601c0e95b6b.png)
```
emp_len.plot(kind='bar')  
```
![image](https://user-images.githubusercontent.com/89992872/132104595-11af64cf-41e8-4b71-ac1a-fa5d336e3c43.png)

## Dropping the emp_length column as charge off rates are similar across all employment lengths
```
df=df.drop('emp_length', axis=1)  
```
## Revisiting the DataFrame to see what columns still have missing values
```
df.isnull().sum()  
```
![image](https://user-images.githubusercontent.com/89992872/132104641-f03e4514-4fe6-4f56-8fdb-6dbb70c08929.png)

## Reviewing the 'title' column vs the 'purpose' column
```
df['purpose'].head(10) 
```
![image](https://user-images.githubusercontent.com/89992872/132104652-23db0854-a848-4d88-9bdb-5aa32b5d367a.png)

```
df['title'].head(10)  
```
![image](https://user-images.githubusercontent.com/89992872/132104661-21f4bf93-f0f4-4637-8d41-c9db846de338.png)

## Dropping the title column as it is simply a string subcategory/description of the purpose column
```
df=df.drop('title', axis=1)  
```
## Exploring mort_acc column
```
feat_info('mort_acc')
```
![image](https://user-images.githubusercontent.com/89992872/132104702-3e3bd4bb-6361-42d6-982a-35b3c5f2a6ec.png)
```
df['mort_acc'].value_counts() 
```
![image](https://user-images.githubusercontent.com/89992872/132104711-7153aefc-a44c-4d46-b4f3-a105d6f5250b.png)

## Reviewing the other columns to see which most highly correlates to mort_acc
```
df.corr()['mort_acc'].sort_values()  
```
![image](https://user-images.githubusercontent.com/89992872/132104730-894ec263-a89d-4403-a7fb-b696b623f157.png)

## The total_acc feature correlates with mort_acc. I'll group the  dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry
```
df.groupby('total_acc').mean()['mort_acc']  
```
![image](https://user-images.githubusercontent.com/89992872/132104776-d01a29f7-7454-4092-a754-61d721c1cf67.png)

## Filling in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then the mean value corresponding to its total_acc value from the Series above will be used.
```
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']  

def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc for a row.
    Returns the average mort_acc for the corresponding total_acc
    if mort_acc is Nan and returns mort_acc otherwise.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else: 
        return mort_acc
```
```
df['mort_acc']=df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/89992872/132105534-debc3c74-96e7-4c72-91ae-d42bce133550.png)

## Removing revol_util and the pub_rec_bankruptcies as they have missing data points, but they account for less than 0.5% of the total data
```
df=df.dropna()
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/89992872/132105576-99044aa8-e076-4206-a176-af6b5cf7c98f.png)

## Converting Categorical variables into Dummy Variables

## Listing all columns that are non-numeric
```
df.select_dtypes(include=['object']).columns
```
![image](https://user-images.githubusercontent.com/89992872/132105613-346a6e48-d8d5-4a69-9af6-9f6cd408cfdb.png)

## Going through 'term' feature
```
df['term'].value_counts()
```
![image](https://user-images.githubusercontent.com/89992872/132105640-e2f45680-2be0-46b2-830c-3f0f69f25fff.png)
```
df['term'] = df['term'].apply(lambda term: int(term[:3]))
```

## Dropping 'grade' feature as it's included in the subgrade
```
df = df.drop('grade',axis=1)
```
## Converting subgrade into dummy variables and concatenating the new columns to the dataframe
```
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns
```
![image](https://user-images.githubusercontent.com/89992872/132105733-5a9cc7d5-10bc-483b-968b-544759d69d6d.png)

## Converting these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenating them with the original dataframe.
```
dummies = pd.get_dummies(df[ ['verification_status', 'application_type','initial_list_status','purpose']],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)
```

## Converting 'home_ownership' column into dummy variables 
```
df['home_ownership'].value_counts()
```
![image](https://user-images.githubusercontent.com/89992872/132105824-ee20fa8e-fc85-4851-82ed-4c90eea4e511.png)


## Replacing NONE and ANY with OTHER to get just 4 categories, MORTGAGE, RENT, OWN, OTHER
```
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
```
```
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)
```
## Converting address into just zipcode
```
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
```
```
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)
```

## Dropping the 'issue_d' column as we wouldn't know beforehand whether or not a loan would be issued when using the model
```
df = df.drop('issue_d',axis=1)
```

## Extracting year from 'earliest_cr_line' and setting this new data to a feature column called 'earliest_cr_year'.
```
df['earliest_cr_year']=df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df=df.drop('earliest_cr_line',axis=1)
```
## Train Test Split
```
from sklearn.model_selection import train_test_split
```
```
df = df.drop('loan_status',axis=1)
```
```
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
```
## Normalizing the data
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
## Creating the model
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
```
model = Sequential()

model.add(Dense(79,  activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
```
```
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
```
```
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[early_stop]
          )
```      
## Evaluating Model Performance
```
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
```
![image](https://user-images.githubusercontent.com/89992872/134999877-cefcba6b-ee88-4a51-8f80-2ac34c1514bb.png)

## Classification Report and Confusion Matrix
```
from sklearn.metrics import classification_report,confusion_matrix
```
```
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
print(classification_report(y_test,predictions))
```
![image](https://user-images.githubusercontent.com/89992872/135000030-75ce0fcf-5189-40ae-a540-7ec13b4110cc.png)
```
confusion_matrix(y_test,predictions)
```
![image](https://user-images.githubusercontent.com/89992872/134999962-30305fbc-bb6b-4671-b17a-a6727374792c.png)
