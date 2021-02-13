import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


#Data Loading
train= pd.read_excel('model/train.xlsx')
val= pd.read_excel('model/test.xlsx')

#Filling Missing Values
train.drop(['id','gender', 'type_of_residence', 'address'], 1, inplace=True)
val.drop(['id','gender', 'type_of_residence', 'address'], 1, inplace=True)

train.loc[train.interest_rate.isna(), 'interest_rate'] = (train.interest_due * 100) / train.loan_amount

train.loc[train.bank.isna(), 'bank'] = 'Access Bank'
val.loc[val.bank.isna(), 'bank'] = 'Access Bank'

train.loc[train.card_expiry.isna(), 'card_expiry'] = 32023.0
val.loc[val.card_expiry.isna(), 'card_expiry'] = 32023.0


train['card_expiry_month'] = train.card_expiry.map(lambda x: str(int(x))[:-4]).astype(int)
train['card_expiry_year'] = train.card_expiry.map(lambda x: str(int(x))[-4:]).astype(int)
train.drop('card_expiry', 1, inplace=True)

val['card_expiry_month'] = val.card_expiry.map(lambda x: str(int(x))[:-4]).astype(int)
val['card_expiry_year'] = val.card_expiry.map(lambda x: str(int(x))[-4:]).astype(int)
val.drop('card_expiry', 1, inplace=True)


date_column= ['date_of_birth', 'work_start_date']
def extract_date(train,cols,):
    for x in cols:
        train[x +'_year'] = train[x].dt.year
    train.drop(columns=date_column,axis=1,inplace=True)
extract_date(train,date_column)

def extract_data(val,cols,):
    for x in cols:
        val[x +'_year'] = val[x].dt.year
    val.drop(columns=date_column,axis=1,inplace=True)
extract_data(val,date_column)


acc_column= ['first_account', 'last_account']
def extract(train,col,):
    for x in col:
        train[x +'_year'] = train[x].dt.year
        train[x +'_month'] = train[x].dt.month
        train[x +'_day'] = train[x].dt.day
        train[x +'_quarter'] = train[x].dt.quarter
    train.drop(columns=acc_column,axis=1,inplace=True)
extract(train, acc_column)

def extract_val(val,col,):
    for x in col:
        val[x +'_year'] = val[x].dt.year
        val[x +'_month'] = val[x].dt.month
        val[x +'_day'] = val[x].dt.day
        val[x +'_quarter'] = val[x].dt.quarter
    val.drop(columns=acc_column,axis=1,inplace=True)
extract_val(val, acc_column)


train['tenor'] = train['tenor'].replace(['4 weeks', '3 weeks', '1 months'], ['28 days', '21 days', '30 days'])
train['tenor'] = train.tenor.map(lambda x: x.split(' ')[0]).astype(int)


val['tenor'] = val['tenor'].replace(['4 weeks', '3 weeks', '1 months'], ['28 days', '21 days', '30 days'])
val['tenor'] = val.tenor.map(lambda x: x.split(' ')[0]).astype(int)

train.proposed_payday = train.proposed_payday.replace(['4 weeks', '3 weeks', '1 months', '2 months', '4 months'], ['28 days', '21 days', '30 days','60 days', '120 days' ])
train.proposed_payday = train.proposed_payday.map(lambda x: x.split(' ')[0]).astype(int)

val.proposed_payday = val.proposed_payday.replace(['4 weeks', '3 weeks', '1 months', '2 months', '4 months'], ['28 days', '21 days', '30 days','60 days', '120 days' ])
val.proposed_payday = val.proposed_payday.map(lambda x: x.split(' ')[0]).astype(int)

train.loc[train.work_start_date_year.isna(), 'work_start_date_year'] = 2018.0


target_map = {
	'SETTLED': 2,
    'PAST DUE': 5}
train.status.replace(target_map, inplace = True)

col= ['card_network', 'bank', 'tier', 'selfie_id_check', 'marital_status', 'no_of_dependent', 'educational_attainment', 'employment_status', 'sector_of_employment', 'monthly_net_income', 'purpose', 'location']

for y in col:
  le= LabelEncoder()
  train[y]= le.fit_transform(train[y].astype(str))

for y in col:
  le= LabelEncoder()
  val[y]= le.fit_transform(val[y].astype(str))

y= train['status']
X= train.drop('status', axis= 1)

train_x, val_x, train_y, val_y = train_test_split(X.values, y.values, test_size=0.2, random_state=99)

gbc_model= GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 200, max_depth= 2, min_samples_split=7)
gbc_model.fit(train_x, train_y)

pred_y = gbc_model.predict(val_x)
print(classification_report(val_y, pred_y))

import pickle
pickle.dump(gbc_model, open("lendsqrdatapredmode.pkl", "wb"))




