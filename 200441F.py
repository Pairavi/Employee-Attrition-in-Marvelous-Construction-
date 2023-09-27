#!/usr/bin/env python
# coding: utf-8

# In[885]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[886]:


chatterbox = pd.read_csv("employees.csv")


# In[887]:


chatterbox.dtypes


# In[888]:


chatterbox.shape


# ### Filling missing values

# In[889]:


chatterbox.isnull().sum()


# In[890]:


chatterbox['Year_of_Birth'] = pd.to_numeric(chatterbox['Year_of_Birth'], errors='coerce')


# In[891]:


chatterbox.loc[chatterbox['Year_of_Birth'] < 1900, 'Year_of_Birth'] = np.nan


# In[892]:


# Define the mapping dictionary
status_mapping = {
    'Active': 1,
    'Inactive': 0
}

# Encode the "Marital_Status" column
chatterbox['status_Encoded'] = chatterbox['Status'].map(status_mapping)


# In[893]:


# Define the mapping dictionary
gender_mapping = {
    'Male': 1,
    'Female': 0
}

# Encode the "Marital_Status" column
chatterbox['gender_Encoded'] = chatterbox['Gender'].map(gender_mapping)


# In[894]:


# Define the mapping dictionary
employment_type_mapping = {
    'Permanant': 1,
    'Contarct Basis': 0
}

# Encode the "Marital_Status" column
chatterbox['employment_type_Encoded'] = chatterbox['Employment_Type'].map(employment_type_mapping)


# In[895]:


# Convert 'Date_Joined' to datetime format
chatterbox['Date_Joined'] = pd.to_datetime(chatterbox['Date_Joined'], format='%m/%d/%Y')


# Find the minimum date as the reference date
min_date = chatterbox['Date_Joined'].min()

# Convert 'Date_Joined' to the number of days since the reference date
chatterbox['Date_Joined_NumDays'] = (chatterbox['Date_Joined'] - min_date).dt.days



# In[896]:


# Define the mapping dictionary
marital_status_mapping = {
    'Married': 1,
    'Single': 0
}

# Encode the "Marital_Status" column
chatterbox['Marital_Status_Encoded'] = chatterbox['Marital_Status'].map(marital_status_mapping)


# In[897]:


chatterbox_copy = chatterbox.copy()


# In[898]:


df_null_values = chatterbox[(chatterbox['Marital_Status_Encoded'].isnull()) & (chatterbox['Year_of_Birth'].isnull())]

print(df_null_values.shape)


# In[899]:


df_1 = chatterbox_copy.dropna(how='all', subset=['Marital_Status_Encoded', 'Year_of_Birth'])


# In[900]:


df_1.shape


# In[901]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Split the data into training and testing sets
train = df_1[df_1['Marital_Status_Encoded'].notnull() & df_1['Year_of_Birth'].notnull()]
test_1 = df_1[df_1['Marital_Status_Encoded'].isnull()]
print(train.shape)
print(test_1.shape)

# Select the features for training
features = ['Employee_No', 'Employee_Code', 'Religion_ID', 'Designation_ID', 'gender_Encoded', 'Date_Joined_NumDays', 'status_Encoded', 'employment_type_Encoded','Year_of_Birth']
X_train = train[features]
X_test_1 = test_1[features]
y_train_marital = train['Marital_Status_Encoded']

# Train a Decision Tree classifier for 'Marital_Status'
clf_marital = DecisionTreeClassifier()
clf_marital.fit(X_train, y_train_marital)
marital_predictions = clf_marital.predict(X_test_1)


# Fill in the missing values in the original DataFrame
test_1['Marital_Status_Encoded'] = pd.Series(marital_predictions, index=X_test_1.index)


# Define the mapping dictionary
marital_status_mapping = {
    1 : 'Married',
    0 : 'Single'
}

# Encode the "Marital_Status" column
test_1['Marital_Status'] = test_1['Marital_Status_Encoded'].map(marital_status_mapping)
#chatterbox['Marital_Status'] = pd.Series(test_1['Marital_Status'], index=X_test_1.index)
chatterbox.update(test_1['Marital_Status'].rename('Marital_Status'))


combined_df = pd.concat([train, test_1], ignore_index=True)

# Updated DataFrame with imputed values
print(combined_df.shape)


# In[902]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Split the data into training and testing sets
train = combined_df[combined_df['Marital_Status_Encoded'].notnull() & combined_df['Year_of_Birth'].notnull()]
test_2 = df_1[df_1['Year_of_Birth'].isnull()]

print(train.shape)
print(test_2.shape)

# Select the features for training
features = ['Employee_No', 'Employee_Code', 'Religion_ID', 'Designation_ID', 'gender_Encoded', 'Date_Joined_NumDays', 'status_Encoded', 'employment_type_Encoded','Marital_Status_Encoded']
X_train = train[features]
X_test_2 = test_2[features]
y_train_birth = train['Year_of_Birth']

# Train a Decision Tree classifier for 'Year_of_Birth'
clf_birth = DecisionTreeClassifier()
clf_birth.fit(X_train, y_train_birth)
birth_predictions = clf_birth.predict(X_test_2)

# Fill in the missing values in the original DataFrame
test_2['Year_of_Birth'] = pd.Series(birth_predictions, index=X_test_2.index)

chatterbox.update(test_2['Year_of_Birth'].rename('Year_of_Birth'))
    
chatterbox_cleaned = pd.concat([train, test_2], ignore_index=True)

# Updated DataFrame with imputed values
print(chatterbox_cleaned.shape)


# In[903]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Split the data into training and testing sets
train = chatterbox_cleaned
test_3 = df_null_values

print(train.shape)
print(test_3.shape)

# Select the features for training
features = ['Employee_No', 'Employee_Code', 'Religion_ID', 'Designation_ID', 'gender_Encoded', 'Date_Joined_NumDays', 'status_Encoded', 'employment_type_Encoded']
X_train = train[features]
X_test_3 = test_3[features]

# Train a Decision Tree classifier for 'Year_of_Birth'
clf_birth = DecisionTreeClassifier()
y_train_birth = train['Year_of_Birth']
clf_birth.fit(X_train, y_train_birth)
birth_predictions = clf_birth.predict(X_test_3)

# Train a Decision Tree classifier for 'Marital_Status'
clf_marital = DecisionTreeClassifier()
y_train_marital = train['Marital_Status_Encoded']
clf_marital.fit(X_train, y_train_marital)
marital_predictions = clf_marital.predict(X_test_3)

# Fill in the missing values in the original DataFrame
test_3['Year_of_Birth'] = birth_predictions
test_3['Marital_Status_Encoded'] = marital_predictions

# Encode the "Marital_Status" column
test_3['Marital_Status'] = test_3['Marital_Status_Encoded'].map(marital_status_mapping)

chatterbox.update(test_3['Marital_Status'].rename('Marital_Status'))

chatterbox.update(test_3['Year_of_Birth'].rename('Year_of_Birth'))


# Concatenate the updated DataFrame with the original training DataFrame
df_2 = pd.concat([train, test_3], ignore_index=True)

# Updated DataFrame with imputed values
print(df_2.shape)


# In[904]:


chatterbox = chatterbox.drop(['status_Encoded','gender_Encoded','employment_type_Encoded','Date_Joined_NumDays','Marital_Status_Encoded'], axis=1)


# In[905]:


chatterbox.shape


# In[906]:


#impute missing values of Date_Resigned and Inactive date based on these 2 and status
chatterbox['Date_Resigned'] = chatterbox['Date_Resigned'].mask(chatterbox['Date_Resigned'] == '0000-00-00', chatterbox['Inactive_Date'])
chatterbox['Inactive_Date'] = chatterbox['Inactive_Date'].mask(chatterbox['Inactive_Date'] == '0000-00-00', chatterbox['Date_Resigned'])
chatterbox['Date_Resigned'] = chatterbox['Date_Resigned'].mask(chatterbox['Date_Resigned'] == '\\N', chatterbox['Inactive_Date'])
chatterbox['Date_Resigned'] = chatterbox['Date_Resigned'].replace('0000-00-00', '\\N')
chatterbox['Inactive_Date'] = chatterbox['Inactive_Date'].replace('0000-00-00', '\\N')


# In[907]:


chatterbox.isnull().sum()


# In[908]:


#chatterbox.head(500)


# ### Checking outliers

# In[909]:


# Define the numerical attributes
numerical_attributes = ['Employee_No', 'Employee_Code', 'Religion_ID', 'Designation_ID','Year_of_Birth']

# Define the threshold value for capping outliers
threshold = 3

# Loop through each numerical attribute
for attribute in numerical_attributes:
    # Calculate the upper and lower bounds for outliers
    upper_bound = chatterbox[attribute].mean() + threshold * chatterbox[attribute].std()
    lower_bound = chatterbox[attribute].mean() - threshold * chatterbox[attribute].std()

    # Cap the outliers at the bounds
    chatterbox[attribute] = chatterbox[attribute].clip(lower=lower_bound, upper=upper_bound)

    # Plot the histogram
    plt.hist(chatterbox[attribute], bins=10)
    plt.xlabel(attribute)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {attribute}')
    plt.show()


# In[910]:


chatterbox.to_csv('employee_preprocess_200441F.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




