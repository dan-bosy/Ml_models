import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Create the DataFrame
data = {
    'Age': [22, 38, None, 45, 28],
    'Salary': [20000, None, 50000, 60000, 40000],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Step 2: Fill missing values with column means
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

mean_salary = df['Salary'].mean()
df['Salary'] = df['Salary'].fillna(mean_salary)

# Step 3: Scale the numerical features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[['Age', 'Salary']])
df[['Age', 'Salary']] = scaled

# Step 4: Encode the target variable
encoder = LabelEncoder()
encoded = encoder.fit_transform(df['Purchased'])
df['Purchased'] = encoded

# Step 5: Split into training and test sets
X = df[['Age', 'Salary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# datasets before converting to data frame
data = {
    'gender': ['Male', 'Female', None, 'Female', 'Male', None],
    'age': [23, 45, 31, None, 52, 40],
    'income': [35000, None, 45000, 50000, 52000, 48000],
    'purchased': [0, 1, 0, 1, 1, 0]
}
# Data frame
df = pd.DataFrame(data)
df = df.replace({None: np.nan})

# encoder, scaler and imputers
encoder = LabelEncoder()
scaler = StandardScaler()  # instantiated properly
num_imputer = SimpleImputer(strategy='mean')
text_imputer = SimpleImputer(strategy='most_frequent')

# handling missing value
df['gender'] = text_imputer.fit_transform(df[['gender']]).ravel()
df['age'] = num_imputer.fit_transform(df[['age']]).ravel()
df['income'] = num_imputer.fit_transform(df[['income']]).ravel()

# encoding and scaling
df['gender'] = encoder.fit_transform(df['gender']).ravel()  # pass 1D array
df['income'] = scaler.fit_transform(df[['income']]).ravel()  # use double brackets for 2D input

print(f"age {df['age']}\n gender {df['gender']}\n income {df['income']}")
