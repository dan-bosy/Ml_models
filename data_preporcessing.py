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
