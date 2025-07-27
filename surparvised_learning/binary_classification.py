from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# features : [study_hours, sleep_hours]
# datasets : is student or not?, 1 = yes, 0 = no
x = [
    [5, 7],
    [1, 4],
    [3, 5],
    [8, 8],
    [2, 3]
]
y = [1, 0, 0, 1, 0]

# train test the datasets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# classification model
model = LogisticRegression().fit(x_train, y_train)

# prediction 
y_pred = model.predict(x_test)

# evaluation 
print('prediction accuracy:', accuracy_score(y_test, y_pred))
