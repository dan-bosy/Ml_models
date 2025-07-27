from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Each item = [weight_in_grams, smoothness (0=rough, 1=smooth)]
X = [
    [150, 1],  # apple
    [170, 1],  # apple
    [140, 1],  # apple
    [130, 0],  # banana
    [120, 0],  # banana
    [110, 0],  # banana
    [160, 1],  # orange
    [180, 0],  # orange
    [165, 1]   # orange
]

# Labels: 0=Apple, 1=Banana, 2=Orange
y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# multi class classification 
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs').fit(X, y)

new_data = [[155, 1]]
new_fruits = ['apple', 'banana', 'orange']

# prediction 
print('prediction:', new_fruits[model.predict(new_data)[0]])
