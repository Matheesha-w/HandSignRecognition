import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

data_dict = pickle.load(open('data.pickle', 'rb'))

X = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

print("Class distribution:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
