import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


DATA_FILE = "hand_gestures_data.csv"

print("Loading dataset...")

data = pd.read_csv(DATA_FILE)

print(f"Total Samples : {len(data)}")


X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




print("Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 40)
print(f"Accuracy : {accuracy*100:.2f}%")
print("=" * 40)


dump(model, "gesture_model.pkl")

print("Model saved successfully!")
