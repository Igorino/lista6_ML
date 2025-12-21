import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("diabetes.xls")

print(df.head())
print(df.columns)

x = df.drop(columns=["Outcome"])
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = (y_pred == y_test).mean()
print("Acurácia:", acc)

print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(y_test, y_pred, digits=4))

