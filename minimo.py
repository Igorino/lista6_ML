import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes.xls")

print(df.head())
print(df.columns)

x = df.drop(columns=["Outcome"])
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = (y_pred == y_test).mean()
print("Acurácia:", acc)

