import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

df = pd.read_csv("diabetes.xls")

print(df.head())
print(df.columns)

x = df.drop(columns=["Outcome"])
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

def eval_model(name, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 80)
    print(f"Modelo: {name}")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de confusão [[TN FP],[FN TP]]:")
    print(cm)
    print("\nRelatório:")
    print(classification_report(y_test, y_pred, digits=4))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")

    return acc

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

print("[[TN, FP],")
print("[FN, TP]]")

print("\nRelatório:")
print(classification_report(y_test, y_pred, digits=4))

