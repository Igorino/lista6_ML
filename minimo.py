import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# Repositório em https://github.com/Igorino/lista6_ML

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


linear = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=3000))
])

l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000))
])

elastic = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, C=1.0,
        solver="saga", max_iter=5000
    ))
])

acc_linear = eval_model("Linear (no penalty)", linear)
acc_l2 = eval_model("Linear + L2 (ridge)", l2)
acc_elastic = eval_model("Linear + Elastic Net", elastic)

print("\n" + "-" * 80)
print("Resumo (acurácia):")
print(f"Linear:      {acc_linear:.4f}")
print(f"L2 (ridge):  {acc_l2:.4f}")
print(f"Elastic Net: {acc_elastic:.4f}")

# Basicamente, o modelo linear simples e o L2 tiveram praticamente o mesmo resultado.
# O Elastic Net teve uma elve queda na acurácia, mas manteve o ROC-AUC alto.
