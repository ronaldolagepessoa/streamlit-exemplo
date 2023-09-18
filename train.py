import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle


def train_model(df: pd.DataFrame):
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(min_max_scaler.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    pickle.dump(model, open("models/my_model.pkl", "wb"))
    pickle.dump(min_max_scaler, open("models/min_max.pkl", "wb"))
    return cm, accuracy, model
