import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def get_feature_importances(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.ylabel('Importance Score')
    plt.show()
    selected_features = feature_importances.nlargest(38).index

    return selected_features
