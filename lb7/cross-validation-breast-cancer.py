from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd

breast_cancer_dataset = fetch_ucirepo(id=17)

X = breast_cancer_dataset.data.features
y = breast_cancer_dataset.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

# print(breast_cancer_dataset.metadata)
# print(breast_cancer_dataset.variables)

