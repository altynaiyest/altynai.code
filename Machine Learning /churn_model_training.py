import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


df = pd.read_csv('/content/Churn_Modelling.csv')
df.head()

df.columns = ['row_number', 'customer_id', 'surname', 'creditscore', 'geography',
       'gender', 'age', 'tenure', 'balance', 'num_of_products', 'has_crcard',
       'isactive_member', 'estimated_salary', 'exited']

feature_cols = ['creditscore', 'geography', 'gender', 'age', 'tenure',
                'balance', 'num_of_products', 'has_crcard', 'isactive_member',
                'estimated_salary']

X = df[feature_cols]
y = df['exited']


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

from sklearn.preprocessing import StandardScaler  # Импортируем StandardScaler

numerical_features = ['creditscore', 'age', 'tenure', 'balance', 'num_of_products', 'has_crcard', 'isactive_member', 'estimated_salary']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Стандартизация числовых признаков
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ], remainder='passthrough')



pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(random_state=16))])
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Определим пайплайн, как и раньше
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=16))
])

# Настраиваем параметры для каждой модели отдельно
param_grid = [
    {
        'classifier': [LogisticRegression(random_state=16)],
        'classifier__C': [0.1, 1.0, 10.0]
    },
    {
        'classifier': [RandomForestClassifier(random_state=16)],
        'classifier__n_estimators': [50, 100, 200],  # Пример параметров для случайного леса
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'classifier': [GradientBoostingClassifier(random_state=16)],
        'classifier__n_estimators': [50, 100, 200],  # Пример параметров для градиентного бустинга
        'classifier__learning_rate': [0.01, 0.1, 0.5]
    }
]

# Кросс-валидация и поиск оптимальной модели
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Лучший классификатор: {best_model.named_steps['classifier']}")

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

y_pred = best_model.predict(X_test)  # Используем best_model вместо pipeline
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Предсказание вероятностей и расчет AUC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc_score:.2f}")

# Построение ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc=4)
plt.show()


# Import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming `cnf_matrix` is defined, as in the previous confusion matrix code
class_names = [0, 1]  # Names of the classes, can be modified as needed

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

from sklearn.metrics import classification_report

# Предполагаем, что у вас уже есть y_test и y_pred
target_names = ['without exit', 'with exit']  # Замените названия на подходящие для вашей задачи
print(classification_report(y_test, y_pred, target_names=target_names))
