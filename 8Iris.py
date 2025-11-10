# import seaborn as sns
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset('iris')
# print(df.head())

# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# model = LogisticRegression(max_iter=1000)
# model.fit(X, y)

# y_pred = model.predict(X)

# conf = confusion_matrix(y, y_pred)
# print("Confusion Matrix:\n", conf, "\n")

# acc = accuracy_score(y, y_pred)
# print("Accuracy:", acc, "\n")

# report = classification_report(y, y_pred)
# print("Classification Report:\n", report)



import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')
print(df.head())

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

conf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf, "\n")

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc, "\n")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


