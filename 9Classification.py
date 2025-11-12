import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score,f1_score, recall_score, log_loss
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
print(df.head())
print(df)