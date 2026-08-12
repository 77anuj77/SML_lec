import pandas as pd
data= pd.read_csv("iphone_purchase_records.csv")
print(data.head())

#the features is independent of each other and this is called as conditional independence
#it is the fastest algorithm amoung the classification machine learning modelss
#therefore this is unrealistic like-- free money  treated as isolated terms though they are corrlated

'''
Gaussian -- for the continous real-values like temperature income medical values
multinominal -- count data word frequency market basket analysis
bernoullis -- binary indicators like true false 
'''

print(data.columns.to_list())
print(data.shape)

data.info()
data.describe()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("purchase distribution")
print(data["Purchase Iphone"].value_counts())
print("\npercentage: ")
print(data['Purchase Iphone'].value_counts(normalize=True).round(2)*100)

'''
stratify -- for training and testing data it ensure that the both the feature will get equal opportunity in traing
joint plot -- combines the histogram and scatter plot
'''

sns.jointplot(x='Age', y='Salary', data=data, hue='Purchase Iphone', kind='scatter')
plt.suptitle('Age vs Salary coloured by Purchase Decision', y=1.02)
plt.show()

fig, axes=plt.subplots(1,2,figsize=(12,4))
fig.suptitle("Feature Distribution by Purchase Decision", fontsize=13, fontweight='bold')
sns.boxplot(data=data, x='Purchase Iphone', y='Age', palette=['steelblue', 'tomato'], ax=axes[0])
axes[0].set_title("age by Purchase Decision")
axes[0].set_xlabel("Purchase (0=No, 1=Yes)")
axes[0].set_ylabel("Age")

#salary distribution
sns.boxplot(data=data, x='Purchase Iphone', y='Salary', palette=["steelblue", "tomato"], ax=axes[1])
axes[1].set_title("Salary by Purchase Decision")
axes[1].set_xlabel("Purchase (0=No, 1=Yes)")
axes[1].set_ylabel("Salary")
plt.tight_layout()
plt.show()

X= data.iloc[:, [1,2]].values
y=data.iloc[:, -1].values
print(f"Feature Matrix Shape:{X.shape} , Target array shape{y.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Training set size:", X_train.shape[0])
print("Test set size    :", X_test.shape[0])

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)

print("Training set ( first 3 rows after scaling): ")
print(np.round(X_train))

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier.fit(X_train, y_train)

print("Model trained Succesfully")
print("Classes learnes:",classifier.classes_)
print("Prior probabilities:", np.round(classifier.class_prior_, 3)) #this is the probability of the model for all three features

y_pred=classifier.predict(X_test)
print("Predicted labels (first 10):", y_pred[:10])
print("Actual labels (first 10):", y_test[:10])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix ")
print(cm)

disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no purchase', 'purchase'])
disp.plot(cmap="Blues")
plt.title("Confussion matrix - Naive Bayes")
plt.show()

#accuracy= tp+tn / all 4 
#recall= tp/ tp + fp it is like the from the purchase i predicted how many are right (sales Guy)
#precesion = tp /tp+fn and it is like from the actual value how many prediction are right
#f1-score= 2(precison*recall)/precision + recall
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Accuracy: {acc*100:.2f}%")

from sklearn.metrics import classification_report
print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))



