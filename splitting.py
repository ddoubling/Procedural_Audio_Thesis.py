import os
from sklearn.model_selection import train_test_split

"""Splitter for training and test data from single folder to specified folders"""
os.listdir( r"C:\Users\dwill\Desktop\drum\audio")
X = y= os.listdir( r"C:\Users\dwill\Desktop\drum\audio")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
for x in X_train:
    os.rename( r"C:\Users\dwill\Desktop\drum\audio/" +x ,  r"C:\Users\dwill\Desktop\drum\train/" +x)
for x in X_test:
    os.rename( r"C:\Users\dwill\Desktop\drum\audio/"+x ,  r"C:\Users\dwill\Desktop\drum\test/"+x)

