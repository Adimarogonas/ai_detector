import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ipywidgets import interact
from sklearn.metrics import log_loss, confusion_matrix

import pickle
def save_model(filename)
    # Save the model to a file
    model_filename = filename
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)


#Calculates rate of false positive and false negative
def calc_loss():
    # Load the dataset
    data = []
    for i in range(len(fakeX)):
        data.append((realX[i], realY[i], realZ[i], "real"))
        data.append((fakeX[i], fakeY[i], fakeZ[i], "fake"))
    # Split the dataset into features and labels
    X = [(x[0], x[1], x[2]) for x in data]
    y_true = [1 if x[3] == "fake" else 0 for x in data]

    # Generate predicted classes for the dataset
    y_pred = model.predict(X)

    # Compute the log loss of the predictions
    loss = log_loss(y_true, y_pred)

    # Compute the confusion matrix of the predictions
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute the false negative rate and false positive rate
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)
    print(f"False Negative: {fn_rate}")
    print(f"False Positive: {fp_rate}")
#Load the dataset - This contains all the data gathered from gather_data.py
data = pd.read_csv('results_1.csv')

#Set our relevant values for data analysis
realX = data["real_perplexity"]
realY = data["real_burstiness"]
realZ = data["real_ari"]

fakeX = data["fake_perplexity"]
fakeY = data["fake_burstiness"]
fakeZ = data["fake_ari"]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(realX,realZ,realY, c=realZ, cmap='Greens')
ax.scatter3D(fakeX, fakeZ,fakeY, c=fakeZ, cmap='Reds')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


real_data = []
fake_data = []
#setup our data for training
for i in range(len(fakeX)):
    #The XYZ order can be changed and may provide different results
    real_data.append((realX[i], realZ[i], realY[i]))
    fake_data.append((fakeX[i], fakeZ[i], fakeY[i]))

# Define features and labels
X = real_data + fake_data
y = [0]*len(real_data) + [1]*len(fake_data)#AI generated will predict [1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict([(25, 18.05, 17.5)]))
# Evaluate model on test set
score = model.score(X_test, y_test)
print("Accuracy on test set: {:.2f}%".format(score*100))
calc_loss()

#Save your model here(oncomment to do)
#save_model("your_model.pkl")