import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Define the categories
Categories = ['apple', 'mango', 'banana', 'strawberry', 'grape']
# Initialize arrays
flat_data_arr = []  # Input array
target_arr = []  # Output array
datadir =r"C:\msinternship\Fruits Classification\train"# Update with your dataset path
#Load the dataset
for i in Categories:
    print(f'Loading... category: {i}')
    path = os.path.join(datadir, i)
    print(path)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'Loaded category: {i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
print(df.shape)
# Input and output data
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Training started')
svc = svm.SVC(probability=True)
svc.fit(x_train, y_train)
print('Training completed')
# Predictions
y_pred = svc.predict(x_test)
# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"The model is {accuracy*100:.2f}% accurate")
print(classification_report(y_test, y_pred, target_names=Categories))
# Predict on a new image
path =r"C:\msinternship\Fruits Classification\train\Apple\Apple (98).jpeg" # Change to the path of your test image
img = imread(path)
plt.imshow(img)
plt.show()
img_resize = resize(img, (150, 150, 3))
l = [img_resize.flatten()]
probability = svc.predict_proba(l)
for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100:.2f}%')
print("The predicted image is: " + Categories[svc.predict(l)[0]])
