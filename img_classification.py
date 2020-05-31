import numpy as np 
import matplotlib.pyplot as plt 
from keras.preprocessing import image
from pathlib import Path 
import random
from mysvm import SVM
plt.style.use('seaborn')

p = Path("./images")
dirs = p.glob("*")

labels_dict = {"cats":0,"dogs":1,"horses":2,"humans":3}
image_data = []
labels = []	
for folder_name in dirs:
	# print(folder_name)
	label = str(folder_name).split('\\')[-1]

	for img_path in folder_name.glob("*.jpg"):
		img = image.load_img(img_path,target_size=(32,32))
		img_array = image.img_to_array(img)
		image_data.append(img_array)
		labels.append(labels_dict[label])

image_data = np.array(image_data,dtype='float32')/255.0#since image_data has float 0-255 so make them usigned int 0-255
labels = np.array(labels)

combined = list(zip(image_data,labels))
random.shuffle(combined)

#unzip
image_data[:],labels[:] = zip(*combined)

def show_image(img):
	plt.imshow(img)
	plt.axis('off')
	plt.show()

# for i in range(5):
# 	show_image(image_data[i])

##-----------using one-vs-one classification for SVM-------------------
image_data = image_data.reshape(image_data.shape[0],-1)
# print(image_data.shape)
# print(labels.shape)

classes = np.unique(labels)

def classWiseData(x,y):
	data = {}

	for i in range(len(classes)):
		data[i] = []

	for i in range(x.shape[0]):
		data[y[i]].append(x[i])

	for k in data.keys():
		data[k] = np.array(data[k])

	return data

data = classWiseData(image_data,labels)

def getDataPairForSVM(d1,d2):
	"""combines data of 2 classes into single one"""
	no_of_samples = d1.shape[0]+d2.shape[0]
	no_of_features = d1.shape[1]

	data_pair = np.zeros((no_of_samples,no_of_features))
	data_labels = np.zeros(no_of_samples)

	data_pair[:d1.shape[0],:] = d1
	data_pair[d1.shape[0]:,:] = d2

	data_labels[:d1.shape[0]] = -1
	data_labels[d1.shape[0]:] = +1

	return data_pair,data_labels

#train nC2 classifier

mySVM = SVM()#inported from mysvm.py

def trainSVMs(data):
	svm_classifiers = {}
	for i in range(len(classes)):
		svm_classifiers[i] = {}
		for j in range(i+1,len(classes)):
			xpair,ypair = getDataPairForSVM(data[i],data[j])
			wts,b,loss = mySVM.fit(xpair,ypair,itrations=1000,learning_rate=0.00001)
			# plt.plot(loss)
			# plt.show()
			svm_classifiers[i][j] = (wts,b)

	return svm_classifiers

svm_classifiers = trainSVMs(data)

def binaryPredict(x,w,b):
	z = np.dot(x,w.T)+b

	if z>=0:
		return 1
	else:
		return -1

def predict(x):
	count = np.zeros(len(classes))
	for i in range(len(classes)):
		for j in range(i+1,len(classes)):
			w,b = svm_classifiers[i][j]
			z = binaryPredict(x,w,b)

			if z==1:
				count[j] += 1
			else:
				count[i] += 1

	
	return np.argmax(count)

# print(predict(image_data[0]))
# print(labels[0])

def accuracy(x,y):
	count = 0
	
	for i in range(x.shape[0]):
		prediction = predict(x[i])

		if prediction==y[i]:
			count += 1

	return count/x.shape[0]

print(accuracy(image_data,labels))


from sklearn import svm
svm_classifier = svm.SVC(kernel='linear',C=1.0)
svm_classifier.fit(image_data,labels)
print(svm_classifier.score(image_data,labels))


#--------sklearn's and implemented SVM has very close accuracy-----




