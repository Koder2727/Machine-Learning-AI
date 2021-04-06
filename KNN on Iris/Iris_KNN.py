#Import the iris dataset from sklearn
#we will use the KNN clasifier and the tree classifier 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#initialize the dataset object
iris=datasets.load_iris()

#get the data into X and final result into Y
X=iris.data
y=iris.target

#split the train and test data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.5)

#initialize the classifier 
my_classifier=KNeighborsClassifier()

#train the classifier
my_classifier.fit(X_train,y_train)

#get the predictions of the classifier
predictions=my_classifier.predict(X_test)

#get the accuracy score for the dataset
print(accuracy_score(predictions,y_test))