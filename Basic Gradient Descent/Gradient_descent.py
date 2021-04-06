#predict the prices of houses in boston
#import the important libraries
import numpy as np
from matplotlib import pyplot as plt

#initialize the data to be used
mean=np.array([5.0,6.0])
cov=np.array([[1.0,0.95],[0.95,1.2]])
data=np.random.multivariate_normal(mean,cov,8000)

#visuallize the data
plt.scatter(data[:500,0],data[:500,1],marker='.')
plt.title("The data points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#split the data into test and train
data = np.hstack((np.ones((data.shape[0], 1)), data))
split_factor = 0.90
split = int(split_factor * data.shape[0]) 
  
X_train = data[:split, :-1] 
y_train = data[:split, -1].reshape((-1, 1)) 
X_test = data[split:, :-1] 
y_test = data[split:, -1].reshape((-1, 1))

#code for linear regression using gradient dissent
#function to compute hypothesis

def hypothesis(X,theta):
    return(np.dot(X,theta))

#function to calculate the gradient of error function
def gradient(X,y,theta):
    h=hypothesis(X,theta)
    grad=np.dot(X.transpose(),(h-y)) #basically squaring the function
    return (grad)

#function to calculate error for current value of theta
def cost(X,y,theta):
    h=hypothesis(X,theta)
    J=np.dot((h-y).transpose(),(h-y))
    J/=2
    return (J[0])

#function to crete a list containing mini batches
def create_mini_batches(X,y,batch_size):
    mini_batches=[]
    data=np.hstack((X,y))#rejoin the data
    np.random.shuffle(data)
    n_minibatches=data.shape[0]//batch_size
        
    for i in range (n_minibatches+1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini=mini_batch[:, :-1]                    #put every thing from minibatch to X_mini all element sexcept last
        y_mini=mini_batch[:, -1].reshape((-1,1))     #put everything from minibatch to y_minin only last elemny
        mini_batches.append((X_mini,y_mini))          #add this partiton to a seprate array
    if(data.shape[0]%batch_size!=0):                #if some part is leftover
        mini_batch=data[i*batch_size:data.shape[0]]  #divide the remaining array into a smaller group 
        X_mini=mini_batch[:, :-1] 
        y_mini=mini_batch[:, -1].reshape((-1,1))
        mini_batches.append((X_mini,y_mini))
    return (mini_batches)

#function to perform the gradient descent
def gradientDescent(X,y,learning_rate=0.001,batch_size=32):
    theta=np.zeros((X.shape[1],1))          #why the shape is 3 and whay it is zero
    error_list=[]
    max_iters=3                             #for iterations in total set
    for itr in range (max_iters):
        mini_batches=create_mini_batches(X,y,batch_size)
        for mini_batch in mini_batches:
            X_mini,y_mini=mini_batch
            theta=theta-learning_rate*gradient(X_mini,y_mini,theta)
            error_list.append(cost(X_mini,y_mini,theta))
    return (theta,error_list)


#calling the gradient descent funtion
theta,error_list=gradientDescent(X_train,y_train)
print('Bias =',theta[0])
print('Coefficients =',theta[1:])

#visualising the gradient descent
plt.plot(error_list)
plt.title("The slope as function of iterations")
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

#predecting the output for X_test
y_pred=hypothesis(X_test,theta)
plt.scatter(X_test[:,1],y_test[:,],marker='.')
plt.plot(X_test[:,1],y_pred,color='orange')
plt.title("Final Best fit line")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

