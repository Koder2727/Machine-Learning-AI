"""
This is the implementation of one hidden layer of the neural network from scratch

Kunal Keshav Damame

Lot of inspiration has been taken from coursera course 'neural-networks-deep-learning'"""

import numpy as np

np.random.seed(1)

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = np.array(cache)
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    #loading the cacahe in working variable
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

class Neural_Network():
    
    def __init__(self):
        """
        We will initialize a lot of variables we are gonna use in the program
        
        inputs :
            layer dims : the dimensions of the layers you wanna have in your neural network

        """
    
        #The variables to make a copy of the input training set
        self.__X = None
        self.__Y = None
        self.__learning_rate = None
        
        #The vairables to store the weights and bias
        self.__parameters = {}
        self.__L = None
        self.__cache = []
        self.__AL = None
        self.__grads = {}
        
        #public variable to store the cost evolution for the user to visualize
        self.cost = []
    
        
    def initialize_parameters(self , layer_dims):
        """

        Returns
       
        params -- 
                    Wn -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bn -- bias vector of shape (size of (layer_dims[l]), 1)

        """
        
        # number of layers in the neural network
        L = len(layer_dims)
        
        for l in range(1,L):
            
            #initialize the parameters according ot the layer_dims
            self.__parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            self.__parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        
        return 
    
    def linear_calculator_forward(self , A , W , b):
        """
        Arguments:
            A - The value of activation from previous layer
            W - The weights of this layer
            b - The bias of this layer

        Returns
            Z -- the input of the activation function, also called pre-activation parameter 
            cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

        """
        
        #compute the Z
        Z = np.dot(W,A) + b
        
        #get the cache
        cache = (A,W,b)
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        
        return Z,cache
    
    def linear_activation_forward(self,W , b , activation , A_prev):
        """

        Parameters
        ----------
        W : The weights of this layer
        b : The bias of this layer
        activation : The type of activation fucntion to be applied
        A_prev : Activation of the previous layer

        Returns
        -------
        A : output of the activation function
        cache : tuple containing the linear and activation cache

        """
        
        #activation funciton and cache if the function is sigmoid
        if activation == "Sigmoid" or activation == "sigmoid":
            Z , linear_cache = self.linear_calculator_forward(A_prev ,W,b)
            A , activation_cache = sigmoid(Z)
        
        #activation function and cache if the activation function is relu    
        elif activation == "relu" or activation == "Relu":
            Z , linear_cache = self.linear_calculator_forward(A_prev ,W,b)
            A, activation_cache = relu(Z)
        
        cache = (linear_cache , activation_cache)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        
        return A , cache
    
    def forward_Propogation(self , X , train = True):
        """
        
        Parameters
        ----------
        X : data, numpy array of shape (input size, number of examples)

        Returns
        -------
        AL : last post-activation value
        """
        if train == True:
            #get the number of layers
            L = len(self.__parameters) // 2
            self.__cache = []
            A = X
            #implement the forward propagation with relu function l-1 times
            for l in range(1,L):
                A_prev = A
                A , cache = self.linear_activation_forward(self.__parameters["W" + str(l)] , self.__parameters["b" + str(l)] , "relu" , A_prev)
                self.__cache.append(cache)
            
            #for the final time implement the sigmoid function
            #A_prev = A
            AL , cache = self.linear_activation_forward(self.__parameters["W" + str(L)] , self.__parameters["b" + str(L)] , "Sigmoid" , A)
            self.__cache.append(cache)
            
            assert(AL.shape == (1,X.shape[1]))
            
            return AL
        
        else:
            L = len(self.__parameters) // 2
            cache = []
            A = X
            
            #implement the forward propagation with relu function l-1 times
            for l in range(1,L):
                A_prev = A
                A , cache_ = self.linear_activation_forward(self.__parameters["W" + str(l)] , self.__parameters["b" + str(l)] , "relu" , A_prev)
                cache.append(cache_)
            
            #for the final time implement the sigmoid function
           # A_prev = A
            AL , cache_ = self.linear_activation_forward(self.__parameters["W" + str(L)] , self.__parameters["b" + str(L)] , "Sigmoid" , A)
            cache.append(cache_)
            
            return AL , cache
            
            
    
    def cost_compute(self):
        """
        Returns
            -- calculates the cost 

        """
        #number of examples
        m = self.__Y.shape[1]
        
        #calculation of the cost fucntion
        cost = (-1 / m) * np.sum(np.multiply(self.__Y, np.log(self.__AL)) + np.multiply(1 - self.__Y, np.log(1 - self.__AL)))
        
        #Adjust the shape of the array into 1 d
        cost = np.squeeze(cost)
        
        assert(cost.shape == ())
        
        return cost
    
    def linear_backward(self,dz,cache):
        """
        

        Parameters
        ----------
        dz : Gradient of the cost with respect to the linear output (of current layer l)
        cache :tuple of values (A_prev, W, b)

        Returns
        -------
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b

        """
        A_prev , W , b = cache
        m = A_prev.shape[1]
        
        #calculate the derivatives 
        dW = np.dot(dz,A_prev.T)/m
        db = 1/m*np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dz)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev ,dW , db
    
    def linear_backward_activation(self, dA ,cache , activation):
        """
        

        Parameters
        ----------
        cache : tuple of values (linear_cache, activation_cache)
        activation : the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns
        -------
        dA_prev :  Gradient of the cost with respect to the activation (of the previous layer l-1),
        dW : Gradient of the cost with respect to W (current layer l), same shape as W
        db : Gradient of the cost with respect to b (current layer l), same shape as b

        """
        
        #seprate the linear and activation cache
        linear_cache, activation_cache = cache[0] , cache[1]
       #print(cache)
        #for the relu activation fucntion
        if activation == "relu" or activation == "Relu":
            
            #compute the values da_prev , dw , db from the current vlaue of da and cache
            dz = relu_backward(dA , activation_cache)
            dA_prev , dW , db = self.linear_backward(dz , linear_cache)
            
        elif activation == "sigmoid" or activation == 'Sigmoid':
            
            ##compute the values da_prev , dw , db from the current vlaue of da and cache
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
                   
        return dA_prev , dW , db
    
    def backward_propogation(self):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        updates the values of the grads dictionary

        Returns
        -------
        None.

        """
        #getting the total layers in model
        L = len(self.__cache)
        
        #makeing the shape of al and output vector same
        self.__Y = self.__Y.reshape(self.__AL.shape)
        
        #getting the first derivative of the back propogation
        dAL = - (np.divide(self.__Y, self.__AL) - np.divide(1 - self.__Y, 1 - self.__AL))
    
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = self.__cache[L-1]
        self.__grads["dA" + str(L-1)], self.__grads["dW" + str(L)], self.__grads["db" + str(L)] = self.linear_backward_activation(dAL, current_cache, "sigmoid")
      
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". 
            # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = self.__cache[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_backward_activation(self.__grads["dA" + str(l+1)], current_cache, "relu")
            
            #append the values of the values of the gradients to use in next iterations 
            self.__grads["dA" + str(l)] = dA_prev_temp
            self.__grads["dW" + str(l + 1)] = dW_temp
            self.__grads["db" + str(l + 1)] = db_temp
            
        return  
    
    def get_new_parameters(self):
        """
        Returns
            -->updates the parameters of the w and b

        """
        
        #specifying the learning rate
        learning_rate = self.__learning_rate
        
        #get the number of layers in neural network
        L = len(self.__parameters)//2
        
        #Updating the parameters in the neural network
        for l in range(L):
            self.__parameters["W" + str(l+1)] = self.__parameters["W" + str(l+1)]-learning_rate*self.__grads["dW" + str(l + 1)]
            self.__parameters["b" + str(l+1)] = self.__parameters["b" + str(l+1)]-learning_rate*self.__grads["db" + str(l + 1)]
        
        return 
    
    def Fit(self,X,Y,layers_dim, learning_rate = 0.0075,num_iterations = 3000):
        """
        

        Parameters
        ----------
        X : the input of the feature set for the training of the moodel
            
        Y : The input of the output set for the training of the model
            

        Returns
        -------
         --> trains the model to make it ready for the prediction

        """
        
        #seding the data (You can remove this seed also  , i put it when i test different sizes of models)
        np.random.seed(1)
        
        #Get the parameters into the X and Y
        self.__X = X
        self.__Y = Y
        self.__learning_rate = learning_rate
        
        #Initialize the parameters
        self.initialize_parameters(layers_dim)
        
        #Run the iteration for 1000 times
        for i in range(num_iterations):
            
            #forward propogation
            self.__AL = self.forward_Propogation(X,train = True)
            
            #compute the cost and store it iin a array
            self.cost.append(self.cost_compute())
            
            #Update the derivatives
            self.backward_propogation()
            
            #Update the parameters
            self.get_new_parameters()
            
        
        return
    
    
    def forward_propagation_predict(self , X):
        """
        Argument:
        X -- input data of size (n_x, m
        
        Computes:
        Prediction -- The sigmoid output of the second activation
        
        and reeturns it 
        """
        
        #forward propogation
        probabilities , cache = self.forward_Propogation(X , train = False)
        
        #convert the probailities to predictions
        predictions = (probabilities > 0.5) 
        
        return predictions
    
    
    def Predict(self , X ):
        """

        Parameters
        ----------
        X : The prediction model dataset

        Returns
        -------
        the predicted values using the model
        
        """
        
        A2 = self.forward_propagation_predict(X)
        #predictions = np.round(A2)
        
        return A2
    
    
    
    