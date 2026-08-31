import math
import numpy as np
import random

class MLP:

  def __init__(self , training_features , training_labels , parameters):
    self.parameters = parameters
    self.training_features = training_features
    self.training_labels = training_labels
    self.sliding_head = 0
    self.learning_rate = self.parameters['LEARNING_RATE']
    self.num_layers = len(self.parameters['CODE_OF_ACTIVATION_FUNCTIONS'])
    self.layers =  np.empty(self.num_layers,dtype=Layer)
    self.build_layers()
    
    

  def build_layers(self):
    for i in range(self.num_layers):
      self.layers[i] = Layer(i , self)

  def train(self):
    return self.predict_row(self.current_feature_row())

  def predict_row(self, X):
    self.reset_caches()
    return self.layers[self.num_layers - 1].calculateLayerOutput(X)

  def current_feature_row(self):
    return self.training_features[self.sliding_head]

  def current_label_row(self):
    return self.training_labels[self.sliding_head]

  def backpropagate(self):
    for layer in self.layers[:0:-1]:
      layer.update_weights( False)
    for layer in self.layers[:0:-1]:
      layer.update_weights( True)
    self.reset_caches()
    

  
  def train(self, epoch=1):
    
    for i in range(epoch):
      # printProgressBar(i, epoch, prefix = 'Progress:', suffix = 'Complete', length = 50)
      self.sliding_head =0
      for j in range(self.training_features.shape[0]):
        self.backpropagate()
        self.sliding_head +=1
      # printProgressBar(i + 1, epoch, prefix = 'Progress:', suffix = 'Complete', length = 50)

  def reset_caches(self):
    for i in range(self.num_layers):
      self.layers[i].resetOutput()
      for j in range(self.layers[i].num_neurons):
        self.layers[i].neurons[j].resetDelta()

  def clearAll(self):
      for i in range(self.num_layers):
        self.layers[i].resetOutput()
        for j in range(self.layers[i].num_neurons):
          self.layers[i].neurons[j].resetDelta()
          for k in range(self.layers[i].neurons[j].num_inputs):
            self.layers[i].neurons[j].input_branches[k].reset_weight()
            print(self.layers[i].neurons[j].input_branches[k].w)
          

      

  

    

#--------------------------------------------------------------------------

class Layer:

  def __init__(self,layer_index , MLP):
    self.MLP = MLP
    self.layer_index = layer_index
    self.num_neurons = self.setNumberOfPerceptrons()
    self.activation = ActivityFunction(self)
    self.neurons =  np.empty(self.num_neurons,dtype=Perceptron)
    self.neurons = self.build_neurons()
    self.output = np.full((self.num_neurons), math.inf)

  def resetOutput(self):
    self.output = np.full((self.num_neurons), math.inf)

  def setNumberOfPerceptrons(self):
    if(self.layer_index == 0 ):
      return  self.MLP.training_features.shape[1]
    if(self.layer_index == self.MLP.num_layers - 1 ):
      return  self.MLP.training_labels.shape[1]
    return self.MLP.parameters['NUMBER_OF_PERCEPTRONS_FOR_HIDDEN_LAYERS'][self.layer_index-1]

  def build_neurons(self):
    neurons =  np.empty(self.num_neurons,dtype=Perceptron) 
    for i in range( self.num_neurons ):
      neurons[i] = Perceptron( i , self)
    return neurons

  def getPreviousLayer(self):
    return self.layer_index != 0 and self.MLP.layers[self.layer_index - 1 ] or -1

  
  def getNextLayer(self):
    return self.layer_index != self.MLP.num_layers - 1 \
     and self.MLP.layers[self.layer_index + 1 ] or -1

  def calculateLayerOutput(self,X):     # receive Vector   # return Vector

    if ((any(self.output==math.inf))==False):
      return self.output
    if(self.layer_index==0):
      X = X
      return X
    else:
      X = self.getPreviousLayer().calculateLayerOutput(X)
    output =  np.empty(self.num_neurons)
    for i in range(self.num_neurons):
      if(self.layer_index == 0 ):
        output[i] = X[i]
      else:
        output[i] = self.neurons[i].forward(X)
    self.output = output
    return self.output

  def derivative(self,net):
    return self.activation.calculateDerivative(net)

  def update_weights(self, hardUpdate = False):
    for perceptron in self.neurons:
      perceptron.update_weights(hardUpdate)
    

  

#--------------------------------------------------------------------------

class ActivityFunction:
  
  def __init__(self,layer):
    self.layer = layer
    self.functionType = self.layer.MLP.parameters['CODE_OF_ACTIVATION_FUNCTIONS'][self.layer.layer_index]
  
  def apply(self,x):
    if (self.functionType == 1) :
      return self.sigmoid(x)
    if (self.functionType == 2) :
      return self.tanh(x)
    if (self.functionType == 3) :
      return self.ReLU(x)
    if (self.functionType == 4) :
      return self.linear(x)

  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def tanh(self , x):
    t=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return t

  def ReLU(self ,x):
    return max(0.0,x)

  def linear(self , x):
    return x

  def calculateDerivative(self , net):
    if (self.functionType == 1) :
      sig = self.sigmoid(net)
      return (1-sig)*sig
    if (self.functionType == 2) :
      return 1 - self.tanh(net)**2
    if (self.functionType == 3) :
      if(net<0):
        return 0
      return 1
    if (self.functionType == 4) :
      return 1

#--------------------------------------------------------------------------

class Perceptron:

  

  def __init__(self , perceptron_index , layer ):   # [layer_index  ,  perceptron] 
    self.bias = 0  # 0 or 1
    self.perceptron_index = perceptron_index
    self.layer = layer
    self.num_inputs  =  self.getNumberOfInputs()
    self.input_branches =  np.empty(self.num_inputs,dtype=Layer)
    self.build_inputs()
    self.delta = math.inf

  def resetDelta(self):
    self.delta = math.inf

  def build_inputs(self):
    for i in range(self.num_inputs):
      self.input_branches[i] = InputBranch(self , i)

  def getNumberOfInputs(self):
    if(self.layer.layer_index == 0 ):
      return  1
    return self.layer.getPreviousLayer().num_neurons + self.bias + 1 

  def forward(self , X):
        net = self.net_output(X)
        return self.layer.activation.apply(net)

        
  def net_output(self , X):    # X is input feature vector
        y=0
        # DONT FORGET BAIAS
        X = np.concatenate((X, [1]), axis=None) 
        for i in range(self.num_inputs):
          y += self.input_branches[i].branch_output(X[i])
        return y

  def getDelta(self):
    if(self.delta != math.inf):
      return self.delta
    desiredOutput=0
    if(self.layer.layer_index==self.layer.MLP.num_layers - 1):
      desiredOutput = self.layer.MLP.current_label_row()[self.perceptron_index]
    X = (self.layer.layer_index == 0)  and self.layer.MLP.current_feature_row() or self.layer.getPreviousLayer().calculateLayerOutput(self.layer.MLP.current_feature_row())
    self.delta =  self.calculateDelta(X ,desiredOutput)
    return self.delta
    # print(self.layer.layer_index, self.perceptron_index,self.delta)

  def calculateDelta(self,X , desiredOutput):  # X is input vector 
    net = self.net_output(X)
    if(self.layer.layer_index == self.layer.MLP.num_layers - 1):     # perceptron in output layer
      return self.layer.derivative(net) * ( desiredOutput - self.forward(X))
    else:       # perceptron in hidden layer
      sigma = 0
      # layerOutput = self.layer.calculateLayerOutput(self.layer.MLP.current_feature_row())
      for perceptron in self.layer.getNextLayer().neurons:
        sigma += (perceptron.input_branches[self.perceptron_index].w * perceptron.getDelta()) 
      return self.layer.derivative(net) * sigma

  
  def update_weights(self,hardUpdate = False):
    for inputBranch in self.input_branches:
      hardUpdate and inputBranch.apply_w_new() or inputBranch.updatew_new()

    

  
#--------------------------------------------------------------------------

class InputBranch:
  
  def __init__(self , perceptron, inputNumber):
    self.inputNumber = inputNumber
    self.perceptron = perceptron
    self.reset_weight()
    self.w_new = self.w

  def reset_weight(self):
    if(self.perceptron.layer.layer_index == 0):
      self.w =  1
    self.w = random.uniform(0,1) 

  def branch_output(self , x):
    return self.w * x 

  

  def updatew_new(self):
    learning_rate = self.perceptron.layer.MLP.learning_rate
    yi = np.concatenate((self.perceptron.layer.getPreviousLayer().calculateLayerOutput(self.perceptron.layer.MLP.current_feature_row()), [self.perceptron.bias]), axis=None)[self.inputNumber]
    self.w_new =self.w +  learning_rate * self.perceptron.getDelta() * yi 

  def apply_w_new(self):
    self.w = self.w_new
