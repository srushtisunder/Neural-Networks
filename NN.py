import time
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plot

class NeuralNetwork:
    def __init(arrayOfNeurons =[],outputFunction=[],  ):
        self.numberOfLayers=len(arrayOfNeurons)
        self.arrayOfNeurons=arrayOfNeurons
        self.weightsArray=weightsArray
        self.biasArray=biasArray
        self.__constructWeightsAndBiases()

        self.X=T.matrix()
        self.Y=T.matrix()
        self.outputFunction=outputFunction

        self.cost
        self.params=[]
        self.updates=





        ##nitialize variables of a NeuralNetwork
        ##Then

    def __constructWeightsAndBiases():

        for i in range(self.numberOfLayers-1):
            self.weightsArray[i]=init_weights(arrayOfNeurons[i],arrayOfNeurons[i+1])
            self.biasArray[i]=init_bias(arrayOfNeurons[i+1]
        #TODO outputfunctions

    def __createTrainFunction(arg):

        for i in range(self.numberOfLayers-1):
            self.__update(i)

        #train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

    def __update(i):
        self.weights[i]=self.weights[i]-
