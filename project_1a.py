import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import timeit

def init_bias(n = 1):
    return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return (theano.shared(value=W_values, name='W', borrow=True))

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

# update parameters
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


decay = 1e-9
learning_rate = 0.01
epochs = 1000
batch_size = 32
neuronArray=[5,10,15,20,25]
numberOfNeuron=10
timeTakenArray=[]
batchArray=[4,8,16,32,64]

# theano expressions
X = T.matrix() #features
Y = T.matrix() #output


#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
trainX = scale(trainX, trainX_min, trainX_max)

train_Y[train_Y == 7] = 6
trainY = np.zeros((train_Y.shape[0], 6))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1


#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = scale(testX, testX_min, testX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)


#for numberOfNeuron in neuronArray:

w1, b1 = init_weights(36, numberOfNeuron), init_bias(numberOfNeuron) #weights and biases from input to hidden layer
w2, b2 = init_weights(numberOfNeuron, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer

h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
py = T.nnet.softmax(T.dot(h1, w2) + b2)

y_x = T.argmax(py, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
params = [w1, b1, w2, b2]
updates = sgd(cost, params, learning_rate)

# compile
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)



# first, experiment with a small sample of data
##trainX = trainX[:1000]
##trainY = trainY[:1000]
##testX = testX[-250:]
##testY = testY[-250:]


# train and test
n = len(trainX)
test_accuracy = []
train_cost = []
#for batch_size in batchArray:
start = timeit.default_timer()

for i in range(epochs):
    if i % 1000 == 0:
        print(i)

    trainX, trainY = shuffle_data(trainX, trainY)
    cost = 0.0
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
        cost += train(trainX[start:end], trainY[start:end])
    train_cost = np.append(train_cost, cost/(n // batch_size))

    test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

end= timeit.default_timer()
timeTakenArray.append((end-start)/batch_size)


print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))


#Plots
plt.figure()
plt.plot(range(epochs), train_cost)
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost | bs=32 | N=10 | decay=1e-9')
plt.savefig('p1a_q4a_sample_cost_decay=1e-9.png')

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy | bs=32 | N=10 | decay=1e-9')
plt.savefig('p1a_q4b_sample_accuracy_decay=1e-9.png')
'''
#for P1A Q2b
plt.figure()
plt.plot(batchArray, timeTakenArray)
plt.xlabel('batchArray')
plt.ylabel('timeTakenArray')
plt.title('parameter update time')
plt.savefig('p1a_parameter_update_time.png')

#for P1a Q3b_updatetime_vs_numberofneurons
plt.figure()
plt.plot(neuronArray, timeTakenArray)
plt.xlabel('Number Of Neurons')
plt.ylabel('Time Taken')
plt.title('parameter update time')
plt.savefig('p1a_q3b_parameter_update_time.png')
'''
#plt.show()
