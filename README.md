# HOPFIELD NETWORK

## Introduction

This readme provides a brief overview of the hopfield network and its implementation in python using the [MNIST data set](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download).

This documentation contains three main sections:
- In what areas can we use the hopfield network?
- Drawbacks on hopfield data sets.
- Hopfield network implementation on mnist data set.

 ## 1.In what areas can we use the hopfield network?

Hopfield networks are versatile and can be used in pattern recognition for **restoring noisy or incomplete patterns**, like in image and character recognition. When a distorted version of an image is presented, the network retrieves the closest match from stored patterns.

**In content-addressable memory (CAM) systems**, Hopfield networks act as associative memories. This is useful in data retrieval systems where partial or inexact queries need to retrieve whole records. CAM applications include fault-tolerant computing and medical diagnosis systems, where noisy or partial data inputs can still yield correct, full results based on stored patterns.

They are also applied in optimization problems like the **traveling salesman problem** and graph partitioning. By minimizing energy states, Hopfield networks converge to near-optimal solutions, which is helpful for scheduling tasks, resource allocation, and supply chain management. .

## 2. Drawbacks on hopfield Networks?

**Limited Storage Capacity:** Hopfield networks can only store up to about **15%** of the number of neurons as patterns, leading to memory limitations in larger datasets.

**Pattern Overlap and Spurious States:** When patterns are similar, the network may create unwanted "spurious states," which are incorrect or blended memories that reduce accuracy.

**Slow Convergence:** Asynchronous updating and energy minimization can lead to slow convergence, especially for large networks, impacting efficiency in real-time applications.

**Sensitivity to Noise:** Although resilient to some noise, excessive distortion in inputs can prevent the network from accurately retrieving stored patterns.

**Binary Limitations:** Traditional Hopfield networks use binary (-1, 1) states, which limit their ability to represent more complex or continuous data.


## 3. Hop field network implementation on MNIST data set

The [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset consists of 60,000 training images and 10,000 testing images, each of which is a 28x28 pixel image of a handwritten digit from 0 to 9.

The repository contains

hop_field.ipynb: a jupyter notebook that implements a hopfield network for recognizing handwritten digits from the MNIST dataset.

requirements.txt: a list of python packages required to run the code.

### How to Run the code
1. create a data folder in the root directory and download the MNIST dataset from the link provided above.([Link](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download))
2. Install required packages
3. run the hop_field.ipynb file

### code explanation

The general guide line can be seen in 6 steps
Step 1: Hopfield Network Implementation
Step 2: Preprocess Data
Step 3: Train Hopfield Network
Step 4: Process Data with Hopfield Network
Step 5: Train Random Forest Classifier
Step 6: Generate Predictions

step 1: load the MNIST dataset
```python
# import data
allTrain = pd.read_csv( 'data/train.csv' )
n = len( allTrain )
X_train = allTrain.drop( 'label', axis=1).to_numpy()
Y_train = allTrain[ 'label' ].to_numpy()
del allTrain
test = pd.read_csv( 'data/test.csv' )
X_test = test.to_numpy()
print( "data import complete" )
```
step 2: 
This step defines two classes, HopfieldNetwork and HopfieldForest, for implementing pattern recognition using Hopfield networks. The HopfieldNetwork class initializes with a set of patterns and can use either the Hebbian or pseudo-inverse learning rule to compute the weight matrix. The processBatch method asynchronously updates the input patterns over a specified number of iterations to converge to a stable state. The HopfieldForest class extends the expressiveness of the Hopfield network by using multiple networks, each trained on different combinations of patterns. This allows the system to handle more complex pattern recognition tasks by aggregating the results from multiple Hopfield networks.

```python
class HopfieldNetwork( object ):
    def __init__( self, pattern, rule='pseudo-inverse' ):
        '''expects patterns to have values belonging to {-1, 1}
           initializes with Hebbian rule'''
        self.n = pattern[0].size
        self.order = np.arange( self.n )
        
        if rule == 'hebbian':
            self.w = np.tensordot( pattern, pattern, axes=( ( 0 ), ( 0 ) ) ) / len( pattern )
            self.w[ self.order, self.order ] = 0.0
        elif rule == 'pseudo-inverse':
            c = np.tensordot( pattern, pattern, axes=( ( 1 ), ( 1 ) ) ) / len( pattern )
            cinv = np.linalg.inv( c )
            self.w = np.zeros( ( self.n, self.n ) )
            for k, l in product( range( len( pattern ) ), range( len( pattern ) ) ):
                self.w = self.w + cinv[ k, l ] * pattern[ k ] * pattern[ l ].reshape( ( -1, 1 ) )
            self.w = self.w / len( pattern )
        else:
            assert false, 'invalid learning rule: {}\nplease choose hebbian or pseudo-inverse'.format( rule )

    def processBatch(self, x, iters=4):
        '''input should be same size format as patterns. Implements asynchronous update'''
        h = np.array(x, dtype=float)

        # Ensure h has the correct shape matching self.n
        if h.shape[1] > self.n:
            h = h[:, :self.n]  # Truncate h if it has an extra column
        elif h.shape[1] < self.n:
            raise ValueError(f"Input dimension {h.shape[1]} does not match expected {self.n}")

        for _ in range(iters):
            np.random.shuffle(self.order)
            for i in self.order:
                # Adjust h.T[:, :self.n] to ensure shapes align
                h[:, i] = np.dot(self.w[i], h.T[:self.n])
                h[:, i] = np.where(h[:, i] < 0, -1.0, 1.0)
        return h
class HopfieldForest( object ):
    '''Because expressiveness is so limited
       use multiple nets for different sets of patterns'''
    
    def __init__( self, pattern, perSet=2 ):
        '''expects patterns to have values belonging to {-1, 1}
           initializes with Hebbian rule'''
        self.net = [ HopfieldNetwork( p ) for p in combinations( pattern, perSet ) ]
    
    def processBatch( self, x, iters=4 ):
        '''input should be same size format as patterns. Implements asynchronous update
        returns the results from all models stacked together'''
        return np.array( [ n.processBatch( x ) for n in self.net ] )
```


step 3:
This code segment normalizes the training data, computes average patterns for each class (if supervised), converts the patterns to bitwise format using an iterative threshold adjustment process, and visualizes the balance metric and threshold values. The resulting bitwise patterns are used to train the Hopfield network.
```python
# create patterns
supervised = True
factor = 2.0 / np.max( X_train )
avg_classes = lambda: np.array( [ factor * np.mean( X_train[ Y_train == l, : ], axis=0 ) - 1.0
                                  for l in np.unique( Y_train ) ] )
patterns = avg_classes() if supervised else ( factor * X_train - 1 )

# convert to bitwise
low = -1.0
high = 1.0
iters = 8
metric = np.zeros( iters )
threshold = np.zeros( iters )
for i in tqdm( range( iters ) ):
    threshold[ i ] = ( low + high ) / 2
    metric[ i ] = np.mean( np.where( patterns < threshold[ i ], -1, 1 ) )
    if metric[ i ] > -0.7: # 0.7 chosen by inspection(not mine, credit to kaggle)
        low = threshold[ i ]
    else:
        high = threshold[ i ]
bitPatterns = np.where( patterns < threshold[ -1 ], -1, 1 )
XB_train = np.where( ( factor * X_train - 1 ) < threshold[ -1 ], -1, 1 ) if supervised else bitPatterns
XB_test = np.where( ( factor * X_test - 1 ) < threshold[ -1 ], -1, 1 )

plt.subplot( 211 )
plt.title( 'Balance' )
plt.plot( metric )
plt.subplot( 212 )
plt.title( 'Threshold' )
plt.plot( threshold )
pass
```
step 4:
This part of the code is responsible for training a Random Forest classifier on the preprocessed data and generating predictions for the test set.
```python
# create and train model
model = RandomForestClassifier()
model.fit( xt, yt )
print( "Training Accuracy: ", model.score( xt, yt ) )
print( "Validation Accuracy: ", model.score( xv, yv ) )

# produce a prediction
pred = model.predict( XH_test )
sub = pd.DataFrame( pred, columns=[ 'Label' ] )
sub.index.names = [ 'ImageId' ]
sub.index += 1
sub.to_csv( 'submission.csv' )
```
