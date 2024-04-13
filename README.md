# Machine Learning on AWS

## Machine Learning Concepts
### What and where does it stand
<details>
    <summary><i>AI vs ML vs DL</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/AI vs ML vs DL.png" />
</details>

### ML Terminologies
    - Features
    - Labels
    - Instances
<details>
    <summary><i>Example</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/ML Terminologies.png" />
</details>

Here ML pridication can be made as with so and so features, on what price house is likely to be sold

### Machine Learning Phases
    > Data Collection 
        > Featurization 
            > Model Training - on a subset of orig. dataset 
                > Model Testing - on a different subset of orig. dataset. Retrain to acheive accuracy.
                    > Base Model

### Types of ML 
    - Supervised Learning: Creates pridictive ML models using feauted and lebals - both.
        - Classification: When Labals can be expressed types of classes
        - Regression: When tags/labels are continuous. eg House Labels
    - Unsupervised Learning: Groups & discover patterns only on features.
        - Used for clustering & anamoly detection

### ML algorithms
<details>
    <summary><i>Supervised learning</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Supervised learning algorithms.png" />
</details>

<details>
    <summary><i>Unsupervised learning</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Unsupervised learning algorithms.png" />
</details>

### Linear Regression
<details>
    <summary><i>Finds best fitting line </i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Linear Regression.png" />
</details>

### Logistic Regression
<details>
    <summary><i>Uses a fn to push values towards a boundary</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Logistic Regression.png" />
</details>

### Dicision Trees
<details>
    <summary><i>Leaf node is final decision</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Dicision Tree.png" />
</details>

### Naive Bayes
<details>
    <summary><i>A collection of classification algo</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/Naive Bayes.png" />
</details>

### SVM - Support Vector Machine
<details>
    <summary><i>Lines are classifiers</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/SVM.png" />
</details>

### KNN - K Nearest Naighbours
<details>
    <summary><i>Applied on splits of dataset</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/KNN.png" />
</details>

### Random Forest
<details>
    <summary><i>Collection of decision trees</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/KNN.png" />
</details>

## Data & ML - An Intro

### Deep Learning real world application
- Image objects identification
- Photo caption generation
- Facebook's Deep Face: A face recognition AI
- For blinds: Audio generation from camara etc
- Real time language translation from photo
- Real tme language translation of over call/chat etc
- Smart Reply
- Automatic Ratinal Disease Assesment
- Self Driving cars
- Games

### Quick checks?
- Which keyboard shortcut in Jupyter can you use to run the current cell and then select the cell below? :- Shift + Enter
- What is the name of the popular Python distribution platform used by many data scientists? :- Anaconda
- Which of the following deep learning applications is most often associated with reinforcement learning? :- Video Game AI development
- Which python libraries provides a high-level API to run machine learning experiments using deep learning? :- Keras

## Machine Learning

### Week and Strong AI
- Week AI: Highly specilized in solving specific probelm
- Strong AI: Solves general problem

### ML Notions 
- y
    - Historical targets
    - The label
    - Ex - Daily sale of the shop when 20% discount is was given
- x
    - A Single feature
    - Ex - Location of the shop
- X
    - Multiple features
    - Ex - % discounts on product, # of hours hours open etc
- D 
    - Training Dataset
    - Represented as: All the features we have/know and historrical observations
    - I.e. D = (X, y)
- ŷ
    - y-hat
    - a guess of y
- f
    - The relationship b/w all the x we have and the historical data we have
    - I.e. ŷ = f(x)

> [ RECAP ]   
**Superwised Learning:** When lebel or y or historical outcomes is given. 
**Unsuperwised Learning**: When lebel or y or historical outcomes is not given.

### Supervised Learning problems
- Regression 
    - y ∈ set of real numbers
    - Ex: Age, Rating, temperature
- Classification
    - y ∈ {0, 1}
    - Ex: Old or young, good or bad, Summer or winter

## Distributed Machine Learning
- Motive
    - Large Dataset
    - Speed
    - Complexity
    - Accuracy
- Apache Spark
    - A lightning fast, cluster computing used to provide capability of distributed computing
    - Built on Scala
    -  <details>
            <summary>Spark stack</summary>
            <img src="resources/images/SparkStack.png">
        </details>
    - <details>
            <summary>Spark master-slave architecture</summary>
            <img src="resources/images/SparkMasterSlave.png">
        </details>
    - Abstractions: RDD (2011) > Dataframe (2013) >  Dataset (2015)
    
- Apache Spark MLib
    - Spark's scalable ML library
    - <details>
            <summary>MLib Agorithms</summary>
            <img src="resources/images/MLibAlgo.png">
        </details>
- Decision Trees
    - A ML model, in form of tree structure
    - Types
        - Classification Tree
            - Person in rich, Will it rain tomorrow etc
        - Regression Tree
            - Someone's income, Stock Price etc

## AWS ML Services [TODO]
### S3
- Backbone of AWS ML services, like SageMaker
- Used of creating Data Lake
- 5 TB is max support
- 11 nines durability, 99.999999999%
- 

## Deep Learning

### Types of Neural Network
- Feedforward Neural Network
    - A stack of neural network, where input is fed into bottom and prediction is generated.
- Convolutional Neural Network (CNN)
    - Used in inage classification
    - Ex, Is there a question mark in image?
- Recurrent Neural Network (RNN)
    - Deals with sequences
    - Ex, Predict stock price, translation etc
    - LTSM and GRU are difrent flavours of RNN

### CNN
#### What are they used for?
- When data is not neatly align with columns
    - Language Translation
    - Sentence Classification, question or request
    - Sentiment Analysis
- CNN can be used for finding a feature when its not aon a specific sopt
    - A 'STOP' sign in a picture
    - A word in a sentence

#### How does CNN work?
- Convolution is way of saying - I am going to breakup data in to little chunks and process those chunks individdually.
- CNN working is inspired by biology of how human identify things.
- How does CNN identify a 'STOP' sign?
    - Input image is chunked out and passed on to a group of neural network 'to identify lines'.
    - Passed on on another neural network, to idenify alignments of those lines
    - Then, to find out colors
    - So on, and at the end combined up to get
        - There is octagonal shape
        - Having more red in it
        - Written letter 'S' 'T' 'O' 'P' in it in same sequence.
    - Here is the result!

#### CNN with TensorFlow/Keras
- Source Data: Wedth  x Height X Color channels 
- Conv2D layer: Does actual convolutining on a 2D data.
    - Conv1D and conv3D is also availale - Need not to be image
- MaxPooling2D: Can be used for reducing a 2D layer down. To help performance capability
- Flatten Layer: 2D to 1D

> [Typical Usages]   
 Conv2D > MaxPooling2D > Flatten > Dense > Dropout > SoftMax

 #### Specilized CNN Architecture
 - Define specific arrangements of layers, padding, hyperparameters
 - LeNet-5: Good for handwriting recognition
 - AlexNet: Image Classification, deeper and compute intense than LeNet-5
 - GoogLeNet: Even Deepar, but better performance. Introduced  'Inception Module' (Group of convolutional Layers)
 - ResNet (Residual Network): Even deepar, maintains 'skip connections' 

 ### RNN

 #### What are they used for?
 - Time series data. 
    - Pridict future based on past data
    - Web Log, Sensor Log, Stock Trade
 - Data that consist of sequence of arbitrary length
    - Machine Translation
    - Image Caption
    - Machine Generated Music

#### How does RNN work?  
<details>
    <summary><i>expand...</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/RNNWorking.png" />
</details>

<details>
    <summary><i>Horizontal Scaling</i></summary><br>
    <img align="center" alt="AI vs ML vs DL" src="resources/images/RNNWorking_HorizontalScaling.png" />
</details>

#### RNN Topologies
- Sequence to Sequence
    - Predict stock price based on historical data
- Sequence to Vector
    - Words in sentance to sentiment
- Vector to Sequence
    - generate caption from image
- Encoder Decoder
    - Sequnce -> Vector -> Sequence
    - Eg, Machine Translation

### Tranfer Learning
    - It can very cumbersome and costly to build a model from scratch and train every time, as modal can be too big.
    - 'Hugging Face' is a pre-train model repository. We can use modal as we want and can be fine tune as well.
    - These pre-train model are considered as transfer learning.
    - 'Hugging Face' is integrated with AWS SageMaker

### Tuning Neural Netwrok
-  Learning Rate
    - Neural Network are trained by 'Gradiant Decendent' or similar means
    - Start with some random point & sample diffrent solution; seeking to minimize some cost funtion over many epochs
    - <details>
            <summary><i>More on LR</i></summary><br>
            <img align="center" alt="AI vs ML vs DL" src="resources/images/LearningRate.png" />
    </details>
- Batch Size
    - Number of training sample are using in each batch of each epoch.
    - [IMP] Smaller the batch size = Better

- Imp. points to remember
    - Smaller batch tends NOT to get stuck in local minima.
    - Larger batch size can converge on wrong solution at random
    - Large LR can overshot the correct solution
    - Small LR increase training time


### Regulization technique
    - Dropout
    - Early Stopping
    - TODO - Watch training video

### L1 & L2 regulization
    - TODO - Watch video

### Confusion Matrix
    - TODO - Watch video and notice Recall, Precision etc

## AWS SageMaker
- Seq2Seq
    - Input is sequence of token and output is sequence of token
    - Implemmted with CNN And RNN.
    - Example: Speach to Text, Text Summarization etc
    - BLUE Score and Preplexity is best suited for ML Translation.
- DeepAR
    - Forcasting 1-Dimensional time series data
    - Example - Stock Price
- BlazeingText 
    - Predict labal of a sentance; mind it - not entire document
    - Input
        - '\__label__\' is important in input. 
        -  __label__4  linux ready for prime time , intel says , despite all the linux hype , the open-source movement has yet to make a huge splash in the desktop market . that may be about to change , thanks to chipmaking giant intel corp .
        - __label__2  bowled by the slower one again , kolkata , november 14 the past caught up with sourav ganguly as the indian skippers return to international cricket was short lived . 
- Object2Vec
    - A general purpose version of Word2Vec
    - Compute nearest neighbour of objects
    - Used to identifying similar items, user etc 
    - <details>
            <summary><i>Usages</i></summary><br>
            <img align="center" alt="Object2Vec Usage" src="resources/images/Object2Vec_usages.png" /> </details>
     - <details>
            <summary><i>Training Input</i></summary><br>
            <img align="center" alt="Object2Vec TrainingInput" src="resources/images/Object2Vec_usages.png" />
    </details>
- Object Detection
    - <details>
            <summary><i>Usages</i></summary><br>
            <img align="center" alt="ObjectDetect_WhatFor" src="resources/images/ObjectDetect_WhatFor.png" /> </details>
     - <details>
            <summary><i>Training Input</i></summary><br>
            <img align="center" alt="Training Input" src="resources/images/ObjectDetect_TrainingInput.png" />
    </details>

- Image Classification
    - Used for naming object in an image, doesn't tell position etc.

- [IMP] Semantic Segmentation in SageMaker
    - <details>
            <summary><i>Usage</i></summary><br>
            <img align="center" alt="Training Input" src="resources/images/SemSeg_WhatFor.png" />
    </details>

- Random cut forest: developed at Amazon
    - Used for Anamoly detection.
    - <details>
            <summary><i>Usages</i></summary><br>
            <img align="center" alt="Usages" src="resources/images/RandomCutForest.png" />
    </details>

    - Usages decision tree, under the hood.
    - How anamonly get detected - Let's say to accomodate new data set, which is aanamoly, it required to add new set of branches in decision tree. That is an indication of anamoly 

- PCA in SageMaker
    - PCA (Principle Component Analysis) is a diamentionality reduction technique.
    - so, if avoids crush of diamentionality
    - A Higher dimentional data -> PCA -> 2D, without loosing much information
    - Reduced dimensions are called componnets
    - How its used:
        - Covariance matrix is created
        - then, SVD (Singular Value Decomposition) algo to distil that down
        - Has two modes
            - Regular: For sparse data and moderate number of observation and feature
            - Randomized: For large # of overvation and feature, usaes complex algorithms.
- Factorization Machines 
    - Used for predicting a classification with a sparse test data.
    - Example - Predicting which product from a huge product catelog will have sale etc

- IP Insights
    - Unupervised learning of IP address usage pattern
    - Identifies suspicious behaviour from IP address, like login attemps, account creation etc
    - Used as a security tool

- Reinforcement Learning
    - Example of Pac-man, where a right action is rewarded and wrong action gets penalized.
    - Q-Learnig
        - A Specific implementation of reinforcement learning
        - We have:
            - A Set of env. states - s
            - A set of possible actions in those states - a
            - A value of each state/action - Q
        - Start with Q = 0
        - Increment Q, when reward received on state/action
        - Decrement Q, when bad things happe on state/action
        - <details>
            <summary><i>Exploration Problem</i></summary><br>
            <img align="center" alt="Usages" src="resources/images/ExplorationProblem.png" />
        </details>
- [IMP]Automatic Model Tuning
    - How do you know best value of Learning Rate, batch size, pth etc?
    - By experimenting? Possible then we have handful of hyperparameter. But, what if there are many different hyerparameters.
    - We can't try every combination of every possible value somehow, train a model, and evaluate every time.
    - SageMaker can automatically tune model
        - Define hyperparameter you care about, ranges you care about and metricsyou are optimizing of
        - SageMaker spins up a 'HyperParameter Tuning Job', that trains many combinations you allowed
        - the set of hyperparameters producing best results, can then deployed as model
        - So, it can learn as it goes, so it doesn't have to try very possible combinations.
    - Taking more time - Limit search space.
- SageMaker Debugger
    - Saves internal state at periodic level.
    - Have reporting capabilities.
- Autopilot / AutoML
    - Automates:
        - Algorithm selection
        - Data Processing
        - Model tuning
        - All Infrastructure
    - Does all the hit and trial
- Which SageMaker algorithm would be best suited for identifying topics in text documents in an unsupervised setting?
    - (LDA) Latent Dirichlet Allocation is a topic modeling technique. 
    - (NTM) Neural Topic Model would also be a correct answer.
- 