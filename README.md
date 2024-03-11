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