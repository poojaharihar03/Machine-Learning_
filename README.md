# Theory
• Precision- Ability of a classification model to identify only the relevant data points.
It is defined as the number of true positives divided by the number true positives + the number of false positives.

• Recall- Ability of a model to find all the relevant cases within a dataset.
The precise definition of recall is the number of true positives divided by the number of true positives plus the number of false negatives.

•Accuracy-It is the number of correct predictions made by the model divided by the total number of predictions.

• F1-Score
In cases where we want to find an optimal blend of precision and recall we can combine the two metrics using what is called the F1 score.

![image](https://user-images.githubusercontent.com/88924201/232049072-27fcdcd1-44d8-48c3-98f1-008ea38f0567.png)


# Supervised learning

Basically supervised learning is when we teach or train the machine using data that is well labelled. Which means some data is already tagged with the correct answer. After that, the machine is provided with a new set of examples(data) so that the supervised learning algorithm analyses the training data(set of training examples) and produces a correct outcome from labelled data.

Types:-
Regression
Logistic Regression
Classification
Naive Bayes Classifiers
K-NN (k nearest neighbors)
Decision Trees
Support Vector Machine

Advantages:-
*Supervised learning allows collecting data and produces data output from previous experiences.
*Helps to optimize performance criteria with the help of experience.
*Supervised machine learning helps to solve various types of real-world computation problems.
*It performs classification and regression tasks.
*It allows estimating or mapping the result to a new sample. 
*We have complete control over choosing the number of classes we want in the training data.

Disadvantages:-
*Classifying big data can be challenging.
*Training for supervised learning needs a lot of computation time. So, it requires a lot of time.
*Supervised learning cannot handle all complex tasks in Machine Learning.
*Computation time is vast for supervised learning.
*It requires a labelled data set.
*It requires a training process.
 
 
 # Unsupervised Learning
 
 Unsupervised learning is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. 
 
The two categories of algorithms: 
1)Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.
Types of Unsupervised Learning:-
Clustering
Exclusive (partitioning)
Agglomerative
Overlapping
Probabilistic

Clustering Types:-
Hierarchical clustering
K-means clustering
Principal Component Analysis
Singular Value Decomposition
Independent Component Analysis

Advantages of unsupervised learning:
*It does not require a training data to be labelled.
*Dimensionality reduction can be easily accomplished using unsupervised learning. 
*Capable of finding previously unknown patterns in data.

Disadvantages of unsupervised learning :
*Difficult to measure accuracy or effectiveness due to lack of predefined answers during training. 
*The results often have lesser accuracy.
*The user needs to spend time interpreting and label the classes which follow that classification.

![image](https://user-images.githubusercontent.com/88924201/232050193-282c9324-ba79-4488-844d-a8c0c49354c7.png)


![image](https://user-images.githubusercontent.com/88924201/232050825-1b5f73d7-99b0-40ba-833f-8ba57ce3147d.png)


The presence or absence of labeling in your data is often used to identify a machine learning task.

ML task
# Supervised tasks
A task is supervised if you are using labeled data. We use the term labeled to refer to data that already contains the solutions, called labels.
For example, predicting the number of snow cones sold based on the average temperature outside is an example of supervised learning.

Snow Cone
In the preceding graph, the data contains both a temperature and the number of snow cones sold. Both components are used to generate the linear regression shown on the graph. Our goal was to predict the number of snow cones sold, and we feed that value into the model. We are providing the model with labeled data and therefore, we are performing a supervised machine learning task.

# Unsupervised tasks
A task is considered to be unsupervised if you are using unlabeled data. This means you don't need to provide the model with any kind of label or solution while the model is being trained.
Let's take a look at unlabeled data.

Tree-Unsupervised learning involves using data that doesn't have a label. One common task is called clustering. Clustering helps to determine if there are any naturally occurring groupings in the data.

Let's look at an example of how clustering works in unlabeled data.

Example: Identifying book micro-genres with unsupervised learning

Imagine that you work for a company that recommends books to readers.\
The assumption is that you are fairly confident that micro-genres exist, and that there is one called Teen Vampire Romance. However, you don’t know which micro-genres exist specifically, so you can't use supervised learning techniques.

This is where the unsupervised learning clustering technique might be able to detect some groupings in the data. The words and phrases used in a book's description might provide some guidance on its micro-genre.


# Quality of data-
The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:

# Outliers
Missing or incomplete values
Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html#sphx-glr-auto-examples-applications-plot-outlier-detection-wine-py


# test and train
Splitting the dataset gives you two sets of data:
Training dataset: The data on which the model will be trained. Most of your data will be here. Many developers estimate about 80%.
Test dataset: The data withheld from the model during training, which is used to test how well your model will generalize to new data.


# The model training algorithm iteratively updates a model's parameters to minimize some loss function.

Let's define those two terms:
**Model parameters: Model parameters are settings or configurations that the training algorithm can update to change how the model behaves. Depending on the context, you’ll also hear other specific terms used to describe model parameters such as weights and biases. Weights, which are values that change as the model learns, are more specific to neural networks.
**Loss function**: A loss function is used to codify the model’s distance from a goal. For example, if you were trying to predict the number of snow cone sales based on the day’s weather, you would care about making predictions that are as accurate as possible. So you might define a loss function to be “the average distance between your model’s predicted number of snow cone sales and the correct number.” You can see in the snow cone example; this is the difference between the two purple dots.



# Linear models

One of the most common models covered in introductory coursework, linear models simply describe the relationship between a set of input numbers and a set of output numbers through a linear function (think of y = mx + b or a line on a x vs y chart). Classification tasks often use a strongly related logistic model, which adds an additional transformation mapping the output of the linear function to the range [0, 1], interpreted as “probability of being in the target class.” Linear models are fast to train and give you a great baseline against which to compare more complex models. A lot of media buzz is given to more complex models, but for most new problems, consider starting with a simple model.

# Tree-based models

Tree-based models are probably the second most common model type covered in introductory coursework. They learn to categorize or regress by building an extremely large structure of nested if/else blocks, splitting the world into different regions at each if/else block. Training determines exactly where these splits happen and what value is assigned at each leaf region. For example, if you’re trying to determine if a light sensor is in sunlight or shadow, you might train tree of depth 1 with the final learned configuration being something like if (sensor_value > 0.698), then return 1; else return 0;. The tree-based model XGBoost is commonly used as an off-the-shelf implementation for this kind of model and includes enhancements beyond what is discussed here. Try tree-based models to quickly get a baseline before moving on to more complex models.

# Deep learning models

Extremely popular and powerful, deep learning is a modern approach that is based around a conceptual model of how the human brain functions. The model (also called a neural network) is composed of collections of neurons (very simple computational units) connected together by weights (mathematical representations of how much information thst is allowed to flow from one neuron to the next). The process of training involves finding values for each weight. Various neural network structures have been determined for modeling different kinds of problems or processing different kinds of data.

A short (but not complete!) list of noteworthy examples includes:
**FFNN**: The most straightforward way of structuring a neural network, the Feed Forward Neural Network (FFNN) structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer.
**CNN**: Convolutional Neural Networks (CNN) represent nested filters over grid-organized data. They are by far the most commonly used type of model when processing images.
**RNN/LSTM**: Recurrent Neural Networks (RNN) and the related Long Short-Term Memory (LSTM) model types are structured to effectively represent for loops in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.
**Transformer**: A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data.

# Machine learning using Python libraries
For more classical models (linear, tree-based) as well as a set of common ML-related tools, take a look at scikit-learn. The web documentation for this library is also organized for those getting familiar with space and can be a great place to get familiar with some extremely useful tools and techniques.
For deep learning, mxnet, tensorflow, and pytorch are the three most common libraries. For the purposes of the majority of machine learning needs, each of these is feature-paired and equivalent.

**Hyperparameters** are settings on the model that are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.
A **loss function** is used to codify the model’s distance from this goal.
**Model parameters**are settings or configurations the training algorithm can update to change how the model behaves.
