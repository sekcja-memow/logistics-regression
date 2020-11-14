# Basics of AI - implementations of ML models

In this project, we focused on implementing two machine learning algorithms for classification - 
Logistic Regression and SVM - using `python` and `numpy` library.

### Introduction
**Logistic regression**, despite its name, is a linear model for classification rather than regression.
Logistic regression is also often called *logit regression* because it uses a logistic function to predict the results.
LR is a propabilistic model that assigns the probability of a sample to a given class.
You can find more about linear regression [here](https://en.wikipedia.org/wiki/Logistic_regression).


**SVM** - Support Vector Machine - it is an algorithm capable of both classification and regression. However, we will focus on its ability to classify samples into the appropriate class.
The concept of SVM is to create a hyperplane that will separate the two classes with a maximum margin.

<img width="600" src="https://www.researchgate.net/publication/304611323/figure/fig8/AS:668377215406089@1536364954428/Classification-of-data-by-support-vector-machine-SVM.png" />


### Interface
Our models provide a clean interface based on the `sklearn` library interface.
In addition to the models themselves, various regularizers and optimization algorithms are available.

<img width="800" src="./docs/images/structure.png"/>

You can find more info in [docs](./docs/docs.md).

#### Basic Example
```python
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.evaluate(y_test, y_pred)
```
You can find more examples in [examples section :fire:](./src/examples.ipynb).