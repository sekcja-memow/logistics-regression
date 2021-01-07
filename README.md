# Basics of AI - implementations of ML models and optimization methods

In this project we will focus on the implementation of the logistic regression model as well as optimization and regularization methods/algorithms using `python` and `numpy library`.
Then we will test and present examples of using our implementation on practical examples.

At the end, we will compare the results of our model with the ready implementation from the `sklearn` library.

<img width="600" src="https://www.equiskill.com/wp-content/uploads/2018/07/WhatsApp-Image-2020-02-11-at-8.30.11-PM.jpeg"/>


### Introduction
**Logistic regression**, despite its name, is a linear model for classification rather than regression.
Logistic regression is also often called *logit regression* because it uses a logistic function to predict the results.
LR is a probabilistic model that assigns the probability of a sample to a given class.
You can find more about linear regression [here](https://en.wikipedia.org/wiki/Logistic_regression).


### Interface
Our models provide a clean interface based on the `sklearn` library interface.
In addition to the models themselves, various regularizers and optimization algorithms are available.

<img width="600" src="./docs/images/interface.png"/>

You can find more details in [docs](./docs/docs.md).

#### Basic Example
```python
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.evaluate(y_test, y_pred)
```
You can find more examples in [examples section :fire:](./src).


### Credits
* *images:* https://www.equiskill.com/