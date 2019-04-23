
# Fitting a Logistic Regression Model - Lab

## Introduction
In the last lecture, you were given a broad overview of logistic regression. This included two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with statsmodels.



## Objectives

You will be able to:
* Implement logistic regression with statsmodels
* Interpret the statistical results associated with regression model parameters


## Review

The stats model example we covered had four essential parts:
* Importing the data
* Defining X and y
* Fitting the model
* Analyzing model results

The corresponding code to these four steps was:

```
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

#Step 1: Importing the data
salaries = pd.read_csv("salaries_final.csv", index_col = 0)

#Step 2: Defining X and y
y, X = dmatrices('Target ~ Age  + C(Race) + C(Sex)',
                  salaries, return_type = "dataframe")

#Step 3: Fitting the model
logit_model = sm.Logit(y.iloc[:,1], X)
result = logit_model.fit()

#Step 4: Analyzing model results
result.summary()
```

Most of this should be fairly familiar to you; importing data with Pandas, initializing a regression object, and calling the fit method of that object. However, step 2 warrants a slightly more in depth explanation.

The `dmatrices()` method above mirrors the R languages syntax. The first parameter is a string representing the conceptual formula for our model. Afterwards, we pass the DataFrame where the data is stored, as well as an optional parameter for the formate in which we would like the data returned. The general pattern for defining the formula string is: `y_feature_name ~ x_feature1_name + x_feature2_name + ... + x_featuren_name`. You should also notice that two of the x features, Race and Sex, are wrapped in `C()`. This indicates that these variables are categorical, meaning that dummy variables need to be created in order to convert them to numerical quantities. Finally, note that y itself returns a pandas DataFrame with two columns as y itself was originally a categorical variable. With that, it's time to try and define a logistic regression model on your own!

## Your Turn - Step 1: Import the Data

Import the data stored in the file **titanic.csv**.


```python
#Your code here
```

## Step 2: Define X and Y

For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the Titanic shipwreck or not (yes it's a bit morbid). Follow the programming patterns described above to define X and y.


```python
#Your code here
```

## Step 3: Fit the model

Now with everything in place, initialize a regression object and fit your model!

### Warning: If you receive an error of the form "LinAlgError: Singular matrix"

Stats models was unable to fit the model due to some Linear Algebra problems. Specifically, the matrix was not invertible due to not being full rank. In layman's terms, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Your code here
```

## Step 4: Analyzing results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
#Your code here
```

## Your analysis here

## Level - up

Create a new model, this time only using those features you determined were influential based on your analysis in step 4.


```python
#your code here
```

## Summary 

Well done! In this lab, you practiced using stats models to build a logistic regression model. You then reviewed interpreting the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Sci-kit learn!
