
# Fitting a Logistic Regression Model - Lab

## Introduction
You were previously given a broad overview of logistic regression. This included two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with statsmodels.



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
import statsmodels.api as sm

#Step 1: Importing the data
salaries = pd.read_csv("salaries_final.csv", index_col = 0)

#Step 2: Defining X and y
x_feats = ["Race", "Sex", "Age"]
X = pd.get_dummies(salaries[x_feats], drop_first=True, dtype=float)
y = pd.get_dummies(salaries["Target"], dtype=float)

#Step 3: Fitting the model
X = sm.add_constant(X)
logit_model = sm.Logit(y.iloc[:,1], X)
result = logit_model.fit()

#Step 4: Analyzing model results
result.summary()
```

Most of this should be fairly familiar to you; importing data with Pandas, initializing a regression object, and calling the fit method of that object. However, step 2 warrants a slightly more in depth explanation.

Recall that we fit the salary data using `Race`, `Sex`, and `Age`. Since `Race` and `Sex` are categorical, we converted them to dummy variables using the `get_dummies()` method. The ```get_dummies()``` method will only convert `object` and `category` data types to dummy variables so it is safe to pass `Age`. Note that we also passed two additional arguments, ```drop_first=True``` and ```dtype=float```. The ```drop_first=True``` argument removes the first level for each categorical variable and the ```dtype=float``` argument converts the data type of all of the dummy variables to float. The data must be float in order to obtain accurate statistical results from statsmodel. Finally, note that y itself returns a pandas DataFrame with two columns as y itself was originally a categorical variable. With that, it's time to try and define a logistic regression model on your own!

## Your Turn - Step 1: Import the Data

Import the data stored in the file **titanic.csv**.


```python
#Your code here
import pandas as pd

df = pd.read_csv('titanic.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Step 2: Define X and Y

For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the Titanic shipwreck or not (yes it's a bit morbid). Follow the programming patterns described above to define X and y.


```python
df.Survived.value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
# Your code here
x_feats = ["Pclass", "Age", "SibSp", "Fare", "Sex", "Embarked"]
X = pd.get_dummies(df[x_feats], drop_first=True, dtype=float)
y = df["Survived"].astype(float)

# Have to dropna in order to fit the model
X = X.dropna()
y = y[y.index.isin(X.index)]
```

## Step 3: Fit the model

Now with everything in place, initialize a regression object and fit your model!

### Warning: If you receive an error of the form "LinAlgError: Singular matrix"

Stats models was unable to fit the model due to some Linear Algebra problems. Specifically, the matrix was not invertible due to not being full rank. In layman's terms, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Your code here
import statsmodels.api as sm
X = sm.tools.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()
```

    Optimization terminated successfully.
             Current function value: 0.443267
             Iterations 6


## Step 4: Analyzing results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
# Your code here
result.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Survived</td>     <th>  No. Observations:  </th>  <td>   714</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   706</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     7</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Thu, 01 Aug 2019</td> <th>  Pseudo R-squ.:     </th>  <td>0.3437</td>  
</tr>
<tr>
  <th>Time:</th>              <td>16:12:26</td>     <th>  Log-Likelihood:    </th> <td> -316.49</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -482.26</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>1.103e-67</td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    5.6503</td> <td>    0.633</td> <td>    8.921</td> <td> 0.000</td> <td>    4.409</td> <td>    6.892</td>
</tr>
<tr>
  <th>Pclass</th>     <td>   -1.2118</td> <td>    0.163</td> <td>   -7.433</td> <td> 0.000</td> <td>   -1.531</td> <td>   -0.892</td>
</tr>
<tr>
  <th>Age</th>        <td>   -0.0431</td> <td>    0.008</td> <td>   -5.250</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.027</td>
</tr>
<tr>
  <th>SibSp</th>      <td>   -0.3806</td> <td>    0.125</td> <td>   -3.048</td> <td> 0.002</td> <td>   -0.625</td> <td>   -0.136</td>
</tr>
<tr>
  <th>Fare</th>       <td>    0.0012</td> <td>    0.002</td> <td>    0.474</td> <td> 0.636</td> <td>   -0.004</td> <td>    0.006</td>
</tr>
<tr>
  <th>Sex_male</th>   <td>   -2.6236</td> <td>    0.217</td> <td>  -12.081</td> <td> 0.000</td> <td>   -3.049</td> <td>   -2.198</td>
</tr>
<tr>
  <th>Embarked_Q</th> <td>   -0.8260</td> <td>    0.598</td> <td>   -1.381</td> <td> 0.167</td> <td>   -1.999</td> <td>    0.347</td>
</tr>
<tr>
  <th>Embarked_S</th> <td>   -0.4130</td> <td>    0.269</td> <td>   -1.533</td> <td> 0.125</td> <td>   -0.941</td> <td>    0.115</td>
</tr>
</table>



## Your analysis here


```python
# Based on our P-values, most of the current features appear to be significant based on a .05 significance level. 
# That said, the 'Embarked' and 'Fare' features were not significant based on their higher p-values.

```

## Level - up

Create a new model, this time only using those features you determined were influential based on your analysis in step 4.


```python
# Your code here
x_feats = ["Pclass", "Age", "SibSp", "Sex"]
X = pd.get_dummies(df[x_feats], drop_first=True, dtype=float)
y = df["Survived"].astype(float)

X = X.dropna()
y = y[y.index.isin(X.index)]

X = sm.tools.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()

result.summary()
```

    Optimization terminated successfully.
             Current function value: 0.445882
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Survived</td>     <th>  No. Observations:  </th>  <td>   714</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   709</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Thu, 01 Aug 2019</td> <th>  Pseudo R-squ.:     </th>  <td>0.3399</td>  
</tr>
<tr>
  <th>Time:</th>              <td>16:12:30</td>     <th>  Log-Likelihood:    </th> <td> -318.36</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -482.26</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>1.089e-69</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>    <td>    5.6008</td> <td>    0.543</td> <td>   10.306</td> <td> 0.000</td> <td>    4.536</td> <td>    6.666</td>
</tr>
<tr>
  <th>Pclass</th>   <td>   -1.3174</td> <td>    0.141</td> <td>   -9.350</td> <td> 0.000</td> <td>   -1.594</td> <td>   -1.041</td>
</tr>
<tr>
  <th>Age</th>      <td>   -0.0444</td> <td>    0.008</td> <td>   -5.442</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.028</td>
</tr>
<tr>
  <th>SibSp</th>    <td>   -0.3761</td> <td>    0.121</td> <td>   -3.106</td> <td> 0.002</td> <td>   -0.613</td> <td>   -0.139</td>
</tr>
<tr>
  <th>Sex_male</th> <td>   -2.6235</td> <td>    0.215</td> <td>  -12.229</td> <td> 0.000</td> <td>   -3.044</td> <td>   -2.203</td>
</tr>
</table>




```python

# Note how removing the insignificant features had little impact on the $R^2$ value 
# of our model.
```

## Summary 

Well done! In this lab, you practiced using stats models to build a logistic regression model. You then reviewed interpreting the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Sci-kit learn!
