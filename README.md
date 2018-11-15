
# Fitting a Logistic Regression Model - Lab

## Introduction
In the last lecture, you were given a broad overview of logistic regression. This included two seperate packages for creating logistic regression models. We'll first investigate building logistic regression models with 

## Objectives

You will be able to:
* Understand and implement logistic regression


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

The `dmatrices()` method above mirrors the R languages syntax. The first parameter is a string representing the conceptual formula for our model. Afterwards, we pass the dataframe where the data is stored, as well as an optional parameter for the formate in which we would like the data returned. The general pattern for defining the formula string is: `y_feature_name ~ x_feature1_name + x_feature2_name + ... + x_featuren_name`. You should also notice that two of the x features, Race and Sex, are wrapped in `C()`. This indicates that these variables are *categorical* and that dummy variables need to be created in order to convert them to numerical quantities. Finally, note that y itself returns a Pandas DataFrame with two columns as y itself was originally a categorical variable. With that, it's time to try and define a logistic regression model on your own! 

## Your Turn - Step 1: Import the Data

Import the data stored in the file **titanic**.


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

For our first foray into logistic regression, we are going to attempt to build a model that classifies whether an indivdual survived the Titanic shiwrech or not (yes its a bit morbid). Follow the programming patterns described above to define X and y.


```python
df.Survived.value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
#Your code here
from patsy import dmatrices

y, X = dmatrices('Survived ~ Pclass  + C(Sex) + Age + SibSp +  Fare + C(Embarked)',
                  df, return_type = "dataframe")
#Notes: PassengerId is simply a numerical ordering. No valuable information encoded here
#Similarly, Name is not useful for predictions. 
#P-values below should indicate such if students do include in initial formulation.
```

## Step 3: Fit the model

Now with everything in place, initialize a regression object and fit your model!

### Warning: If you receive an error of the form "LinAlgError: Singular matrix"
Stats models was unable to fit the model due to some Linear Algebra problems. Specifically, the matrix was not invertable due to not being full rank. In layman's terms, there was a lot of redundant superfulous data. Try removing some features from the model and running it again.


```python
# Your code here
import statsmodels.api as sm
logit_model = sm.Logit(y, X)
result = logit_model.fit()
```

    Optimization terminated successfully.
             Current function value: 0.444229
             Iterations 6


## Step 4: Analyzing results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
#Your code here
result.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Survived</td>     <th>  No. Observations:  </th>  <td>   712</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   704</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     7</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Thu, 15 Nov 2018</td> <th>  Pseudo R-squ.:     </th>  <td>0.3417</td>  
</tr>
<tr>
  <th>Time:</th>              <td>16:12:57</td>     <th>  Log-Likelihood:    </th> <td> -316.29</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -480.45</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>5.360e-67</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>    5.6378</td> <td>    0.633</td> <td>    8.901</td> <td> 0.000</td> <td>    4.396</td> <td>    6.879</td>
</tr>
<tr>
  <th>C(Sex)[T.male]</th>   <td>   -2.6168</td> <td>    0.217</td> <td>  -12.040</td> <td> 0.000</td> <td>   -3.043</td> <td>   -2.191</td>
</tr>
<tr>
  <th>C(Embarked)[T.Q]</th> <td>   -0.8155</td> <td>    0.598</td> <td>   -1.363</td> <td> 0.173</td> <td>   -1.988</td> <td>    0.357</td>
</tr>
<tr>
  <th>C(Embarked)[T.S]</th> <td>   -0.4036</td> <td>    0.270</td> <td>   -1.494</td> <td> 0.135</td> <td>   -0.933</td> <td>    0.126</td>
</tr>
<tr>
  <th>Pclass</th>           <td>   -1.2102</td> <td>    0.163</td> <td>   -7.427</td> <td> 0.000</td> <td>   -1.530</td> <td>   -0.891</td>
</tr>
<tr>
  <th>Age</th>              <td>   -0.0433</td> <td>    0.008</td> <td>   -5.263</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.027</td>
</tr>
<tr>
  <th>SibSp</th>            <td>   -0.3796</td> <td>    0.125</td> <td>   -3.043</td> <td> 0.002</td> <td>   -0.624</td> <td>   -0.135</td>
</tr>
<tr>
  <th>Fare</th>             <td>    0.0012</td> <td>    0.002</td> <td>    0.474</td> <td> 0.635</td> <td>   -0.004</td> <td>    0.006</td>
</tr>
</table>



# Your analysis here
Based on our P-values, most of the current features appear to be significant based on a .05 significance level. That said, the 'Embarked'and 'Fare' features were not significant based on their higher p-values.

## Level - up

Create a new model, this time only using those features you determined were influential based on your analysis in step 4.


```python
#your code here
y, X = dmatrices('Survived ~ Pclass  + C(Sex) + Age + SibSp ',
                  df, return_type = "dataframe")

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
  <th>Date:</th>          <td>Thu, 15 Nov 2018</td> <th>  Pseudo R-squ.:     </th>  <td>0.3399</td>  
</tr>
<tr>
  <th>Time:</th>              <td>16:16:27</td>     <th>  Log-Likelihood:    </th> <td> -318.36</td> 
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
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td>    5.6008</td> <td>    0.543</td> <td>   10.306</td> <td> 0.000</td> <td>    4.536</td> <td>    6.666</td>
</tr>
<tr>
  <th>C(Sex)[T.male]</th> <td>   -2.6235</td> <td>    0.215</td> <td>  -12.229</td> <td> 0.000</td> <td>   -3.044</td> <td>   -2.203</td>
</tr>
<tr>
  <th>Pclass</th>         <td>   -1.3174</td> <td>    0.141</td> <td>   -9.350</td> <td> 0.000</td> <td>   -1.594</td> <td>   -1.041</td>
</tr>
<tr>
  <th>Age</th>            <td>   -0.0444</td> <td>    0.008</td> <td>   -5.442</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.028</td>
</tr>
<tr>
  <th>SibSp</th>          <td>   -0.3761</td> <td>    0.121</td> <td>   -3.106</td> <td> 0.002</td> <td>   -0.613</td> <td>   -0.139</td>
</tr>
</table>



# Comments:

Note how removing the insignificant features had little impact on the r^2 value of our model.

## Summary 

Well done. In this lab we practiced using stats models to build a logistic regression model. We then reviewed interpreting the results, building upon our previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Sci-kit learn!
