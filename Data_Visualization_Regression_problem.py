
'''
    Tutorial website: https://www.springboard.com/blog/data-mining-python-tutorial/
'''

import pandas # module to clean/restructure data 
import matplotlib.pyplot  # fundamental package for data visualization
import numpy # functions to work with arrays
import scipy # a collection of tools  for statistics (regression and analysis functions)
import seaborn # function for graphing regression lines as well as distribution plots for each variable

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')


df = pandas.read_csv(r'C:\Users\silvaart\Desktop\workspace\pessoal\Programacao\AI-playing-Pong-master\Data Visualization - Data Mining\kc_house_data.csv')
df.head()


df.isnull().any() # checking to see if any of our data has null values. If there were any, we'd drop or filter the null values


df.dtypes 
'''
 checking out the data types for each of our variables. 
 We want to get a sense of whether or not data is numerical (int64, float64) or not (object)
'''


'''
    I imported the data frame from the csv file using Pandas, and the first thing I did was make sure it reads properly. 
    I also used the “isnull()” function to make sure that none of my data is unusable for regression. 
    In real life, a single column may have data in the form of integers, strings, or NaN, all in one place – 
    meaning that you need to check to make sure the types are matching and are suitable for regression. 
    This dataset happens to have been very rigorously prepared, something you won’t see often in your own database. 
'''



'''
 Simple exploratory analysis and regression results.
'''
df.describe()
'''
 Quick takeaways: We are working with a dataset that contains 21,613 observations, mean price is approximately $540k, 
 median price is approximately $450k, and the average house’s area is 2080 ft2
'''


figure = pyplot.figure(figsize=(20, 6))
sqft = figure.add_subplot(121)
cost = figure.add_subplot(122)

sqft.hist(df.sqft_living, bins=80)
sqft.set_xlabel('Ft^2')
sqft.set_title('Histogram of House Square Footage')

cost.hist(df.price, bins=80)
cost.set_xlabel("Price ($)")
cost.set_title('Histogram of House Prices')



'''
    Now that we have a good sense of our data set and know the distributions of the variables we are trying to measure, 
    let’s do some regression analysis. 
    First we import statsmodels to get the least squares regression estimator function. 
    The “Ordinary Least Squares” (OLS) module will be doing the bulk of the work when it comes to crunching numbers for regression 
    in Python.
'''
import statsmodels.api
from statsmodels.formula.api import ols

'''
When you code to produce a linear regression summary with OLS with only two variables this will be the formula that you use:

An example of simple linear regression model summary output
'''
m = ols('price ~ sqft_living', df).fit()
print(m.summary())


'''
Having the regression summary output is important for checking the accuracy of the regression model and 
data to be used for estimation and prediction – but visualizing the regression is an important step to take to communicate
the results of the regression in a more digestible format.
'''
seaborn.jointplot(x="sqft_living", y="price", data=df, kind="reg", fit_reg=True, size=7)
pyplot.show()

