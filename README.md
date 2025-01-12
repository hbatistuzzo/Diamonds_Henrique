# Diamonds_Henrique

![GitHub top language](https://img.shields.io/github/languages/top/hbatistuzzo/Diamonds_Henrique)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hbatistuzzo/Diamonds_Henrique)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/hbatistuzzo/Diamonds_Henrique)
![GitHub last commit](https://img.shields.io/github/last-commit/hbatistuzzo/Diamonds_Henrique)

## Project objective

<img src="images/diamonds.jpg" align="right" width="40%" style="margin-left: 20px;"/>

<div style="text-align: justify;">
This project is based on a [somewhat classic kaggle dataset from 2016](https://www.kaggle.com/datasets/shivam2503/diamonds) used to explain introductory level machine learning.  

Given a [historic dataset](/data/diamonds.csv) with over 54,000 diamond prices and their characteristics, we are tasked by our client (Rick Harrison from _Pawn Stars_) to estimate the price of [his own list](/data/rick_diamonds.csv) of 5,000 diamonds, thus setting up a classic regression problem. Specifically, the goals are:
</div>

<br>

- to infer which characteristics are more likely to influence a diamond's price;
- to progressively train and test a regression model until its accuracy meet a certain standard (defined by the RMSE). Rick’s goal is to obtain an average error below 900 dollars.

<p align="center"><img src="images/challenge_objectives.png" alt="full"  width="60%"></p>

This ReadMe is divided into 2 main sections:
1) the first focusing on the theory behind Linear Regression and Machine Learning,
2) and the second dealing with the project itself.

<br>

# $\color{goldenrod}{\textrm{1 - Machine Learning and Linear Regression Theory}}$

As with other showcase projects, I add my personal notes on how I tackle the subject. Far from didactic, they're simply here to show that I actually did the legwork.

In regression analysis, groups of variables can be correlated to a single target, or outcome. The relationships between these variables can be used to predict the future values of that outcome. We call the variables that are correlated with the outcome `independent` X variables, and the outcome variable the `dependent` Y variable. 

Regression analysis is one of the most common techniques used to make predictions. Depending on the question we would like to answer, and the format of the outcome variable, regression analysis can be used to both make value predictions (what will my income be next year?) and classifications (based on the qualities of a song, will I like it or not?). The relationship between the X variables and the Y variables can also take different formats. The case that an increase or decrease in an X variable always produces the same, fixed increase or decrease in the Y variable is called linear regression. When this relationship is not always the same we classify it as non-linear regression. 

In some cases, there is only one predictor variable, which makes the relation a simple (univariate) linear regression. In other cases, there are more than one predictor variables which is called multiple (multivariate) linear regression, as is the case in the project below.

- Univariate analysis, or simple linear regression, is when only one X (independent) variable is used to predict the outcome variable. In the case of linear univariate analysis, we can model this relationship using a straight line.

- Multivariate analysis is when multiple variables all work together to explain the outcome variable. For instance, using the data above, let’s say we still want to predict the number of minutes a person is awake during the night. We think that this outcome could be determined by multiple factors in addition to the number of times a person wakes up during the night: the minutes of sleep they get overall, and their daytime activity level (maybe people who are more active are likely to sleep more).

[This ipynb](linear-regression-part1.ipynb) uses [this data file](/data/advertising.csv) to exemplify the use of seaborn/matplotlib tolls such as regplot and pairplot below, to visualize and study regression cases.

<div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="images/regplot_TV.png" alt="alt text" width="33%" style="display: block; margin: auto;" />
    <img src="images/regplot_news.png" alt="alt text" width="33%" style="display: block; margin: auto;" />
    <img src="images/regplot_radio.png" alt="alt text" width="33%" style="display: block; margin: auto;" />
</div>

<br>

<p align="center"><img src="images/pairplot_test.png" alt="alt text" width="75%" style="display: block; margin: auto;" /></p>

<br>

Tools such as these make it easier to provide inferences regarding the nature of the relationships between the different variables in the dataset. There is clearly a higher correlation between TV adverts and sales, than with the Newspaper and Radio counterparts.

In quantitative terms, we could use pandas.corr to output Pearson correlation coefficients and reach this conclution:

|          | TV        | Radio     | Newspaper | Sales     |
|----------|-----------|-----------|-----------|-----------|
| **TV**   | 1.000000  | 0.054809  | 0.056648  | 0.901208  |
| **Radio**| 0.054809  | 1.000000  | 0.354104  | 0.349631  |
| **Newspaper** | 0.056648  | 0.354104  | 1.000000  | 0.157960  |
| **Sales**| 0.901208  | 0.349631  | 0.157960  | 1.000000  |

Heatmaps are useful in these scenarios to make the most critical relationships pop out:

```
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='Greens', vmin=.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
```

<p align="center"><img src="images/sales_heatmap.png" alt="alt text" width="75%" style="display: block; margin: auto;" /></p>

<br>

This is all swell and good, but since there is a clear linear relationship between the variables, we could try implementing a predictive analysis on this dataset. This is where the `scikit-learn` library comes in handy. 

- We separate our variables:
    - X: predictive variables - or explicative variables (should be a **pandas dataframe or n-D numpy array**)
    - y: the variable you want to predict - or target (should be a **pandas series or 1-D numpy array**)

In this case, say we want to predict `y = sales`  given the value I invest in `X = TV` advertising. This is an example of **supervised learning** / **supervised machine learning**, since the model is trained on labeled data, where the input (features) and output (target) are known.

- `model.fit(X, y)` is the most important step in our linear regression. It will train our model. Specifically, it will calculate the values of the **intercept** and the **coefficients**.

- After training our model, we can use the method `model.predict(X)` to obtain a predicted value of `Sales` given a value of `TV`. Say we want to know the value our model predicts for `TV = 100`. We have to pass a dataframe like the one we've used to `train` our model. We can also predict several values at once, or even the whole dataset. The process is described in the code below:


```
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = df[['TV']]
y = df['Sales']

model.fit(X,y)

data_to_predict = pd.DataFrame([100], columns=['TV'])
model.predict(data_to_predict)

####
multiple_data_to_predict = pd.DataFrame([100, 150, 200, 250, 300, 350, 400, 450], columns=['TV'])
model.predict(multiple_data_to_predict)
####
qw 
X_all = X
y_predicted_all = model.predict(X_all)
```

We can then visualize the results with matplotlib, for example
```
plt.figure(figsize=(8,6))
plt.xlabel('TV')
plt.ylabel('Sales')
plt.scatter(X, y, color='red', label='observed')

# plot the predicted values together with the observed values
plt.scatter(X_all, y_predicted_all, label='predicted')


plt.legend();
```

<p align="center"><img src="images/predicted_add.png" alt="alt text" width="75%" style="display: block; margin: auto;" /></p>

<br>

This works quite well in this scenario because the relationship is clearly linear. We can now use either the `scikit-learn` or the `scipy` library to explicitly provides us with measures of the slope, intercept, r_value, value and std_err of the linear fit:

```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

print ('The slope is: ' , str(model.coef_[0]))
print ('The intercept is: ' , str(model.intercept_))
```

This yields a slope of 0.055464770469558874' and an intercept of 6.974821488229891 i.e. <b> if I change 1 unit of investment in TV, i gain ~0.055 units of Sales </b>; also, <b> the value of sales for which TV equals 0 is ~6.97 </b>

```
# with scipy:
from scipy import stats
X = df['TV']
y = df['Sales']
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y_pred)

print('The slope is: ' + str(slope))
print('The intercept is: '+ str(intercept))
print('The r_value is: ' + str(r_value))
print('The p_value is: '+ str(p_value))
print('The std_err is: ' + str(std_err))
```

>__Warning__ Scipy takes 2 dataframes, instead of an 1-D array in the target as sk-learn does, so mind this detail.

This yields _very slightly_ different values (beyond the 10th decimal) due to the internal workings of the libraries themselves. Scipy, beingn a strong stats tool, also provides us automatically with some more relevant info:

```
The r_value is: 0.9012079133023304
The p_value is: 7.92791162532341e-74
The std_err is: 0.001895551178040245
```

The r_value is the Pearson coefficient coefficient which we derived earlier from `pandas.corr`, which measures the strength and direction of the linear relationship between X (tv adds) and y (sales). An r-value of 0.9012 indicates a strong positive linear relationship between tv adds and sales. This means higher spending on TV ads is generally associated with higher sales.

The p-value tests the null hypothesis that there is no correlation between X and y. A very small p-value (close to 0) indicates that the observed correlation is highly unlikely to have occurred by chance. With a p-value of 7.93×10^(−74), the evidence strongly rejects the null hypothesis. The correlation between TV ads and sales is statistically significant.

std_err is the standard error of the slope, which measures the precision of the slope estimate. It quantifies how much the slope would vary if you repeatedly sampled from the population and performed the regression each time. A smaller standard error indicates that the slope estimate is more precise.
With a value of 0.0019, the slope estimate of b1 = 0.0555 is very precise, which suggests the relationship between TV ads and sales is robust.

So the verdict is in, specifically in this linear regression predictive scenario, `scipy.stats.linregress` is a more handy tool than sk-learn, which really shines on other sorts of ML cases. The __biggest caveat__ to consider is that scipy doesn't handle performance well with big datasets, so that's something to keep in mind.

<br>
<br>

![Abhinandan Trilokia](https://raw.githubusercontent.com/Trilokia/Trilokia/379277808c61ef204768a61bbc5d25bc7798ccf1/bottom_header.svg)

# $\color{goldenrod}{\textrm{2 - The Project}}$

## Technologies

- Python 3.8.3
	- Pandas 1.4.4
	- Numpy 1.20.3
	- Pycaret 2.3.10
	- Seaborn 0.11.2
	- Matplotlib 3.5.3
	- Scikit-learn 1.1

---

## Dataset Description and Inspection:

The list of diamonds contains the following information:

- carat (0.2-5.01): The carat is the diamond’s physical weight measured in metric carats. One carat equals 0.20 gram and is subdivided into 100 points.
- cut (Fair, Good, Very Good, Premium, Ideal): The quality of the cut. The more precise the diamond is cut, the more captivating the diamond is to the eye thus of high grade.
- color (from J (worst) to D (best)): The colour of gem-quality diamonds occurs in many hues. In the range from colourless to light yellow or light brown. Colourless diamonds are the rarest. Other natural colours (blue, red, pink for example) are known as "fancy,” and their colour grading is different than from white colorless diamonds.
- clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)): Diamonds can have internal characteristics known as inclusions or external characteristics known as blemishes. Diamonds without inclusions or blemishes are rare; however, most characteristics can only be seen with magnification.
- depth (43-79): It is the total depth percentage which equals to z / mean(x, y) = 2 * z / (x + y). The depth of the diamond is its height (in millimetres) measured from the culet (bottom tip) to the table (flat, top surface) as referred in the labelled diagram above.
- table (43-95): It is the width of the top of the diamond relative to widest point. It gives diamond stunning fire and brilliance by reflecting lights to all directions which when seen by an observer, seems lustrous.
- price ($326 - $18826): It is the price of the diamond in US dollars. It is our very target column in the dataset.
- x (0 - 10.74): Length of the diamond (in mm)
- y (0 - 58.9): Width of the diamond (in mm)
- z (0 - 31.8): Depth of the diamond (in mm)

<p align="center"><img src="images/diamonds.jfif" alt="fuller"  width="60%"></p>

- The dataset itself doesn't need any cleaning other than the removal of a few lines where dimensions (y or x) are set to zero, which is physically impossible.
- diamonds.describe yields an univariate analysis for statistical description:

|       |        carat |        depth |        table |        price |            x |            y |            z |
|------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| count | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 |
|  mean |     0.797817 |    61.751931 |    57.451161 |  3934.409644 |     5.730712 |     5.734333 |     3.538648 |
|   std |     0.474126 |     1.430026 |     2.233450 |  3989.333861 |     1.121920 |     1.145344 |     0.706817 |
|   min |     0.200000 |    43.000000 |    43.000000 |   326.000000 |     0.000000 |     0.000000 |     0.000000 |
|   25% |     0.400000 |    61.000000 |    56.000000 |   949.000000 |     4.710000 |     4.720000 |     2.910000 |
|   50% |     0.700000 |    61.800000 |    57.000000 |  2401.000000 |     5.690000 |     5.710000 |     3.520000 |
|   75% |     1.040000 |    62.500000 |    59.000000 |  5331.250000 |     6.540000 |     6.540000 |     4.040000 |
|   max |     5.010000 |    79.000000 |    95.000000 | 18823.000000 |    10.740000 |    58.900000 |    31.800000 |

---

## Exploring each of the attributes:

### Price

- "Price", as expected, is skewed. There are few diamonds which are worth too much and a lot of diamonds with reasonably small prices.

<p align="center"><img src="images/prices.png" alt="prices"  width="100%"></p>

---

### Cuts

- Most of the diamonds have **Ideal Cuts** with a ratio of **39.95%** followed by **Premium Cuts** and **Very Good Cuts**
<p align="center"><img src="images/cuts.png" alt="cut"  width="75%"></p>

-In absolute values, we get:

<p align="center"><img src="images/cuts_abs.png" alt="cut"  width="100%"></p>

#### Price distribution of diamond cuts:

<p align="center"><img src="images/cut_prices.png" alt="cuts"  width="100%"></p>

- Most of the diamonds with **Ideal Cut** costs between **$326** and **$2500**
- Most of the diamonds with **Premium Cut** costs between **$326** and **$5000**
- Most of the diamonds with **Very Good Cut** costs between **$336** and **$4800**
- Most of the diamonds with **Good Cut** costs between **$327** and **$4700**
- Most of the diamonds with **Fair Cut** costs between **$337** and **$5000**

---

### Colors

- Most of the diamonds have **G** color with a ratio of **20.93%** followed by **E** and **F**
- Only a few have **J** (worst) color with a ratio of **5.21%**.

<p align="center"><img src="images/color.png" alt="color"  width="75%"></p>

-In absolute values, we get:

<p align="center"><img src="images/color_abs.png" alt="colors"  width="100%"></p>

#### Price distribution of diamond colors:

<p align="center"><img src="images/color_prices.png" alt="color_prices"  width="100%"></p>

**Insights:**

- Most of the diamonds with **G Color** costs between **$354** and **$2500**
- Most of the diamonds with **E Color** costs between **$326** and **$3700**
- Most of the diamonds with **F Color** costs between **$342** and **$4500**
- Most of the diamonds with **H Color** costs between **$337** and **$5200**
- Most of the diamonds with **D Color** costs between **$357** and **$2500**
- Most of the diamonds with **I Color** costs between **$334** and **$6200**
- Most of the diamonds with **J Color** costs between **$335** and **$6400**

---

### Clarity

- Most of the diamonds have **SI1** clarity with a ratio of **24.22%** followed by **VS2** and **SI2**
- Only a few have **I1** clarity with a ratio of **1.37%**.

<p align="center"><img src="images/clarity.png" alt="clar"  width="75%"></p>

- In absolute values, we get:

<p align="center"><img src="images/clarity_abs.png" alt="colors"  width="100%"></p>

#### Price distribution of diamond clarities:

<p align="center"><img src="images/clarity_prices.png" alt="clarity_prices"  width="100%"></p>

**Insights:**

- Most of the diamonds with **SI1 Clarity** costs in between **$326** and **$5100**
- Most of the diamonds with **VS2 Clarity** costs in between **$334** and **$2600**
- Most of the diamonds with **SI2 Clarity** costs in between **$326** and **$5200**
- Most of the diamonds with **VS1 Clarity** costs in between **$327** and **$3600**
- Most of the diamonds with **VVS2 Clarity** costs in between **$336** and **$3500**
- Most of the diamonds with **VVS1 Clarity** costs in between **$336** and **$3000**
- Most of the diamonds with **IF Clarity** costs in between **$369** and **$2500**
- Most of the diamonds with **I1 Clarity** costs in between **$345** and **$7500**

---

### Weights

- The weight distribution is skewed. They mostly fall between **0.2 carat** and **1.2 carat**. 

<p align="center"><img src="images/weights.png" alt="weights"  width="100%"></p>

#### Price distribution of diamond weights:

<p align="center"><img src="images/weights_prices.png" alt="weights_prices"  width="100%"></p>

- The majority costs between **$326** and **$5000**.
- Interestingly enough, KDE plots really do extrapolate values to ranges that may not make any sense (as is the case of diamonds with negative values).
- Please bear in mind that I'm just testing and exploring a couple of different graphs!

---

### Diamond Depth Percentage

- We can see that the depth percentage distribution is normally distributed. Most diamonds fall between values of **60%** and**64%**.

<p align="center"><img src="images/depth.png" alt="depths"  width="100%"></p>

#### Price distribution of depth percentage:

<p align="center"><img src="images/depth_prices.png" alt="depths_prices"  width="100%"></p>

- Most costs fall between **$326** and **$6200**.

---

### Diamond Tables

- Most of diamond tables fall between **54** and **61**.

<p align="center"><img src="images/table.png" alt="depths"  width="100%"></p>

#### Price distribution of tables:

- The majority costs between **$326** and **$3800**.

<p align="center"><img src="images/table_prices.png" alt="depths_prices"  width="100%"></p>

- The table is measured as the width of the top of the diamond relative to the widest point, and it's an artificial feature imposed by the artisan, which explains the
"sawed" distribution seen in both graphs above, as certain ratios are prefered when cutting and shaping the diamonds.

---

## Data Cleaning

- The pairplot immediately tells us that there are some features with datapoints that are far from the rest of its colleagues. This will affect the outcome of our regression model and hence they will be removed.

<p align="center"><img src="images/pairplot_before.png" width="100%"></p>

- Let's examine the regression lines in these distributions as well.

<p align="center"><img src="images/regression.png" alt="reg"  width="100%"></p>

- After dropping the outliers, let's have a look at the new pairwise relationships:

<p align="center"><img src="images/pairplot_clean.png" alt="pp"  width="100%"></p>

---

## Bivariate Analysis:

- Let's analyze the correlation matrix between the variables:

<p align="center">

|       |    carat |     depth |     table |     price |         x |         y |        z |
|------:|---------:|----------:|----------:|----------:|----------:|----------:|---------:|
| carat | 1.000000 |  0.027074 |  0.181688 |  0.922186 |  0.975152 |  0.949687 | 0.951824 |
| depth | 0.027074 |  1.000000 | -0.297123 | -0.012037 | -0.025858 | -0.029903 | 0.094344 |
| table | 0.181688 | -0.297123 |  1.000000 |  0.127832 |  0.195367 |  0.183362 | 0.150646 |
| price | 0.922186 | -0.012037 |  0.127832 |  1.000000 |  0.885019 |  0.864059 | 0.860247 |
|     x | 0.975152 | -0.025858 |  0.195367 |  0.885019 |  1.000000 |  0.972447 | 0.969336 |
|     y | 0.949687 | -0.029903 |  0.183362 |  0.864059 |  0.972447 |  1.000000 | 0.948768 |
|     z | 0.951824 |  0.094344 |  0.150646 |  0.860247 |  0.969336 |  0.948768 | 1.000000 |

</p>

Which can also be visualized as a heatmap of correlations:

<p align="center"><img src="images/heatmap2.png" alt="heat"  width="100%"></p>

The price of a diamond has a direct correlation with its dimensions (and hence with the carat, since the weight of the diamonds is itself a function of its dimensions). It is not a straight linear correlation but an exponential one.
There are other relevant features which also influence its price, such as color, clarity and cut.

---

## A Further Look

### Let's now look at how this diagram varies by associating other features

- Correlation 'price' and 'carat' associated to 'color'

<p align="center"><img src="images/pc_color.png" alt="pc1"  width="80%"></p>

- Correlation 'price' and 'carat' associated to 'clarity'

<p align="center"><img src="images/pc_clarity.png" alt="pc2"  width="80%"></p>

- Correlation 'price' and 'carat' associated to 'cut'

<p align="center"><img src="images/pc_cut.png" alt="pc3"  width="80%"></p>

Since the first plot clearly points to a logarithmic relation between price and carat, we will perform a log transformation so that the additional relations become clearer:

```
	diamonds["carat_log"] = np.log(diamonds["carat"])
	diamonds["price_log"] = np.log(diamonds["price"])
```

- Correlation 'price_log' and 'carat_log' associated to 'color'

<p align="center"><img src="images/pc_color_log.png" alt="pc1l"  width="80%"></p>

- Correlation 'price_log' and 'carat_log' associated to 'clarity'

<p align="center"><img src="images/pc_clarity_log.png" alt="pc2l"  width="80%"></p>

- Correlation 'price_log' and 'carat_log' associated to 'cut'

<p align="center"><img src="images/pc_cut_log.png" alt="pc3l"  width="80%"></p>

---

## Model Creation & Performance Evaluation

- We will first standardize the data and then split the dataset with a ratio of 0.2 i.e. 80% of data will be used for training and 20% for the validation process:

```
	sc = StandardScaler()
	x = diamonds.drop(["price"],axis =1) # price will be our target (y)
	x = sc.fit_transform(x)
	y = diamonds["price"]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

- With the pre-processing done, let's try a couple of different predictive models with SKLearn and compare their respective RMSE's (ideally, this comparison would be much faster with PyCaret, but let's stick with a manual implementation for now, for didactic purposes):

### Linear Regression

```
	lr = LinearRegression()
	lr.fit(x_train,y_train)
	y_pred = lr.predict(x_test)
```

- After performing the Linear Regression with SKLearn, we get:

```
	R Squared Value: 0.8885407086951569
	Adjusted R Squared Value: 0.8884381178154531
	Mean Absolute Error: 844.8370225588242
	Mean Squared Error: 1733016.3581049452
	Root Mean Squared Error: 1316.4407917202145
```

### Decision Tree Regression

```
	dt = DecisionTreeRegressor()
	dt.fit(x_train,y_train)
	y_pred = dt.predict(x_test)
```

- After performing the Decision Tree Regression with SKLearn, we get:

```
	R Squared Value: 0.9640975493985238
	Adjusted R Squared Value: 0.9640645035757162
	Mean Absolute Error: 364.3079280751941
	Mean Squared Error: 558226.5368818962
	Root Mean Squared Error: 747.1455928277273
```

### Random Forest Regression

```
	rf = RandomForestRegressor()
	rf.fit(x_train,y_train)
	y_pred = rf.predict(x_test)
```

- After performing the Random Forest Regression with SKLearn, we get:

```
	R Squared Value: 0.9805862653291031
	Adjusted R Squared Value: 0.9805683962748959
	Mean Absolute Error: 269.82116456983283
	Mean Squared Error: 301852.9847328354
	Root Mean Squared Error: 549.4114894437823
```

### K-Neighbours Regression

```	
	kn = KNeighborsRegressor()
	kn.fit(x_train,y_train)
	y_pred = kn.predict(x_test)
```

- After performing the K-Neighbours Regression with SKLearn, we get:

```
	R Squared Value: 0.9599407056786297
	Adjusted R Squared Value: 0.9599038337570821
	Mean Absolute Error: 402.15884756845116
	Mean Squared Error: 622858.906963629
	Root Mean Squared Error: 789.2141071747444
```

### Model Explainability with SHAP (SHapley Additive exPlanations)

- AI solutions are somewhat mystic in nature to non-technical personel due to their "black box" nature. SHAP comes in handy for model explainability. It can break down the mechanics of any machine learning model and deep neural net to make them understandable to anyone. It is a Python package based on the 2016 NIPS paper about SHAP values. The premise of this paper and Shapley values comes from approaches in game theory. In a nutshel, SHAP offers two advantages: global model interpretability and local interpretability.

- Plotting the Shapley values for one of the models above tells us the relative importante of each attribute on the model output magnitude:

<p align="center"><img src="images/shap1.png" alt="shap1"  width="100%"></p>

- The carat stands out as the driving factor for a diamond's price. Reading the axis title below, we see that the importances are just the average absolute Shapley values for a feature.
- We won't also stop here. In the plot above, we only looked at absolute values of importance. We don't know which feature positively or negatively influences the model. Let's do that with SHAP summary_plot:

<p align="center"><img src="images/shap2.png" alt="shap2"  width="100%"></p>

- The left vertical axis denotes feature names, ordered based on importance from top to bottom.
- The horizontal axis represents the magnitude of the SHAP values for predictions.
- The vertical right axis represents the actual magnitude of a feature as it appears in the dataset and colors the points.
- We see that as carat increases, its effect on the model is more positive. The same is true for y feature. The x and z features are a bit tricky with a cluster of mixed points around the center.

### Exploring each feature with dependence plots

- We can get a deeper insight into each feature's effect on the entire dataset with dependence plots. Let's see an example:

```
	shap.dependence_plot("carat", shap_values_xgb, X_train, interaction_index=None)
```

<p align="center"><img src="images/shap3.png" alt="shap3"  width="100%"></p>

- This plot aligns with what we saw in the summary plot before. As carat increases, its SHAP value increases. By changing the interaction_index parameter to auto, we can color the points with a feature that most strongly interacts with carat:

```
	shap.dependence_plot("carat", shap_values_xgb, X_train, interaction_index="auto");
```

<p align="center"><img src="images/shap4.png" alt="shap4"  width="100%"></p>

- It seems that the carat interacts with the clarity of the diamonds much stronger than other features.
- Similar plots can be created for the other categorical features, as seen in the notebook.
- From this information, we can already answer the first of the 2 objectives set at the beginning of the project:

Objective 1: to infer which characteristics are more likely to influence a diamond's price;

- Answer: If we disregard the x, y and z variables (since their effect is redundant with the carat attribute), the characteristics more likely to influence a diamond's price are,
in order of importance: carat, clarity, color, cut, depth and table. The first 4 attributes are known in the diamond business as the **4 C's** due to their importance in gauging
the quality of any diamond.

Now to predict the prices for Ricks' diamonds!

### Predicting the price of Ricks' diamonds

```
	fig = px.histogram(diamonds, x=["price_log"], title = 'Histgram of Log Price', template = 'plotly_dark')
	fig.show()
```

- The distribution is not exactly normal... Still, it's what we got to work with! Let's see the outcome.
- Also, we should check what Shapiro-Wilks has to say about this!

<p align="center"><img src="images/normal.png" alt="normal"  width="100%"></p>

The PyCaret model comparison tells us that the GradientBoostingRegressor yields the best results for predicting the price. Let's create a function for the model:

```
	def cria_modelos(df, df_pred, x_test, y_test):
    	X = df[x_test]
    	y = df[y_test]

	model = GradientBoostingRegressor()
	model.fit(X,y)
    
    	price_pred = model.predict(df_pred[x_test])
    
    	return price_pred
```

We will now create masks for clarity and color to optimize the modelling process:


```
	# Clarity masks
	# test dataframe
	teste_if = diamonds_test['clarity'] == 'IF'
	teste_vvs1 = diamonds_test['clarity'] == 'VVS1'
	teste_vvs2 = diamonds_test['clarity'] == 'VVS2'
	teste_vs1 = diamonds_test['clarity'] == 'VS1'
	teste_vs2 = diamonds_test['clarity'] == 'VS2'
	teste_si1 = diamonds_test['clarity'] == 'SI1'
	teste_si2 = diamonds_test['clarity'] == 'SI2'
	teste_i1 = diamonds_test['clarity'] == 'I1'

	# rick dataframe
	rick_if = diamonds_rick['clarity'] == 'IF'
	rick_vvs1 = diamonds_rick['clarity'] == 'VVS1'
	rick_vvs2 = diamonds_rick['clarity'] == 'VVS2'
	rick_vs1 = diamonds_rick['clarity'] == 'VS1'
	rick_vs2 = diamonds_rick['clarity'] == 'VS2'
	rick_si1 = diamonds_rick['clarity'] == 'SI1'
	rick_si2 = diamonds_rick['clarity'] == 'SI2'
	rick_i1 = diamonds_rick['clarity'] == 'I1'

	# Color masks
	# test dataframe
	teste_d = diamonds_test['color'] == 'D'
	teste_e = diamonds_test['color'] == 'E'
	teste_f = diamonds_test['color'] == 'F'
	teste_g = diamonds_test['color'] == 'G'
	teste_h = diamonds_test['color'] == 'H'
	teste_i = diamonds_test['color'] == 'I'
	teste_j = diamonds_test['color'] == 'J'
	# rick dataframe
	rick_d = diamonds_rick['color'] == 'D'
	rick_e = diamonds_rick['color'] == 'E'
	rick_f = diamonds_rick['color'] == 'F'
	rick_g = diamonds_rick['color'] == 'G'
	rick_h = diamonds_rick['color'] == 'H'
	rick_i = diamonds_rick['color'] == 'I'
	rick_j = diamonds_rick['color'] == 'J'
```

- Now to create lists

```
	#listas clarity mask
	teste_clarity_list = [
	    teste_if, teste_vvs1, teste_vvs2, teste_vs1, teste_vs2, teste_si1,
	    teste_si2, teste_i1
	]

	rick_clarity_list = [
	    rick_if, rick_vvs1, rick_vvs2, rick_vs1, rick_vs2, rick_si1, rick_si2,
	    rick_i1
	]
	#listas color mask
	teste_color_list = [
	    teste_d, teste_e, teste_f, teste_g, teste_h, teste_i, teste_j
	]

	rick_color_list = [rick_d, rick_e, rick_f, rick_g, rick_h, rick_i, rick_j]
```

- And finally a loop to test all combination of models:

``` 
	for clarity in list(zip(teste_clarity_list, rick_clarity_list)):
		for color in list(zip(teste_color_list, rick_color_list)):
			diamonds_rick.loc[clarity[1] & color[1],
                          'price_predicted'] = cria_modelos(
                              diamonds_test[clarity[0] & color[0]],
                              diamonds_rick[clarity[1] & color[1]],
                              ['carat', 'cut','depth','x','y','z','table'], 'price')
```

- Rounding up the results and saving in a new csv:

```
	diamonds_test[['carat', 'cut','depth','x','y','z','table','price']].applymap(lambda x: round(np.exp(x),ndigits=2))
	diamonds_rick[['carat', 'cut','depth','x','y','z','table', 'price_predicted']].applymap(lambda x: round(np.exp(x),ndigits=2))

	diamonds_rick.to_csv('price_pred.csv', index=False)
```

- This process yields an RMSE of just $579.42, way below our second objective,
	"- to progressively train and test a regression model until its accuracy meet a certain standard (defined by the RMSE). Rick’s goal is to obtain an average error below 900 dollars."

- Rick is now a richer man and we have proved the usefulness of machine learning! 🥳 🥂 🍾

![Abhinandan Trilokia](https://raw.githubusercontent.com/Trilokia/Trilokia/379277808c61ef204768a61bbc5d25bc7798ccf1/bottom_header.svg)
