# Extrapolation

In this notebook you will find:
- How to get a survival curve using xgbse
- How to extrapolate your predicted survival curve using the `xgbse.extrapolation` module

## Metrabic

We will be using the Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) dataset from [pycox](https://github.com/havakv/pycox#datasets) as base for this example.


```python
from xgbse.converters import convert_to_structured
from pycox.datasets import metabric
import numpy as np

# getting data
df = metabric.read_df()

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
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>duration</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.603834</td>
      <td>7.811392</td>
      <td>10.797988</td>
      <td>5.967607</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>56.840000</td>
      <td>99.333336</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.284882</td>
      <td>9.581043</td>
      <td>10.204620</td>
      <td>5.664970</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>85.940002</td>
      <td>95.733330</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.920251</td>
      <td>6.776564</td>
      <td>12.431715</td>
      <td>5.873857</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>48.439999</td>
      <td>140.233337</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.654017</td>
      <td>5.341846</td>
      <td>8.646379</td>
      <td>5.655888</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>66.910004</td>
      <td>239.300003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.456747</td>
      <td>5.339741</td>
      <td>10.555724</td>
      <td>6.008429</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>67.849998</td>
      <td>56.933334</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Split and create Time Bins

Split the data in train and test, using sklearn API. We also setup the TIME_BINS arange, which will be used to fit the survival curve


```python
from xgbse.converters import convert_to_structured
from sklearn.model_selection import train_test_split

# splitting to X, T, E format
X = df.drop(['duration', 'event'], axis=1)
T = df['duration']
E = df['event']
y = convert_to_structured(T, E)

# splitting between train, and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)
TIME_BINS = np.arange(15, 315, 15)
TIME_BINS
```




    array([ 15,  30,  45,  60,  75,  90, 105, 120, 135, 150, 165, 180, 195,
           210, 225, 240, 255, 270, 285, 300])



## Fit model and predict survival curves

The package follows `scikit-learn` API, with a minor adaptation to work with time and event data. The model outputs the probability of survival, in a `pd.Dataframe` where columns represent different times.


```python
from xgbse import XGBSEDebiasedBCE

# fitting xgbse model
xgbse_model = XGBSEDebiasedBCE()
xgbse_model.fit(X_train, y_train, time_bins=TIME_BINS)

# predicting
survival = xgbse_model.predict(X_test)
survival.head()
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
      <th>15</th>
      <th>30</th>
      <th>45</th>
      <th>60</th>
      <th>75</th>
      <th>90</th>
      <th>105</th>
      <th>120</th>
      <th>135</th>
      <th>150</th>
      <th>165</th>
      <th>180</th>
      <th>195</th>
      <th>210</th>
      <th>225</th>
      <th>240</th>
      <th>255</th>
      <th>270</th>
      <th>285</th>
      <th>300</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.983502</td>
      <td>0.951852</td>
      <td>0.923277</td>
      <td>0.900028</td>
      <td>0.862270</td>
      <td>0.799324</td>
      <td>0.715860</td>
      <td>0.687257</td>
      <td>0.651314</td>
      <td>0.610916</td>
      <td>0.568001</td>
      <td>0.513172</td>
      <td>0.493194</td>
      <td>0.430701</td>
      <td>0.377675</td>
      <td>0.310496</td>
      <td>0.272169</td>
      <td>0.225599</td>
      <td>0.184878</td>
      <td>0.144089</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.973506</td>
      <td>0.917739</td>
      <td>0.839154</td>
      <td>0.710431</td>
      <td>0.663119</td>
      <td>0.558886</td>
      <td>0.495204</td>
      <td>0.364995</td>
      <td>0.311628</td>
      <td>0.299939</td>
      <td>0.226226</td>
      <td>0.191373</td>
      <td>0.171697</td>
      <td>0.144864</td>
      <td>0.112447</td>
      <td>0.089558</td>
      <td>0.081137</td>
      <td>0.057679</td>
      <td>0.048563</td>
      <td>0.035985</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.986894</td>
      <td>0.959209</td>
      <td>0.919768</td>
      <td>0.889910</td>
      <td>0.853239</td>
      <td>0.777208</td>
      <td>0.725381</td>
      <td>0.649177</td>
      <td>0.582569</td>
      <td>0.531787</td>
      <td>0.485275</td>
      <td>0.451667</td>
      <td>0.428899</td>
      <td>0.386413</td>
      <td>0.344369</td>
      <td>0.279685</td>
      <td>0.242064</td>
      <td>0.187967</td>
      <td>0.158121</td>
      <td>0.118562</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.986753</td>
      <td>0.955210</td>
      <td>0.910354</td>
      <td>0.857684</td>
      <td>0.824301</td>
      <td>0.769262</td>
      <td>0.665805</td>
      <td>0.624934</td>
      <td>0.583592</td>
      <td>0.537261</td>
      <td>0.493957</td>
      <td>0.443193</td>
      <td>0.416702</td>
      <td>0.376552</td>
      <td>0.308947</td>
      <td>0.237033</td>
      <td>0.177140</td>
      <td>0.141838</td>
      <td>0.117917</td>
      <td>0.088937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.977348</td>
      <td>0.940368</td>
      <td>0.873695</td>
      <td>0.804796</td>
      <td>0.742655</td>
      <td>0.632426</td>
      <td>0.556008</td>
      <td>0.521490</td>
      <td>0.493577</td>
      <td>0.458477</td>
      <td>0.416363</td>
      <td>0.391099</td>
      <td>0.364431</td>
      <td>0.291472</td>
      <td>0.223758</td>
      <td>0.190398</td>
      <td>0.165911</td>
      <td>0.120061</td>
      <td>0.095512</td>
      <td>0.069566</td>
    </tr>
  </tbody>
</table>
</div>



## Survival curves visualization


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4), dpi=120)

plt.plot(
    survival.columns,
    survival.iloc[42],
    'k--',
    label='Survival'
)

plt.title('Sample of predicted survival curves - $P(T>t)$')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fc38026cef0>





![svg](../img/output_8_1.svg)



Notice that this predicted survival curve does not end at zero (cure fraction due to censored data). In some cases it might be useful to extrapolate our survival curves using specific strategies. `xgbse.extrapolation` implements a constant risk extrapolation strategy.

## Extrapolation


```python
from xgbse.extrapolation import extrapolate_constant_risk

# extrapolating predicted survival
survival_ext = extrapolate_constant_risk(survival, 450, 11)
survival_ext.head()
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
      <th>15.0</th>
      <th>30.0</th>
      <th>45.0</th>
      <th>60.0</th>
      <th>75.0</th>
      <th>90.0</th>
      <th>105.0</th>
      <th>120.0</th>
      <th>135.0</th>
      <th>150.0</th>
      <th>...</th>
      <th>315.0</th>
      <th>330.0</th>
      <th>345.0</th>
      <th>360.0</th>
      <th>375.0</th>
      <th>390.0</th>
      <th>405.0</th>
      <th>420.0</th>
      <th>435.0</th>
      <th>450.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.983502</td>
      <td>0.951852</td>
      <td>0.923277</td>
      <td>0.900028</td>
      <td>0.862270</td>
      <td>0.799324</td>
      <td>0.715860</td>
      <td>0.687257</td>
      <td>0.651314</td>
      <td>0.610916</td>
      <td>...</td>
      <td>0.112299</td>
      <td>0.068213</td>
      <td>0.032292</td>
      <td>0.011915</td>
      <td>0.003426</td>
      <td>0.000768</td>
      <td>0.000134</td>
      <td>1.825794e-05</td>
      <td>1.937124e-06</td>
      <td>1.601799e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.973506</td>
      <td>0.917739</td>
      <td>0.839154</td>
      <td>0.710431</td>
      <td>0.663119</td>
      <td>0.558886</td>
      <td>0.495204</td>
      <td>0.364995</td>
      <td>0.311628</td>
      <td>0.299939</td>
      <td>...</td>
      <td>0.026665</td>
      <td>0.014641</td>
      <td>0.005957</td>
      <td>0.001796</td>
      <td>0.000401</td>
      <td>0.000066</td>
      <td>0.000008</td>
      <td>7.404100e-07</td>
      <td>4.986652e-08</td>
      <td>2.488634e-09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.986894</td>
      <td>0.959209</td>
      <td>0.919768</td>
      <td>0.889910</td>
      <td>0.853239</td>
      <td>0.777208</td>
      <td>0.725381</td>
      <td>0.649177</td>
      <td>0.582569</td>
      <td>0.531787</td>
      <td>...</td>
      <td>0.088900</td>
      <td>0.049982</td>
      <td>0.021071</td>
      <td>0.006660</td>
      <td>0.001579</td>
      <td>0.000281</td>
      <td>0.000037</td>
      <td>3.735612e-06</td>
      <td>2.798762e-07</td>
      <td>1.572266e-08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.986753</td>
      <td>0.955210</td>
      <td>0.910354</td>
      <td>0.857684</td>
      <td>0.824301</td>
      <td>0.769262</td>
      <td>0.665805</td>
      <td>0.624934</td>
      <td>0.583592</td>
      <td>0.537261</td>
      <td>...</td>
      <td>0.067080</td>
      <td>0.038160</td>
      <td>0.016373</td>
      <td>0.005299</td>
      <td>0.001293</td>
      <td>0.000238</td>
      <td>0.000033</td>
      <td>3.462388e-06</td>
      <td>2.734946e-07</td>
      <td>1.629408e-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.977348</td>
      <td>0.940368</td>
      <td>0.873695</td>
      <td>0.804796</td>
      <td>0.742655</td>
      <td>0.632426</td>
      <td>0.556008</td>
      <td>0.521490</td>
      <td>0.493577</td>
      <td>0.458477</td>
      <td>...</td>
      <td>0.050668</td>
      <td>0.026879</td>
      <td>0.010385</td>
      <td>0.002923</td>
      <td>0.000599</td>
      <td>0.000089</td>
      <td>0.000010</td>
      <td>7.701555e-07</td>
      <td>4.442463e-08</td>
      <td>1.866412e-09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# plotting extrapolation #

plt.figure(figsize=(12,4), dpi=120)

plt.plot(
    survival.columns,
    survival.iloc[42],
    'k--',
    label='Survival'
)

plt.plot(
    survival_ext.columns,
    survival_ext.iloc[42],
    'tomato',
    alpha=0.5,
    label='Extrapolated Survival'
)

plt.title('Extrapolation of survival curves')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fc3801842b0>





![svg](../img/output_12_1.svg)
