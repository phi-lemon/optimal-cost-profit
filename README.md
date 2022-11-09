Given a list of stocks with price and cost (see format of the file in data folder), look for the optimal subset for a given max cost.<br>
Only one of each stock can be bought.<br>

Project made while studying python, with 2 algorithms, bruteforce and 0/1 knapsack.

# Bruteforce
```python
INPUT_FILE = 'data/actions.csv'
```

## Load csv


```python
import pandas as pd

df = pd.read_csv(INPUT_FILE)

# Transform percent to float
df['profit'] = df['profit'].apply(lambda x: x / 100)
df['gain'] = df['price'] * df['profit']

df.index +=1
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>price</th>
      <th>profit</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A-1</td>
      <td>20</td>
      <td>0.05</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A-2</td>
      <td>30</td>
      <td>0.10</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A-3</td>
      <td>50</td>
      <td>0.15</td>
      <td>7.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A-4</td>
      <td>70</td>
      <td>0.20</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A-5</td>
      <td>60</td>
      <td>0.17</td>
      <td>10.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A-6</td>
      <td>80</td>
      <td>0.25</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A-7</td>
      <td>22</td>
      <td>0.07</td>
      <td>1.54</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A-8</td>
      <td>26</td>
      <td>0.11</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A-9</td>
      <td>48</td>
      <td>0.13</td>
      <td>6.24</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A-10</td>
      <td>34</td>
      <td>0.27</td>
      <td>9.18</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A-11</td>
      <td>42</td>
      <td>0.17</td>
      <td>7.14</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A-12</td>
      <td>110</td>
      <td>0.09</td>
      <td>9.90</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A-13</td>
      <td>38</td>
      <td>0.23</td>
      <td>8.74</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A-14</td>
      <td>14</td>
      <td>0.01</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A-15</td>
      <td>18</td>
      <td>0.03</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A-16</td>
      <td>8</td>
      <td>0.08</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A-17</td>
      <td>4</td>
      <td>0.12</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A-18</td>
      <td>10</td>
      <td>0.14</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>19</th>
      <td>A-19</td>
      <td>24</td>
      <td>0.21</td>
      <td>5.04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>A-20</td>
      <td>114</td>
      <td>0.18</td>
      <td>20.52</td>
    </tr>
  </tbody>
</table>
</div>



Make a list of 20 tuples (action, cout action, gain action)


```python
# Make a list of 20 tuples (action, cout action, gain action) prior to compute all combinations
actions_cost_profit = []

# Iterate over each row
for index, rows in df.iterrows():  # n steps -> O(n)
    actions_cost_profit.append((rows.name, rows.price, rows.gain))  # method append: O(1)
```

**Time complexity to get the list of actions : O(n)**

## Get all possible combinations


```python
def get_combos(actions):
    """
    Recursive function to get all actions combinations
    :param actions: list of actions (tuples: 'action_name', cost, profit)
    :return: all actions combination
    """

    if not actions:
        # BASE CASE
        return [[]]

    # RECURSIVE CASE
    combos = []
    head = [actions[0]]
    tail = actions[1:]

    tail_combos = get_combos(tail)
    for t in tail_combos:
        combos.append(head + t)

    combos = combos + tail_combos
    return combos
```

### Time complexity of get_combos

3 tests :
* dataset actions1.csv (20 actions) : time 0.4157
* dataset actions2.csv (40 actions) : time 0.4201
* dataset actions3.csv (80 actions) : time 0.4236


```python
import matplotlib.pyplot as plt
plt.plot([20, 40, 80], [0.4157, 0.4201, 0.4236], color='green', marker='o', linewidth=2)
```




    [<matplotlib.lines.Line2D at 0x2002f55cdc0>]




    
![png](output_11_1.png)
    


The get_combos function has a complexity of **O(log n)**.<br>

## Get ROI


```python
def roi(combos):
    """
    Function that computes cost and profit for each combination and append it to combination
    :param combos: list of all combinations (each combination is a list of tuples: action_name, cost, profit)
    :return: generator
    """

    for combo in combos:  # steps = c = nb combos
        cost = 0
        profit = 0

        for action in combo:  # steps = n = nb actions
            cost += action[1]
            profit += action[2]

        combo.append(cost)  # method append() : O(1)
        combo.append(profit)

        yield combo
```

### Time complexity of roi
The roi function has a time complexity of O(nc)<br>
As c = 2<sup>n</sup>, the roi function has a time complexity of O(n2<sup>n</sup>), or **O(2<sup>n</sup>)**

## Get max profit for cost <= 500


```python
def best_combo(max_cost):
    all_combos = []
    for comb in roi(get_combos(actions_cost_profit)):  # steps = nomb combos
        *actions, cost, profit = comb
        if cost > max_cost:
            continue
        all_combos.append([[*actions], cost, profit])  # method append(): O(1)
    best = max(all_combos, key=lambda sublist: sublist[2])  # max: O(n)

    best_comb_ = [i[0] for i in best[0]]
    min_cost_ = best[1]
    max_profit_ = best[2]

    return best_comb_, min_cost_, max_profit_


best_comb, min_cost, max_profit = best_combo(500)

print(f"Meilleure combinaison : {best_comb} \n"
      f"Cout: {min_cost} \n"
      f"Gain: {max_profit}")
```

    Meilleure combinaison : [4, 5, 6, 8, 10, 11, 13, 18, 19, 20] 
    Cout: 498 
    Gain: 99.08000000000001
    

### Time complexity of best_combo
The best_combo function has a time complexity of **O(n)**

### Total complexity : O(2<sup>n</sup>)
Time will increase exponentially as data grows


# Optimize w knapsack 0/1 algorithm


```python
INPUT_FILE = 'data/dataset2.csv'
```


```python
import pandas as pd

df = pd.read_csv(INPUT_FILE)

# Transform percent to float
df['profit'] = df['profit'].apply(lambda x: x / 100)
df['gain'] = df['price'] * df['profit']

df.index +=1

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>price</th>
      <th>profit</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Share-MOEX</td>
      <td>40.60</td>
      <td>0.1669</td>
      <td>6.776140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Share-GBGY</td>
      <td>27.08</td>
      <td>0.3409</td>
      <td>9.231572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Share-AXWK</td>
      <td>-9.27</td>
      <td>0.2719</td>
      <td>-2.520513</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Share-FJTI</td>
      <td>33.50</td>
      <td>0.2081</td>
      <td>6.971350</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Share-LGDP</td>
      <td>15.26</td>
      <td>0.0340</td>
      <td>0.518840</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>price</th>
      <th>profit</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.613130</td>
      <td>0.196603</td>
      <td>2.485969</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.238445</td>
      <td>0.119171</td>
      <td>4.113573</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.950000</td>
      <td>0.001500</td>
      <td>-3.518640</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.089750</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.370000</td>
      <td>0.198100</td>
      <td>0.457746</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27.162500</td>
      <td>0.305725</td>
      <td>4.424702</td>
    </tr>
    <tr>
      <th>max</th>
      <td>51.460000</td>
      <td>0.399800</td>
      <td>19.441917</td>
    </tr>
  </tbody>
</table>
</div>



### Missing values


```python
df.isna().sum()
```




    name      0
    price     0
    profit    0
    gain      0
    dtype: int64



### Duplicated rows


```python
df[df.duplicated()].shape[0]
```




    0



### Incorrect values
Neither price or profit can't be <= 0.


```python
df_OK = df.drop(df[df['price']<=0].index)
df_OK = df_OK.drop(df_OK[df_OK['profit']<=0].index)
```


```python
df_OK.describe()
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
      <th>price</th>
      <th>profit</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.606543</td>
      <td>0.195685</td>
      <td>5.055476</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.498809</td>
      <td>0.122011</td>
      <td>4.039245</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.120000</td>
      <td>0.001500</td>
      <td>0.010968</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.310000</td>
      <td>0.085800</td>
      <td>1.703547</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.120000</td>
      <td>0.197100</td>
      <td>3.996740</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.010000</td>
      <td>0.310000</td>
      <td>7.796250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>51.460000</td>
      <td>0.399700</td>
      <td>19.441917</td>
    </tr>
  </tbody>
</table>
</div>



### round costs to superior int
So that we can use 0/1 knapsack algorithm, and real cost should never be > max allowed.<br>
If action == 0: do not round. Actions == 0 should be deleted first.


```python
# Arrondir à l'entier supérieur sauf si l'action = 0
import math
df_OK.insert(2, "ceil_price", 0)

if df_OK.price.dtype != 'int64':
    df_OK.ceil_price = df_OK.price.apply(lambda x: math.ceil(x))
else:
    df_OK.ceil_price = df_OK.price
```


```python
def knapsack(w, wt, val):
    """
    Create a dynamic programming table to get the best profit for each cost
    :param w: int, total maximum price.
    :param wt: list of prices.
    :param val: list of gains
    :return: optimal value, dp table (list of lists)
    """
    n = len(val)  # Nb items
    dp = [[0] * (w + 1) for _ in range(n + 1)]  # Fill array with 0

    for i in range(1, n + 1):
        for w_ in range(1, w + 1):
            if wt[i - 1] <= w_:
                dp[i][w_] = max(val[i - 1] + dp[i - 1][w_ - wt[i - 1]], dp[i - 1][w_])
            else:
                dp[i][w_] = dp[i - 1][w_]
    return dp[n][w_], dp
```


```python
def knapsack_possible_subset(w: int, wt: list, val: list):
    """
    returns one of the optimal subsets.
    :param w: int, total maximum price.
    :param wt: list, the vector of prices. wt[i] is the weight of the i-th item
    :param val: list, the vector of gains. val[i] is the gain of the i-th item
    :return: optimal_val: float, the optimal gain
    opt_set: set, the indices of the optimal subsets
    total_cost: total cost of optimal subset
    """
    optimal_val, dp_table = knapsack(w, wt, val)
    opt_set: set = set()
    nb_items = len(val)
    _make_subset_from_table(dp_table, wt, nb_items, w, opt_set)
    total_cost = sum([wt[i-1] for i in opt_set])
    return optimal_val, total_cost, opt_set
```


```python
def _make_subset_from_table(dp: list, wt: list, i: int, j: int, optimal_set: set):
    """
    Recursively look for an optimal subset from the dp table
    and the list of prices

    :param dp: list of list (dp table)
    :param wt: list, prices of the items
    :param i: int, index of the current item
    :param j: int, the current possible maximum price
    :param optimal_set: set, optimal subset recursively modified
    by the function.
    :return: None
    """
    if i > 0 and j > 0:
        if dp[i - 1][j] == dp[i][j]:
            _make_subset_from_table(dp, wt, i - 1, j, optimal_set)
        else:
            optimal_set.add(i)
            _make_subset_from_table(dp, wt, i - 1, j - wt[i - 1], optimal_set)
```


```python
# reindex
df_OK = df_OK.reset_index(drop=True)
df_OK.index +=1
```


```python
ceil_prices = df_OK.ceil_price.tolist()
gains = df_OK.gain.tolist()

optimal_val, total_cost, example_set = knapsack_possible_subset(500, ceil_prices, gains)

actions = list(example_set)

opt_df = df_OK.loc[actions,:]
```


```python
total_cost_real = opt_df.price.sum()
best_set = opt_df.name.tolist()

print(f"Meilleure combinaison : {best_set} \n"
      f"Cout: {round(total_cost_real, 2)} €\n"
      f"Gain: {round(optimal_val, 2)} €")
```

    Meilleure combinaison : ['Share-LFXB', 'Share-GEBJ', 'Share-OPBR', 'Share-NDKR', 'Share-PLLK', 'Share-ZOFA', 'Share-PATS', 'Share-FWBE', 'Share-IJFT', 'Share-ANFX', 'Share-JWGF', 'Share-JGTW', 'Share-ZKSN', 'Share-DWSK', 'Share-ECAQ', 'Share-FAPS', 'Share-ALIY'] 
    Cout: 493.1 €
    Gain: 194.9 €
    


```python
opt_df
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
      <th>name</th>
      <th>price</th>
      <th>ceil_price</th>
      <th>profit</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>Share-LFXB</td>
      <td>14.83</td>
      <td>15</td>
      <td>0.3979</td>
      <td>5.900857</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Share-GEBJ</td>
      <td>5.87</td>
      <td>6</td>
      <td>0.3795</td>
      <td>2.227665</td>
    </tr>
    <tr>
      <th>454</th>
      <td>Share-OPBR</td>
      <td>39.00</td>
      <td>39</td>
      <td>0.3895</td>
      <td>15.190500</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Share-NDKR</td>
      <td>33.06</td>
      <td>34</td>
      <td>0.3991</td>
      <td>13.194246</td>
    </tr>
    <tr>
      <th>524</th>
      <td>Share-PLLK</td>
      <td>19.94</td>
      <td>20</td>
      <td>0.3991</td>
      <td>7.958054</td>
    </tr>
    <tr>
      <th>526</th>
      <td>Share-ZOFA</td>
      <td>25.32</td>
      <td>26</td>
      <td>0.3978</td>
      <td>10.072296</td>
    </tr>
    <tr>
      <th>399</th>
      <td>Share-PATS</td>
      <td>27.70</td>
      <td>28</td>
      <td>0.3997</td>
      <td>11.071690</td>
    </tr>
    <tr>
      <th>528</th>
      <td>Share-FWBE</td>
      <td>18.31</td>
      <td>19</td>
      <td>0.3982</td>
      <td>7.291042</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Share-IJFT</td>
      <td>40.91</td>
      <td>41</td>
      <td>0.3889</td>
      <td>15.909899</td>
    </tr>
    <tr>
      <th>437</th>
      <td>Share-ANFX</td>
      <td>38.55</td>
      <td>39</td>
      <td>0.3972</td>
      <td>15.312060</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Share-JWGF</td>
      <td>48.69</td>
      <td>49</td>
      <td>0.3993</td>
      <td>19.441917</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Share-JGTW</td>
      <td>35.29</td>
      <td>36</td>
      <td>0.3943</td>
      <td>13.914847</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Share-ZKSN</td>
      <td>22.83</td>
      <td>23</td>
      <td>0.3863</td>
      <td>8.819229</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Share-DWSK</td>
      <td>29.49</td>
      <td>30</td>
      <td>0.3935</td>
      <td>11.604315</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Share-ECAQ</td>
      <td>31.66</td>
      <td>32</td>
      <td>0.3949</td>
      <td>12.502534</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Share-FAPS</td>
      <td>32.57</td>
      <td>33</td>
      <td>0.3954</td>
      <td>12.878178</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Share-ALIY</td>
      <td>29.08</td>
      <td>30</td>
      <td>0.3993</td>
      <td>11.611644</td>
    </tr>
  </tbody>
</table>
</div>


