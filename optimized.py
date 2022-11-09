import pandas as pd
import math

INPUT_FILE = 'data/dataset1.csv'

df = pd.read_csv(INPUT_FILE)

# Transform percent to float
df['profit'] = df['profit'].apply(lambda x: x / 100)
df['gain'] = df['price'] * df['profit']

# Make index start from 1 (not from 0)
df.index += 1

# clean data
df_OK = df.drop(df[df['price'] <= 0].index)
df_OK = df_OK.drop(df_OK[df_OK['profit'] <= 0].index)

# round costs to superior int
# So that we can use 0/1 knapsack algorithm,
# and real cost should never be > max allowed.
# If action == 0: do not round. Actions == 0 should be deleted first.

df_OK.insert(2, "ceil_price", 0)

# Convert prices only if not int
if df_OK.price.dtype != 'int64':
    df_OK.ceil_price = df_OK.price.apply(lambda x: math.ceil(x))
else:
    df_OK.ceil_price = df_OK.price


def knapsack(w: int, wt: list, val: list):
    """
    Create a dynamic programming table to get the best profit for each cost
    :param w: int, total maximum price.
    :param wt: list of prices.
    :param val: list of gains
    :return: optimal value, dp table (list of lists)
    """
    n = len(val)
    dp = [[0] * (w + 1) for _ in range(n + 1)]  # Fill array with 0

    w_ = 0
    for i in range(1, n + 1):
        for w_ in range(1, w + 1):
            if wt[i - 1] <= w_:
                dp[i][w_] = max(val[i - 1] + dp[i - 1][w_ - wt[i - 1]],
                                dp[i - 1][w_])
            else:
                dp[i][w_] = dp[i - 1][w_]
    return dp[n][w_], dp


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
    total_cost = sum([wt[i - 1] for i in opt_set])
    return optimal_val, total_cost, opt_set


def _make_subset_from_table(dp: list, wt: list, i: int, j: int,
                            optimal_set: set):
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


# reindex
df_OK = df_OK.reset_index(drop=True)
df_OK.index += 1

# Make lists from dataframe
ceil_prices = df_OK.ceil_price.tolist()
gains = df_OK.gain.tolist()

opt_val, total_price, set_ = knapsack_possible_subset(500, ceil_prices, gains)

actions = list(set_)

# Final dataframe
opt_df = df_OK.loc[actions, :]

total_cost_real = opt_df.price.sum()
best_set = opt_df.name.tolist()

print(f"Meilleure combinaison : {best_set} \n"
      f"Cout: {round(total_cost_real, 2)} \n"
      f"Gain: {round(opt_val, 2)}")
