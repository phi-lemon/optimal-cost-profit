import pandas as pd

INPUT_FILE = 'data/actions.csv'

df = pd.read_csv(INPUT_FILE)

# Transform percent to float
df['profit'] = df['profit'].apply(lambda x: x / 100)
df['gain'] = df['price'] * df['profit']

# Make index start from 1 (not from 0)
df.index += 1

# Make a list of 20 tuples (action, cout action, gain action) prior to compute all combinations
actions_cost_profit = []
for index, rows in df.iterrows():  # n steps -> O(n)
    actions_cost_profit.append((rows.name, rows.price, rows.gain))  # method append: O(1)


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


def roi(combos):
    """
    Function that computes cost and profit for each combination and append it to combination
    :param combos: list of all combinations (each combination is a list of tuples: action_name, cost, profit)
    :return: generator
    """

    for combo in combos:
        cost = 0
        profit = 0

        for action in combo:
            cost += action[1]
            profit += action[2]

        combo.append(cost)
        combo.append(profit)

        yield combo


def best_combo(max_cost):
    all_combos = []
    for comb in roi(get_combos(actions_cost_profit)):
        *actions, cost, profit = comb
        if cost > max_cost:
            continue
        all_combos.append([[*actions], cost, profit])
    best = max(all_combos, key=lambda sublist: sublist[2])

    best_comb_ = [i[0] for i in best[0]]
    min_cost_ = best[1]
    max_profit_ = best[2]

    return best_comb_, min_cost_, max_profit_


best_comb, min_cost, max_profit = best_combo(500)

print(f"Meilleure combinaison : {best_comb} \n"
      f"Cout: {min_cost} \n"
      f"Gain: {max_profit}")
