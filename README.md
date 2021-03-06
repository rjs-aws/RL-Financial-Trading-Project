

### Usage
Invoking the main script involves passing command line arguments:
The second is the local location of the output, the third is the mode (train or evaluate)

```bash
python run_rl_agent.py <your-output-dir> train > run.log &
python run_rl_agent.py <your-output-dir> evaluate <your-input-dir-from-training> eval.log &
```

### Overview for 'total'

The agent holds ```TOTAL``` stocks, comprised of a combination of four assets. (see ```__init__``` trading_env)

At each time step, the agent receives a vector of weights pertaining to the desired balance of the portfolio. (see ```step``` in trading_env)

Upon receiving these weights, stocks are bought/sold to achieve the desired balance for each asset in the portfolio. The quantity to buy/sell is determined in ```_balance_portfolio``` meethod in ```trading_env```.

___Example___: a weight of .25 indicates 25% of the portfolio should be comprised of that asset.

Assets are then bought/sold to achieve this balance. If 18 of this asset are held currently, and the target balance is 25% (15/60), three of this asset can be sold to reach the target balance. (see ```_step```)

The reward is the cumulative profit/loss resulting from the buying or selling stocks to achieve the desired balance.

Take for example the following, using four assets:
```
1. Corresponding Assets ['Asset_One', 'Asset_Two', 'Asset_Three', 'Asset_Four']
2. Portfolio composition (pre-balance): {0: 3, 1: 20, 2: 31, 3: 6}
3. Weights :[0.08124855 0.10962831 0.50053082 0.30859232]
3. Portfolio target list: [5, 7, 30, 19]
4. Action to take: 2 -13 -1 13
```

For this composition (2), provided these weights (3), we have a target allocation (3), reached by these 
corresponding actions (4), where a postive quantity pertains to buying the quantity, and a negative quantity
pertains to selling this quantity.


For each of these buy/sell actions, the inventory quantity for each asset is adjusted. When assets are sold 
the corresponding profit/loss is added to the total. This same profit/loss is used for rewarding the agent.

### Overview for 'budget'

The agent is given a budget, set in the constructor, from which it can purchase assets (25000.00 in its current state)

At the start of each episode, half of the total budget is invested evenly into each asset by invoking ```_initial_asset_allocation_budget```.

At each time step, the agent recieves a vector of weights pertaining to the target monetary balance of each asset in the portfolio.

To achieve this balance, stocks are bought/sold for each asset. The buy/sell process is performed in ```_balance_portfolio_budget```.
