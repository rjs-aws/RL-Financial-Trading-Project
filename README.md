## Overview 
The agent holds ```TOTAL``` stocks, comprised of a combination of four assets. (see ```__init__``` trading_env)

At each time step, the agent receives a vector of weights pertaining to the desired balance of the portfolio. (see ```step``` in trading_env)

Upon receiving these weights, stocks are bought/sold to achieve the desired balance for each asset in the portfolio. The quantity to buy/sell is determined in ```_balance_portfolio``` meethod in ```trading_env```.

___Example___: a weight of .25 indicates 25% of the portfolio should be comprised of that asset.

Assets are then bought/sold to achieve this balance. If 18 of this asset are held currently, and the target balance is 25% (15/60), three of this asset can be sold to reach the target balance. (see ```_step```)

The reward is the cumulative profit/loss resulting from the buying or selling stocks to achieve the desired balance.

### Usage
Invoking the main script involves passing command line arguments:
The second is the local location of the output, the third is the mode (train or evaluate)

```bash
python run_rl_agent.py <your-output-dir> train > run.log &
python run_rl_agent.py <your-output-dir> evaluate <your-other-outdir> > eval.log &
```

