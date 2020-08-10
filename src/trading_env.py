import gym
from gym import spaces
import numpy as np
import pandas as pd
import math
import logging
import csv

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Dense, Input, GRU, concatenate, Activation
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from config import *
from actions import Action
import collections

# current time string used for the log file
date_now_str = datetime.datetime.now().strftime("%m%d%y_%H:%M")

# if the logging configuration is invoked w/o
# a filename argument, logs will be written to stdout
# if a filename is used for the argument, the log file
# will be written within the running container
logging.basicConfig(
    #     filename="{}_run.log".format(date_now_str),
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class TradingEnv(gym.Env):
    """ 
        @param assets: the names of the assets' data files to trade
        @param mode: Decision can be made on past (using actual past) or future (using RNN forecast)
    """

    def __init__(
        self,
        assets=[
            "GOOG_test",
            "AMZN_test",
            "MSFT_test",
            "AAPL_test",
            "CSCO_test"
        ],  
        mode="budget",
        span=9,  # number of days lag
    ):

        if mode not in ("total", "budget"):
            raise ValueError("Argument must be either 'budget' or 'total'")

        NUM_ASSETS = len(assets)

        self.csv_file = CSV_DIR  # Output filename
        self.assets = assets
        self.mode = mode
        self.span = span
        self.infos = list()
        self.t = 0

        # this arrangement allows for TOTAL assets to be held at one
        # time.

        if self.mode == "total":
            self.TOTAL = 60

        # allocate a certain budget at the start

        if self.mode == "budget":
            self.budget = 25000.00

        self.data = self._get_data_sets(
            assets_list=assets
        )  # collect data from assets' files

        # store the "Close" prices from each corresponding csv into a local member dict,
        # key is literal asset name: XXX_test, value is list of prices
        self.observations = self.get_all_observations(assets)

        # initialize inventory
        self._initialize_inventory()
        
        # initial inventory allocation
        self._initial_asset_allocation_budget(NUM_ASSETS)

        # throws assertion error if all assets' observations are not the same length (ensures files are same length)
        self._check_observations_length()

        # the number of steps is determined by the length of the input files.
        self.steps = len(list(self.observations.values())[0]) - 1

        # Each index represents the target weight of the asset in the portfolio
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(NUM_ASSETS,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_ASSETS, 1, self.span + 1),
            dtype=np.float32,
        )

    def reset(self):
        """
            Initializes a training episode.
            
            t: internal state for timestamp
            total_profit: tracks the accumulated profit
            inventory: key, value pairs str, List of stocks for each asset_name
            infos: debugging info, populated during step
        """
        self.t = 0
        self.total_profit = 0
        self.infos = []
        self.budget = 25000.00

        # create the empty inventory, and allocate the budget
        self._initialize_inventory()
        self._initial_asset_allocation_budget(len(self.assets))

        # Define environment data
        obs = self.get_state()

        return obs

    def _step(self, corresponding_asset_index, action, using_budget=False):
        """
            Invoked once per asset.
            Performs the corresponding action for
            the given asset and calculates the margin
            and reward for that asset.
            @param corresponding_asset_index the asset's index, uses for retrieving the asset's observations
            @param action the action to take (-TOTAL, TOTAL), where the action corresponds to the number of that
            asset to buy or sell (pos is buy; neg is sell)
        """

        # assets are stored as an ordered dict
        # get the corresponding assets from dict
        # The asset index arg corresponds to the index
        # from the asset index list
        asset_index_list = list(self.observations.keys())
        asset_name = asset_index_list[corresponding_asset_index]

        logging.info(
            "Asset Index: {} Asset name: {} Recieved Action {} in _step".format(
                corresponding_asset_index, asset_name, action
            )
        )

        # determine the appropriate action
        margin = None

        # record action for logging
        action_str = None

        # BUY the positive quantity from the action
        if action > 0:
            action_str = "BUY"

            # add to the inventory the corresponding asset at the timestamp
            asset_price_at_t = self._get_price_for_asset_at_time(asset_name)

            # total cost is determined by the current asset's price * the number purchased for the asset
            total_cost = asset_price_at_t * (action)

            # attempt to buy over budget
            # ignore buy attempt; add penalty of the overage
            if using_budget and self.budget - total_cost < 0:
                logging.info(
                    "Over budget attempt in buy: total cost {} - budget {} = {}".format(
                        total_cost, self.budget, (self.budget - total_cost)
                    )
                )
                margin = 0
                reward = self.budget - total_cost

            elif using_budget and self.budget - total_cost > 0 or not using_budget:

                # adjust the budget to account for the purchase
                if using_budget:
                    self.budget -= total_cost

                # determine initial quantity of holdings for asset
                initial_asset_quantity = len(self.inventory[corresponding_asset_index])

                # add the corresponding quantity to the inventory, by adding the action number of assets
                self.inventory[corresponding_asset_index].extend(
                    [asset_price_at_t] * (action)
                )

                # Determine the number purchased
                after_purchase_asset_quantity = len(
                    self.inventory[corresponding_asset_index]
                )
                num_purchased = after_purchase_asset_quantity - initial_asset_quantity

                # determine $ for a given asset's holdings (total amount currently invested in asset)
                total_for_asset = sum(self.inventory[corresponding_asset_index])

                # penalize the agent for the purchase
                margin = 0
                reward = 0

                logging.info(
                    "Bought {} of {} at {}. Currently Inventory quantity for asset: {}. Total investment in asset: ${}, Current budget: ${}".format(
                        num_purchased,
                        asset_name,
                        asset_price_at_t,
                        after_purchase_asset_quantity,
                        "{:.2f}".format(total_for_asset),
                        "{:.2f}".format(self.budget),
                    )
                )

        # SELL the quantity of the action -> abs(action) used for indexing, as action is negative for sell
        elif action < 0:
            action_str = "SELL"

            # get the coresponding number of assets from the list (we'll need a positive value for the index)
            sell_assets = self.inventory[corresponding_asset_index][0 : abs(action)]

            # sum of prices for assets to sell
            sell_price_sum = sum(sell_assets)

            # current inventory size for the asset
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])

            # change the inventory to reflect the sold assets (sells action number of assets from start of inventory)
            # the inventory for this asset now contains the assets NOT accounted for above (those sold)
            self.inventory[corresponding_asset_index] = self.inventory[
                corresponding_asset_index
            ][abs(action) :]

            # determine current quantity (used for logging)
            final_asset_quantity = len(self.inventory[corresponding_asset_index])

            # determine the quantity sold
            num_sold = initial_asset_quantity - final_asset_quantity

            # margin is current price - bought price. Get the current price from the corresponding asset list
            current_price = self._get_price_for_asset_at_time(asset_name)

            # determine $ for a given asset's holdings (total amount currently invested in this asset)
            total_for_asset = sum(self.inventory[corresponding_asset_index])

            # margin is the profit from selling action number of assets
            margin = (current_price * len(sell_assets)) - sell_price_sum

            # adjust the budget accordingly
            if using_budget:
                self.budget += num_sold * current_price

            # reward is the profit or loss from the sale
            reward = margin

            logging.info(
                "Sold {} of {} for {}. Margin: {}, Current Inventory quantity for asset: {}. Total investment in asset: ${} Current Budget: ${}".format(
                    num_sold,
                    asset_name,
                    str(sell_price_sum),
                    str(margin),
                    len(self.inventory[corresponding_asset_index]),
                    "{:.2f}".format(total_for_asset),
                    "{:.2f}".format(self.budget),
                )
            )

        # SIT
        else:
            action_str = "SIT"

            logging.info(
                "Sat on {}. Current Inventory quantity for asset: {}. Total investment in asset: ${} Current budget: ${}".format(
                    asset_name,
                    len(self.inventory[corresponding_asset_index]),
                    sum(self.inventory[corresponding_asset_index]),
                    self.budget,
                )
            )

            margin = 0
            reward = 0

        logging.info(
            "Action taken = {} Action Type: {} Margin {}".format(
                action, action_str, margin
            )
        )

        return margin, reward

    def step(self, action):
        """
            This is the goal--a quantative reward function. 
            agent action as input. 
            the outputs is: next state of the environment, the reward, 
            whether the episode has terminated, and an info dict to for debugging purposes.
            Iterate through the collection of actions, and calculate the margin/reward for each.
        """

        weights = np.clip(action, 0, 1)

        # divide each weight by the sum of the weights
        weights /= weights.sum() + 0.0000001

        # compensate for when weights are all zeroes
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        assert ((action >= 0) * (action <= 1)).all(), (
            "all action values should be between 0 and 1. Not %s",
            action,
        )

        # throw in the event of unacceptable input
        np.testing.assert_almost_equal(
            np.sum(weights),
            1.0,
            3,
            err_msg='weights should sum to 1. action="%s"' % weights,
        )

        # get individual y values
        y_values = self._get_y_values()

        # w1y1 + w2y2 + w3y3 + w4y4
        weighted_sum = 0

        for w, y in zip(weights, y_values):
            weighted_sum += w * y

        if self.mode == "budget":
            margin, reward = self._balance_portfolio_budget(weights=weights, budget=self.budget)

        elif self.mode == "total":
            margin, reward = self._balance_portfolio_total(weights)

        self.total_profit += margin
        self.t += 1

        # The episode ends when the number of timesteps equals the predefined number of steps.
        done = bool(self.t >= self.steps)

        info = {
            "timestep": self.t,
            "margin": margin,
            "reward": reward,
            "total_profit": "{:.2f}".format(self.total_profit),
            "budget": "{:.2f}".format(self.budget),
        }

        logging.info("Info " + str(info))

        # collect debug info internally within the class
        self.infos.append(info)

        # At end of episode, print total profit made in this episode and save logs to file
        # file is appended to at the end of each episode
        if done:
            logging.info(
                "Total Profit for episode {}".format(
                    self.formatPrice(self.total_profit)
                )
            )
            keys = self.infos[0].keys()

            # records margin, reward, timestep, eg 0,0,1
            with open(self.csv_file, "a", newline="") as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)

        # get state for the following day
        obs = self.get_state()

        return obs, reward, done, info

    def _get_price_for_asset_at_time(self, asset_name, time=None):
        """
            Given an asset's name, return the price
            for that asset at time. Default invocation
            returns the asset's price at the current time
            @param asset_name the asset (str) to retrieve the price for
            @param time the timestamp from which to retrieve the price for the asset
        """
        asset_current_price = (
            self.observations[asset_name][time]
            if time is not None
            else self.observations[asset_name][self.t]
        )
        return asset_current_price

    def get_observations(self, file):
        """
            Read asset values from a single file,
            return list of 'Close' prices.
            Used internally to store prices for
            each asset, from each corresponding file.                
            @param file the assets historical price data
        """
        vec = []
        lines = open(file, "r").read().splitlines()
        # ignore header
        for line in lines[1:]:
            # value in the fourth index is "Close"
            vec.append(float(line.split(",")[4]))
        return vec

    def get_all_observations(self, asset_list, data_dir="./datasets/"):
        """
            Create the observations dict where
            key is asset name and value is
            observations from the corresponding file
            @param asset_list the list of assets in the observations
        """
        asset_dict = (
            collections.OrderedDict()
        )  # safety check to assure assets are in insertion order
        for asset in asset_list:
            data_file = "{}{}.csv".format(data_dir, asset)
            observations = self.get_observations(data_file)
            logging.info(
                "Observations for {} retrieved from {}".format(asset, data_file)
            )
            asset_dict[asset] = observations
        return asset_dict

    def get_state(self):
        """
            Create observation space
        """
        state_arr = list()

        # iterate through assets' prices
        for asset_idx, (asset_name, asset_observations) in enumerate(
            self.observations.items()
        ):

            # construct individual list for each asset
            span_list = list()

            difference = self.t - self.span
            if difference < 0:
                # pad with prices from t0 at the outset
                for i in range(-difference):
                    span_list.append(asset_observations[0])
                for j in range(self.span - (-difference)):
                    span_list.append(asset_observations[j + 1])
            else:
                for i in range((self.t - self.span), self.t):
                    span_list.append(asset_observations[i])

            num_holdings_for_asset = len(self.inventory[asset_idx])

            # get asset's current price
            asset_current_price = self._get_price_for_asset_at_time(asset_name)

            # calculate the total monetary amount from the total holdings for this asset
            current_total_holdings_price = asset_current_price * num_holdings_for_asset

            # add current price * num held in inventory for asset
            span_list.append(current_total_holdings_price)

            # normalize the data prior to appending
            state_arr.append([self._normalize_list(span_list)])

        ret_list = np.array(state_arr)
        return ret_list

    # Function to format the asset prices
    def formatPrice(self, n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

    # Function to return sigmoid
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def _get_data_sets(self, assets_list, data_dir="./datasets/"):
        """                
            Given a list of assets, retrieve and return their datasets.
            Returns a dictionary where asset name is key (eg. AMZN), corresponding csv data
            read via pandas is value. Stored in the class internally.
        """
        asset_dict = {}
        for asset in assets_list:
            # XXXX_test.csv
            data_file = "{}{}.csv".format(data_dir, asset)
            asset_csv_data = pd.read_csv(data_file)
            # remove _test from asset name
            asset_name = asset[0 : asset.index("_")]
            asset_dict[asset_name] = asset_csv_data
        return asset_dict

    def _check_inventory_quantity(self):
        """
            Returns a integer indicating
            how many of the assets are empty in
            the inventory.
        """
        num_empty_assets = 0
        for asset_inventory in self.inventory.values():
            if not asset_inventory:
                num_empty_assets += 1
        return num_empty_assets

    def _normalize_list(self, _list):
        """
            Apply min-max normalization to the
            list. In the event min == max, return 
            [0.0] * len list (adopted from scikitlearn's implementation)
        """
        max_value = max(_list)
        min_value = min(_list)

        if min_value == max_value:
            return [0.0 for i in range(len(_list))]
        else:
            for i in range(len(_list)):
                _list[i] = (_list[i] - min_value) / (max_value - min_value)
            return _list

    def _get_y_values(self):
        """
            returns y1, y2, y3, y4
            where yn = the corresponding asset's price today - 
            the corresponding asset's price yesterday.
        """
        y_values = tuple()
        for asset_name, _ in self.observations.items():
            # yn = asset_price_today / asset_price_yesterday
            y_val = self._get_price_for_asset_at_time(
                asset_name
            ) / self._get_price_for_asset_at_time(asset_name, self.t - 1)
            y_values += (y_val,)
        return y_values

    def _check_observations_length(self):
        """
            Check that the length of each observation
            (retrieved from each data set)
            for each asset is the same.
            Throws if every length is not the same.
        """
        obs_length = len(list(self.observations.values())[0])
        for key_name, obs_list in self.observations.items():
            assert len(obs_list) == obs_length, (
                "Observations are not all the same size " + key_name
            )

    def _determine_purchase_limit(self, asset_name, provided_budget):
        """
            Determine the quantity of the given asset that
            can be purchased while remaining in the intended $ allocation for this asset.
            @param: asset_name the string for the asset
            @param: provided_budget the portion $ of the portfolio allocated for this asset
        """
        current_asset_price = self._get_price_for_asset_at_time(asset_name)
        num_to_purchase = math.floor(provided_budget / current_asset_price)
        return num_to_purchase

    def _initialize_inventory(self):
        """
            Used in c'tor/reset to set the state of the inventory.
        """
        # each key corresponds to the asset in the order provided via the constructor, here GOOG is [0]
        self.inventory = {}

        # construct the inventory: key is index, value is list where purchases are held
        # upon purchase or sell, the items are removed from the list associated with the
        # asset's index
        for i in range(len(self.assets)):
            self.inventory[i] = []

    def _balance_portfolio_budget(self, weights, budget):
        """
            Determine the quantity of each asset to buy/sell
            to meet the target dollar allocation of the assets
            @param weights each index in this list pertains to the weight of the asset
            such that the weight for an index * total budget is the target dollar allocation
            returns a list where each index is the 'action' to perform to achieve the balance
            (pos is buy quantity, neg is sell quantity)
            @param budget the budget to allocate (the current budget, excluding initial allocations)
        """

        # each index corresponds to the allocated budget based on the weights
        target_budget_allocations_each_asset = [w * budget for w in weights]

        # key is the asset_idx (corresponds to budget above) value is the total $ for the asset
        current_portfolio_allocation = {
            asset_idx: sum(asset_list)
            for (asset_idx, asset_list) in self.inventory.items()
        }
        
        logging.info(
            "Current Portfolio $ allocation (pre-balance): {}, Corresponding Assets {}".format(
                current_portfolio_allocation, self.assets
            )
        )

        # each index contains the coresponding buy/sell for each asset
        action_list = list()

        for idx, (target_budget_alloc, current_alloc) in enumerate(
            zip(
                target_budget_allocations_each_asset,
                current_portfolio_allocation.values(),
            )
        ):
            current_asset_price = self._get_price_for_asset_at_time(self.assets[idx])

            # Buy, to increase the budget allocation for this asset
            if target_budget_alloc > current_alloc:
                difference = target_budget_alloc - current_alloc
                num_to_buy = math.floor(difference / current_asset_price)
                action_list.append(num_to_buy)

            # Sell, to decrease budget to target for this asset
            elif target_budget_alloc < current_alloc:
                difference = current_alloc - target_budget_alloc
                num_to_sell = math.floor(difference / current_asset_price)
                action_list.append(-num_to_sell)

            # Same, so sit
            else:
                action_list.append(0)

        # perform corresponding buy/sell for each asset; accumulate rewards
        margin = reward = 0
        
        for idx, action in enumerate(action_list):
            _margin, _reward = self._step(idx, action, True)
            margin += _margin
            reward += _reward

        return margin, reward

    def _balance_portfolio_total(self, weights):
        """
            Given the weights of the portfolio, balance the portfolio, and
            accumulate and return both the margin and
            reward for each corresponding result (buy/sell/sit for the corresponding asset)
            @param weights the list of weights used for balancing the portfolio, where each weight represents
            the portfolio's desired compostion for that asset
        """
        # key is the asset index, value is the length of current inventory for that asset
        portfolio_compositon = {
            asset_idx: len(asset_list)
            for (asset_idx, asset_list) in self.inventory.items()
        }

        # for readable logs
        logging.info(
            "Current Portfolio composition (pre-balance): {}, Corresponding Assets {}".format(
                portfolio_compositon, self.assets
            )
        )

        # each index is the desired quantity for each asset
        portfolio_target_list = [int(round(w * self.TOTAL)) for w in weights]

        # accumulate profit/loss from the buy/sell actions taken to balance
        margin = reward = 0

        for target_idx, (asset_idx, num_held) in enumerate(
            portfolio_compositon.items()
        ):

            # subtract each asset's target quantity from the quantity currently held for this asset
            action_to_take = portfolio_target_list[target_idx] - num_held

            # buy/sell (quantity) or sit for the corresponding asset is performed
            _margin, _reward = self._step(asset_idx, action_to_take)
            margin += _margin
            reward += _reward

        return margin, reward

    def _initial_asset_allocation_budget(self, num_assets):
        """
            Take the original budget, and allocate equally 
            between all assets, while leaving a portion of the budget
            unallocated to allow for purchases
        """
        initial_investment_budget = self.budget / 2
        logging.info("Initializing budget. {} allocated for purchase".format(initial_investment_budget))
        # for simplicity, allocate all evenly
        default_weights = [.25] * num_assets
        self._balance_portfolio_budget(weights=default_weights, budget=initial_investment_budget)