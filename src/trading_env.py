"""
A simple financial trading RL algorithm
Jeremy David Curuksu, AWS
2/4/2019

"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
import math
import logging
import random
import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.models import load_model, Model
from keras.layers import Dense, Input, GRU, concatenate, Activation
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from config import *
from actions import Action
import collections
from pprint import pprint
import csv
import h5py

# current time string used for the log file
now_str = datetime.datetime.now().strftime("%m%d%y_%H:%M")

logging.basicConfig(
#     filename="{}_run.log".format(now_str), 
    level=logging.DEBUG, 
    format="%(asctime)s:%(levelname)s:%(message)s"
)

    
class TradingEnv(gym.Env):
    """
        A simple environment for financial trading
        with Deep Reinforcement Learning
        
        asset: the name of the asset to trade
        mode: Decision can be made on past (using actual past) or future (using RNN forecast)
        data: datasets directly from the assets' files
        observations: dict where Key: assetname, Value: corresponding list of observations for the asset (observations are 'Close' prices from the dataset)
    """

    def __init__(self,
                 assets = ['GOOG_test', 'AMZN_test', 'MSFT_test', 'AAPL_test'], # Individual assets corresponding to data sources
                 mode = 'past', 
                 span = 9 # number of days lag               
                 ):
        
        NUM_ASSETS = len(assets)
            
        self.csv_file = CSV_DIR # Output filename
        self.assets = assets
        self.mode = mode
        self.span = span 
        
        # total that can be held in the portfolio at a time
        self.TOTAL = 60
        
        self.data = self._get_data_sets(assets_list=assets) # collect data from assets' files
        
        # each key corresponds to the asset, here GOOG is [0] 
        self.inventory = {}
        
        # construct the inventory: key is index, value is list
        # upon purchase or sell, the items are removed from the list associated with the 
        # asset's index
        for i in range(NUM_ASSETS): self.inventory[i] = []

        # store the "Close" prices from each corresponding csv into a local member dict, key is literal asset name:
        # XXX_test, value is list of prices
        self.observations = self.get_all_observations(assets) 

        # TODO: do properly; the number of steps is determined by the length of the test file.
        self.steps = len(self.observations['GOOG_test']) - 1   # Number of time steps in data file
        
        # number of actions in discrete action space (0-20), where (value - 10) is
        # the corresponding action to take for the asset. Positive value is buy n stocks
        # for the given asset, negative value is sell n stocks for the value.
        # eg: value = 20; 20 - 10 = 10; buy 10 stocks for this asset
        # value = 4; 4 - 10 = -6; sell 6 stocks for this asset
          
        
        # Each represents the weight of the portfolio
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(NUM_ASSETS,), dtype=np.float32)
            
        # Observation space is defined from the data min and max
        # Defines the observations the agent receives from the environment, as well as minimum and maximum for continuous observations.
        # (4, span, 1) (4, (span + 1)), (44)
        
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (NUM_ASSETS, 1, self.span + 1), dtype = np.float32)
        
    def reset(self):
        '''
            Initializes a training episode.
            
            t: internal state for timestamp
            total_profit: tracks the accumulated profit
            inventory: key, value pairs str, List of stocks for each asset_name
            infos: debugging info, populated during step
        '''
        
        # Set the state to the initial state--the agent will use this initial state 
        # to take its first action
        self.t = 0 
        self.total_profit = 0
        self.infos = []
        
        
        # Define environment data
        obs = self.get_state()
        
        return obs
    
    def _step(self, corresponding_asset_index, action):
        """
            Invoked once per asset.
            Performs the corresponding action for
            the given asset and calculates the margin
            and reward for that asset.
        """

        # assets are stored as an ordered dict
        # get the corresponding assets from dict
        # The asset index arg corresponds to the index
        # from the asset index list
        asset_index_list = list(self.observations.keys())
        asset_name = asset_index_list[corresponding_asset_index]
        
        logging.info("Asset Index: {} Asset name: {} Recieved Action {} in _step".format(corresponding_asset_index, asset_name, action))
        
        # determine the appropriate action
        margin = None
        
        # record step for logging
        action_str = None
        
        # BUY
        if action > 0:
            action_str = "BUY"
            
            # add to the inventory the corresponding asset at the timestamp
            asset_price_at_t = self._get_current_price_for_asset(asset_name)
            
            # determine initial quantity of holdings for asset
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # add the corresponding quantity to the inventory, by adding the action number of assets
            self.inventory[corresponding_asset_index].extend([asset_price_at_t] * (action))
            
            # total cost is determined by the current asset's price * the number purchased for the asset
            total_cost = asset_price_at_t * (action)
            
            # Determine the number purchased
            after_purchase_asset_quantity = len(self.inventory[corresponding_asset_index])
            num_purchased = after_purchase_asset_quantity - initial_asset_quantity
            
            # determine $ for a given asset's holdings (total amount currently invested in asset)
            total_for_asset = sum(self.inventory[corresponding_asset_index])
            
            # penalize the agent for the purchase
            margin = 0
#             reward = 0
            
            logging.info("Bought {} of {} at {}. Currently Inventory quantity for asset: {}. Total investment in asset: ${}".format(num_purchased, asset_name, asset_price_at_t, after_purchase_asset_quantity, total_for_asset))
        
        # SELL
        elif action < 0:
            action_str = "SELL"
            
            # get the coresponding number of assets from the list (we'll need a positive value for the index)
            sell_assets = self.inventory[corresponding_asset_index][0:abs(action)]
            
            # sum of prices for assets to sell
            sell_price_sum = sum(sell_assets)
            
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # change the inventory to reflect the sold assets (sells action number of assets from start of inventory)
            # the inventory for this asset now contains the assets NOT accounted for above (those sold)
            self.inventory[corresponding_asset_index] = self.inventory[corresponding_asset_index][abs(action):]
            
            # determine current quantity (used for logging)
            final_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # determine the quantity sold
            num_sold = initial_asset_quantity - final_asset_quantity
            
            # margin is current price - bought price. Get the current price from the corresponding asset list
            current_price = self._get_current_price_for_asset(asset_name)
            
            # determine $ for a given asset's holdings (total amount currently invested in asset)
            total_for_asset = sum(self.inventory[corresponding_asset_index])
            
            # margin is the profit from selling action number of assets
            margin = (current_price * len(sell_assets)) - sell_price_sum
            
#             reward = max(margin, 0)
            
            logging.info("Sold {} of {} for {}. Margin: {}, Current Inventory quantity for asset: {}. Total investment in asset: ${}".format(num_sold, asset_name, str(sell_price_sum), str(margin), len(self.inventory[corresponding_asset_index]), total_for_asset))
        
        # SIT
        else:
            action_str = "SIT"
            
            logging.info("Sat on {}. Current Inventory quantity for asset: {}. Total investment in asset: ${}".format(asset_name, len(self.inventory[corresponding_asset_index]), sum(self.inventory[corresponding_asset_index])))            
            
            margin = 0
#             reward = 0

        
        logging.info("Action taken = {} Action Type: {} Margin {}".format(action, action_str, margin))
        
        return margin

    
    def step(self, action):
        """
            This is the goal--a quantative reward function. 
            agent action as input. 
            the outputs is: next state of the environment, the reward, 
            whether the episode has terminated, and an info dict to for debugging purposes.
            Iterate through the collection of actions, and calculate the margin/reward for each.
        """
        
        # normalize weights between 1, 0
        weights = np.clip(action, 0, 1)
        
        # divide each weight by the sum of the weights
        weights /= (weights.sum() + 0.0000001)
        
        # compensate for when weights are all zeroes
        weights[0] += np.clip(1 - weights.sum(), 0, 1) 

        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)
        
        # get individual y values
        y_values = self._get_y_values()
        
        # w1y1 + w2y2 + w3y3 + w4y4
        weighted_sum = 0
        for w,y in zip(weights, y_values):
            weighted_sum += w * y
        
        margin = self._balance_portfolio(weights)
        
        self.total_profit += margin
        
        reward = margin
            
        # Increment time
        self.t += 1
         
        # The episode ends when the number of timesteps equals the predefined number of steps.
        if self.t >= self.steps:
            done = True 
        else:
            done = False
            
        
        info = {
            'timestep': self.t,
            'margin': margin,
            'reward': reward,
            'total_profit': self.total_profit
        }
        
        logging.info("Info " + str(info))
        
        # collect debug info internally within the class
        self.infos.append(info)
        
        # At end of episode, print total profit made in this episode and save logs to file
        # file is appended to at the end of each episode
        if done: 
            logging.info("Total Profit for episode {}".format(self.total_profit))
            print("Total Profit for episode: " + self.formatPrice(self.total_profit))
            keys = self.infos[0].keys()
            
            # records margin, reward, timestep, eg 0,0,1
            with open(self.csv_file, 'a', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        
        # get state for the following day
        obs = self.get_state()
                
        return obs, reward, done, info 
   

    def _balance_portfolio(self, weights):
        """
            Balance the portfolio based on the 
            weights argument
        """
        # key is the asset index, value is the length of current inventory for that asset
        portfolio_compositon = { asset_idx: len(asset_list) for (asset_idx, asset_list ) in self.inventory.items() }
        
        # for ease of reading logs
        logging.info("Current Portfolio composition: {}, Corresponding Assets {}".format(portfolio_compositon, self.assets))

        
        # given the weights determine how many of each asset should be held
        # each index holds how many of each asset should be present to meet the 
        # targets of the weights
        portfolio_target_list = [ int(round(w * self.TOTAL)) for w in weights ]
        
        # accumulate profit/loss from the buy/sell actions taken to balance
        margin = 0
        
        for target_idx, (asset_idx, num_held) in enumerate(portfolio_compositon.items()):
            action_to_take = portfolio_target_list[target_idx] - num_held
            
            # action for the corresponding asset is perfomed
            margin += self._step(asset_idx, action_to_take)
            
        return margin
 

    def _get_current_price_for_asset(self, asset_name):
        """
            Given an asset's name, return the current
            price for that asset (at self.t)
        """
        asset_current_price = self.observations[asset_name][self.t]
        return asset_current_price
            
                  
    def get_observations(self, file):
            """
                Read asset values from a single file,
                return list of 'Close' prices.
                Used internally to store prices for
                each asset, from each corresponding file.
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
            Create an observations dict where
            key is asset name and value is
            observations from the corresponding file
        """
        asset_dict = collections.OrderedDict() # maintain assets in insertion order
        for asset in asset_list:
            data_file = '{}{}.csv'.format(data_dir, asset)
            observations = self.get_observations(data_file)
            logging.info('Observations for {} retrieved from {}'.format(asset, data_file))
            asset_dict[asset] = observations
        return asset_dict
  

    def get_state(self):
        """
            Create observation space
        """
        state_arr = list() 
        # iterate through assets' prices
        asset_idx = 0
        
        for asset_name, asset_observations in self.observations.items():
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
            try:
                asset_current_price = self._get_current_price_for_asset(asset_name)
            except:
                asset_current_price = self.observations[asset_name][len(self.observations[asset_name]) - 1]

            
            # calculate the total monetary amount based on the inventory for this asset
            current_total_holdings_price = asset_current_price * num_holdings_for_asset
                     
            # add current price * num held in inventory for asset
            span_list.append(current_total_holdings_price)
            
            asset_idx += 1
            
            # normalize the data prior to appending
            state_arr.append([self._normalize_list(span_list)])
           
        ret_list = np.array(state_arr)
        return ret_list
            
        
        
    # Function to format the asset prices
    def formatPrice(self, n):
            return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

        
    # Function to return sigmoid
    def sigmoid(self, x):
            return 1 / (1 + math.exp(-x))
        
    
    def _get_data_sets(self, assets_list, data_dir='./datasets/'):
            """
                Given a list of assets, retrieve and return their datasets.
                Returns a dictionary where asset name is key (eg. AMZN), corresponding csv data
                read via pandas is value. Stored in the class internally.
            """
            asset_dict = {}
            for asset in assets_list:
                # XXXX_test.csv
                data_file = '{}{}.csv'.format(data_dir, asset)
                asset_csv_data = pd.read_csv(data_file)
                # remove _test from asset name
                asset_name = asset[0: asset.index("_")]
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
            [0.0] * len list
        """
        max_value = max(_list)
        min_value = min(_list)
        if min_value == max_value:
            # see: scikitlearn's implementation
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
        y_values = list()
        for asset_name, asset_observations in self.observations.items():
            # yn = asset_price_today / asset_price_yesterday
            y_val = self._get_current_price_for_asset(asset_name) / self.observations[asset_name][self.t - 1]
            y_values.append(y_val)
        y_tup = tuple(y_values)
        return y_tup[0], y_tup[1], y_tup[2], y_tup[3]
            