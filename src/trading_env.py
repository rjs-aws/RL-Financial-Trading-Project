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

# logging.info('Keras backend is: ', str(keras.backend.backend()))

    
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
        self.steps = len(self.observations['AMZN_test']) - 1   # Number of time steps in data file
        
        # RNN model if decision based on forecasted horizon instead of lag
        if self.mode == 'future': 
            self.rnn_model = CUR_DIR + asset + '-custom-rnn.h5' # Custom RNN model
        
        elif self.mode == 'future_static':
            self.horizons = DATA_DIR + asset + '_horizons.npy' # Horizons pre-calculated with custom RNN model
        
        # number of actions in discrete action space (0-20), where (value - 10) is
        # the corresponding action to take for the asset. Positive value is buy n stocks
        # for the given asset, negative value is sell n stocks for the value.
        # eg: value = 20; 20 - 10 = 10; buy 10 stocks for this asset
        # value = 4; 4 - 10 = -6; sell 6 stocks for this asset
          
        
        # 21 actions for each asset
        self.action_space = gym.spaces.Box(low=0, high=21, shape=(NUM_ASSETS,), dtype=np.uint8)
            
        # Observation space is defined from the data min and max
        # Defines the observations the agent receives from the environment, as well as minimum and maximum for continuous observations.
        # (4, span, 1) (4, (span + 1)), (44)
        
        # currently doesnt represent the portfolio holdings.
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (NUM_ASSETS, self.span, 1), dtype = np.float32)
        
        
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
        
        # If using 'forecasts' to take decisions, train a custom RNN forecasting model
        ''' Training of RNN done separately, would need reconcile Keras mxnet/tf backend to do it here        
        if self.mode == 'future':
            horiz = self.span  # Define size of predicted horizon (e.g. set same as span)
            nfeatures = 1      # Set nfeatures = 2 (TBD) to use both close and volume as co-factors to make predictions
            self.trainRNN(self.data, self.span, horiz, nfeatures)
        '''
        
        # Define environment data
        obs = self.get_state(self.observations, 0, self.span + 1)
        
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
        margin = reward = None
        
        # record step for logging
        action_str = None
        
        # BUY
        if action - 10 > 0:
            action_str = "BUY"
            
            # add to the inventory the corresponding asset at the timestamp
            asset_price_at_t = self.observations[asset_name][self.t]
            
            # determine initial quantity of holdings for asset
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # add the corresponding quantity to the inventory, by adding the action number of assets
            self.inventory[corresponding_asset_index].extend([asset_price_at_t] * (action - 10))
            
            # total cost is determined by the current asset's price * the number purchased for the asset
            total_cost = asset_price_at_t * (action - 10)
            
            # Determine the number purchased
            after_purchase_asset_quantity = len(self.inventory[corresponding_asset_index])
            num_purchased = after_purchase_asset_quantity - initial_asset_quantity
            
            # determine $ for a given asset's holdings (total amount currently invested in asset)
            total_for_asset = sum(self.inventory[corresponding_asset_index])
            
            # penalize the agent for the purchase
            margin = 0
            reward = -total_cost
            
            logging.info("Bought {} of {} at {}. Currently Inventory quantity for asset: {}. Total investment in asset: ${}".format(num_purchased, asset_name, asset_price_at_t, after_purchase_asset_quantity, total_for_asset))
        
        # SELL
        elif action - 10 < 0:
            action_str = "SELL"
            
            # get the coresponding number of assets from the list (we'll need a positive value for the index)
            sell_assets = self.inventory[corresponding_asset_index][0:abs(action - 10)]
            
            # sum of prices for assets to sell
            sell_price_sum = sum(sell_assets)
            
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # change the inventory to reflect the sold assets (sells action number of assets from start of inventory)
            # the inventory for this asset now contains the assets NOT accounted for above (those sold)
            self.inventory[corresponding_asset_index] = self.inventory[corresponding_asset_index][abs(action - 10):]
            
            # determine current quantity (used for logging)
            final_asset_quantity = len(self.inventory[corresponding_asset_index])
            
            # determine the quantity sold
            num_sold = initial_asset_quantity - final_asset_quantity
            
            # margin is current price - bought price. Get the current price from the corresponding asset list
            current_price = self.observations[asset_name][self.t]
            
            # determine $ for a given asset's holdings (total amount currently invested in asset)
            total_for_asset = sum(self.inventory[corresponding_asset_index])
            
            # margin is the profit from selling action number of assets
            margin = (current_price * len(sell_assets)) - sell_price_sum
            reward = (current_price * len(sell_assets))
            
            logging.info("Sold {} of {} for {}. Margin: {}, Current Inventory quantity for asset: {}. Total investment in asset: ${}".format(num_sold, asset_name, str(sell_price_sum), str(margin), len(self.inventory[corresponding_asset_index]), total_for_asset))
        
        # SIT
        else:
            action_str = "SIT"
            
            logging.info("Sat on {}. Current Inventory quantity for asset: {}. Total investment in asset: ${}".format(asset_name, len(self.inventory[corresponding_asset_index]), sum(self.inventory[corresponding_asset_index])))            
            
            margin = reward = 0

        
        logging.info("Action {} - 10 taken = {} Action Type: {} Reward {}".format(action, (action - 10), action_str, reward))
        
        return margin, reward

    
    def step(self, action):
        """
            This is the goal--a quantative reward function. 
            agent action as input. 
            the outputs is: next state of the environment, the reward, 
            whether the episode has terminated, and an info dict to for debugging purposes.
            Iterate through the collection of actions, and calculate the margin/reward for each.
        """
        
        _action = action.astype(int)
        
        # Reminder: high valued actions correspond to purchases; low values correspond to sell
        # Ex: 4 -> sell 6; 17 -> buy 7
        logging.info("Actions recieved {}".format(_action))
        
        margin = reward = 0
            
        # iterate through the actions, and accumulate the margin & reward for each action
        for i in range(len(_action)):
            _margin, _reward = self._step(corresponding_asset_index=i, action=_action[i])
            margin += _margin
            reward += _reward

        self.total_profit += margin
            
        # Increment time
        self.t += 1
         
        # The episode ends when the number of timesteps equals the predefined number of steps.
        if self.t >= self.steps:
            done = True 
        else:
            done = False
            
        # calculate basic info
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
            logging.info("Total Profit {}".format(self.total_profit))
            print("Total Profit: " + self.formatPrice(self.total_profit))
            keys = self.infos[0].keys()
            
            # records margin, reward, timestep, eg 0,0,1
            with open(self.csv_file, 'a', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        
        # get state for the following day
        obs = self.get_state(self.observations, self.t, self.span + 1)
                
        return obs, reward, done, info 


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


    # Implement a RNN enocoder in case the state is defined based on forecasted horizon
    def trainRNN(self, data, lag, horiz, nfeatures):

            # Set hyperparameters
            nunits = 64            # Number of GRUs in recurrent layer
            nepochs = 10           # Number of epochs
            d = 0.2                # Percent of neurons to drop at each epoch
            optimizer = 'adam'     # Optimization algorithm (also tried rmsprop)
            activ = 'elu'          # Activation function for neurons (elu faster than sigmoid)
            activr = 'hard_sigmoid'# Activation function for recurrent layer
            activd = 'linear'      # Dense layer's activation function
            lossmetric = 'mean_absolute_error'  # Loss function for gradient descent
            verbose = False        # Whether or not to list results of each epoch

            # Prepare data
            df = data
            df["Adjclose"] = df.Close # Moving close price to last column
            df.drop(['Date','Close','Adj Close'], 1, inplace=True)
            df = df.diff() 
            #df = df.replace(np.nan, 0)
            #scaler = preprocessing.StandardScaler()
            #for feat in df.columns:
            #    df[feat] = scaler.fit_transform(df.eval(feat).values.reshape(-1,1))

            data = df.as_matrix()
            lags = []
            horizons = []
            nsample = len(data) - lag - horiz  # Number of time series (Number of sample in 3D)
            
            for i in range(nsample):
                    lags.append(data[i: i + lag , -nfeatures:])
                    horizons.append(data[i + lag : i + lag + horiz, -1])
            
            lags = np.array(lags)
            horizons = np.array(horizons)
            print("Number of horizons: ", len(horizons))
            lags = np.reshape(lags, (lags.shape[0], lags.shape[1], nfeatures))

            # Design RNN architecture
            rnn_in = Input(shape = (lag, nfeatures), dtype = 'float32', name = 'rnn_in')
            rnn_gru = GRU(units = nunits, return_sequences = False, activation = activ, recurrent_activation = activr, dropout = d)(rnn_in)
            rnn_out = Dense(horiz, activation = activd, name = 'rnn_out')(rnn_gru)
            model = Model(inputs = [rnn_in], outputs = [rnn_out])
            model.compile(optimizer = optimizer, loss = lossmetric)

            # Train model
            fcst = model.fit({'rnn_in': lags},{'rnn_out': horizons}, epochs=nepochs,verbose=verbose)

            # Save model
            model.save(self.rnn_model)
  

    def get_state(self, observations, t, lag):
        """
            Create observation space
        """
        state_arr = list()
        d = t - lag + 1
        for _, asset_observations in observations.items():
            span_list = list()
            block = asset_observations[d:t + 1] if d >= 0 else -d * [asset_observations[0]] + asset_observations[0:t + 1] 
            # span days
            for i in range(lag - 1):
                span_list.append([ self.sigmoid(block[i + 1] - block[i]) ])
            state_arr.append(span_list)
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
            