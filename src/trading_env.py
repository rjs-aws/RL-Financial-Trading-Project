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

now_str = datetime.datetime.now().strftime("%m%d%y_%H:%M")
logging.basicConfig(
    filename="{}_run.log".format(now_str), 
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

    # The size of the state space--here, the number of previous time steps or forecasted horizon used for decision making
    def __init__(self,
                 assets = ['GOOG_test', 'AMZN_test', 'MSFT_test', 'AAPL_test'], # TODO: find a way to invoke the constuctor with an argument list 
                 mode = 'past', 
                 span = 9               
                 ):
        
        NUM_ASSETS = len(assets)
            
        self.csv_file = CSV_DIR # Output filename
        self.assets = assets
        self.mode = mode
        self.span = span # number of days in the past/future
        self.data = self._get_data_sets(assets_list=assets) # collect data from assets' files
        self.inventory = {} # each key corresponds to the asset, here GOOG is [0] 
        
        # construct the inventory
        for i in range(NUM_ASSETS): self.inventory[i] = []

        # store the "Close" prices from each corresponding csv into a local member dict, key is literal asset name:
        # XXX_test, value is list of prices
        self.observations = self.getAllObservations(assets) 

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
        
      
        # Results in a discrete space of num_assets.
        # Action contains an index into the list of tuples where (asset_name, action)
        # should be 4 dim
        # self.action_space = spaces.Box(low=0, high=21, shape=(1,), dtype=np.float32) 

        # 21 actions for each asset
        TOTAL_ACTIONS = 21 * NUM_ASSETS
        self.action_space = gym.spaces.Discrete(TOTAL_ACTIONS)
            
        # Observation space is defined from the data min and max
        # Defines the observations the agent receives from the environment, as well as minimum and maximum for continuous observations.
        # (4, span, 1) (4, (span + 1)), (44)
        
        # currently doesnt represent the portfolio holdings.
        total = NUM_ASSETS * (self.span)
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (total,), dtype = np.float32)
        
    # Initializes a training episode.
    def reset(self):
        '''
            t: internal state for timestamp
            total_profit: tracks the accumulated profit
            inventory: 
            infos: 
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
        obs = self.getState(self.observations, 0, self.span + 1, self.mode)
        
        return obs
    
    def _step(self, corresponding_asset_index, action):
        NUM_ACTIONS = 21
        # assets are stored as an ordered dict
        # get the corresponding assets from dict
        asset_index_list = list(self.observations.keys())
        asset_name = asset_index_list[corresponding_asset_index]
        action = action - (corresponding_asset_index * NUM_ACTIONS)
        margin = reward = None
        
        if action - 10 > 0:
            # add to the inventory the corresponding asset at the timestamp
            asset_at_t = self.observations[asset_name][self.t]
            # self.inventory[corresponding_asset_index].append(self.observations[asset_name][self.t])
            initial_asset_quantity = len(self.inventory[corresponding_asset_index])
            # add the corresponding quantity to the inventory, by adding the action number of assets
            self.inventory[corresponding_asset_index].extend([asset_at_t] * (action - 10))
            # Determine the number purchased
            after_purchase_asset_quantity = len(self.inventory[corresponding_asset_index])
            num_purchased = after_purchase_asset_quantity - initial_asset_quantity
            margin = 0
            reward = 0
            logging.info("Bought {} of {} at {}. Currently have {}".format(num_purchased, asset_name, self.observations[asset_name][self.t], after_purchase_asset_quantity))
        
        elif action - 10 < 0 and len(self.inventory[corresponding_asset_index]) > 0:
            # get the coresponding number of assets from the list
            bought_price = self.inventory[corresponding_asset_index].pop(0)
            # remove purchase price to calculate reward
            margin = self.observations[asset_name][self.t] - bought_price
            reward = max(margin, 0)
            logging.info("Sold {} at {}. Margin: {}, Currently have {}".format(asset_name, str(bought_price), str(margin), len(self.inventory[corresponding_asset_index])))
        
        else:
            logging.info("Sat")
            margin = reward = 0
        
        return margin, reward

    
    def step(self, action):
        """
        This is the goal--a quantative reward function. 
        agent action as input. 
        the outputs is: next state of the environment, the reward, 
        whether the episode has terminated, and an info dict to for debugging purposes.
        """
        logging.info("Action {}".format(action))
        
            
        if action >= 0 and action <= 21:
            margin, reward = self._step(corresponding_asset_index=0, action=action)
            
        elif action > 21 and action <= 42:
            margin, reward = self._step(corresponding_asset_index=1, action=action)
        
        elif action > 42 and action <= 63:
            margin, reward = self._step(corresponding_asset_index=2, action=action)
            
        else:
            margin, reward = self._step(corresponding_asset_index=3, action=action)


        self.total_profit += margin
            
        # Increment time
        self.t += 1
        
        # Store state for next day
        obs = self.getState(self.observations, self.t, self.span + 1, self.mode)
        
            
        # The episode ends when the number of timesteps equals the predefined number of steps.
        if self.t >= self.steps:
            done = True 
        else:
            done = False
            
        # calculate basic info
        info = {
            'timestep': self.t,
            'margin': margin,
            'reward': reward
        }
        
        logging.info("Info " + str(info))
        
        # collect debug info internally within the class
        self.infos.append(info)
        
        # At end of episode, print total profit made in this episode and save logs to file 
        if done: 
            logging.info("Total Profit {}".format(self.total_profit))
            print("Total Profit: " + self.formatPrice(self.total_profit))
            keys = self.infos[0].keys()
            with open(self.csv_file, 'a', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)

        return obs, reward, done, info 


                
    # Function to read the asset price time series.
    # Collects the "Close" values and returns them as a list
    def getObservations(self, file):
            vec = []
            lines = open(file, "r").read().splitlines()
            # ignore header
            for line in lines[1:]:
                    # value in the fourth index is "Close"
                    vec.append(float(line.split(",")[4]))
            return vec
        
    
    def getAllObservations(self, asset_list, data_dir="./datasets/"):
        """
            Create an observations dict where
            key is asset name and value is
            observations from the corresponding file
        """
        asset_dict = collections.OrderedDict() # maintain assets in insertion order
        for asset in asset_list:
            data_file = '{}{}.csv'.format(data_dir, asset)
            observations = self.getObservations(data_file)
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
  
    # Function to define a state at time t:
    def getState(self, observations, t, lag, mode):

        # Mode "past": State at time t defined as n-day lag in the past
        if mode == 'past':
            
            d = t - lag + 1
            res = []
            for asset, asset_observations in observations.items():             
                block = asset_observations[d:t + 1] if d >= 0 else -d * [asset_observations[0]] + asset_observations[0:t + 1] # pad with t0
                for i in range(lag - 1):
                        res.append(self.sigmoid(block[i + 1] - block[i]))
                # len must be 40, same as total above, in a single dimension
            return np.array([res])

        # Mode "future": State at time t defined as the predicted n-day horizon in the future using RNN
        elif mode == 'future':
            rnn = load_model(self.rnn_model)
            d = t - lag + 1
            block = observations[d:t + 1] if d >= 0 else -d * [observations[0]] + observations[0:t + 1] # pad with t0
            horiz = rnn.predict(block)
            res = []
            for i in range(lag - 1): # WARNING: Assume size of lag = size of horizon
                    res.append(self.sigmoid(horiz[i + 1] - horiz[i]))
            return np.array([res])
        
        # Mode "future_static": State at time t defined as the predicted n-day horizon in the future from pre-computed forecasts
        elif mode == 'future_static': 
            horiz = np.load(self.horizons)
            #block = [observations[t]] + horiz[t] # Add obs for current t to forecasted horizon (1 + 10 = 11 pts => 10 intervals)
            block = horiz[t]
            res = []
            for i in range(lag - 1):
                    # creates forecast
                    res.append(self.sigmoid(block[i + 1] - block[i]))
            return np.array([res])            
        

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