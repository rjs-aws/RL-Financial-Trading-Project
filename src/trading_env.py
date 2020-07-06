"""
A simple financial trading RL algorithm
Jeremy David Curuksu, AWS
2/4/2019

"""
import gym
import gym.spaces
import numpy as np
import pandas as pd
import math
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
from collections import deque
from pprint import pprint
import csv
import h5py

print('Keras backend is: ', keras.backend.backend())

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
                 assets = ['GOOG_test', 'AMZN_test', 'MSFT_test'], # TODO: find a way to invoke the constuctor with an argument list 
                 mode = 'past', 
                 span = 9                
                 ):
             
        
        self.assets = assets
        self.mode = mode
        self.span = span # span is the state space
        self.data = self._get_data_sets(assets_list=assets) # collect data from assets' files
        self.observations = self.getAllObservations(assets) # store the "Close" prices from each corresponding csv into a local member dict
        # TODO: remove
        self.steps = len(self.observations['AMZN_test']) - 1   # Number of time steps in data file
        
        # RNN model if decision based on forecasted horizon instead of lag
        if self.mode == 'future': 
            self.rnn_model = CUR_DIR + asset + '-custom-rnn.h5' # Custom RNN model
        
        elif self.mode == 'future_static':
            self.horizons = DATA_DIR + asset + '_horizons.npy' # Horizons pre-calculated with custom RNN model
        
        # Output filename
        self.csv_file = CSV_DIR
        
        # Action space for each asset contains 3 discrete states: BUY, SELL, SIT.
        NUM_ACTION_STATES = 3
        NUM_ASSETS = len(assets)
        
        # Each requires a corresponding action
        # M x N, where actions (M) is 3, and num_assets (N) is the length of the assets list
        self.action_space = gym.spaces.MultiDiscrete([NUM_ACTION_STATES] * NUM_ASSETS)

        # Observation space is defined from the data min and max
        # Defines the observations the agent receives from the environment, as well as minimum and maximum for continuous observations. 
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (self.span, NUM_ASSETS), dtype = np.float32)
 
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
        self.inventory = []
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

    def step(self, action):
        """
        This is the goal--a quantative reward function. 
        agent action as input. 
        outputs is: next state of the environment, the reward, 
        whether the episode has terminated, and an info dict to for debugging purposes.
        
        Increment time 
        Action is either: buy, sit or sell
        """
            
        # Action.BUY:
        if action == 1:
            # add to the inventory list the price at the index corresponding to the current
            # time stamp
            self.inventory.append(self.observations[self.t])
            margin = 0  # Could be defined as negative of the price of the asset 
            reward = 0
            # print("Buy: " + self.formatPrice(self.observations[self.t]))
        
        # Action.SELL:
        # reward based on profit
        elif action == 2 and len(self.inventory) > 0:
            # check the purchase price for the price at the inventory's first index
            bought_price = self.inventory.pop(0)
            # remove purchase price to calculate reward
            margin = self.observations[self.t] - bought_price
            # penalty does not need to be negative.
            reward = max(margin, 0)
            # print("Sell: " + self.formatPrice(self.observations[self.t]) + " | Profit: " + self.formatPrice(margin))
            
        else: # (Action.SIT)
            margin = 0
            reward = 0
            # print('No action/sitting, (action = ', action, ')')

        self.total_profit += margin
            
        # Increment time
        self.t += 1
        
        # Store state for next day
        obs = self.getState(self.observations, self.t, self.span + 1, self.mode)
            
        # The episode ends when the number of timesteps equals the predefined number of steps.
        # TODO watch how much money? assess profit, how much spent how much risk
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
        
        # collect debug info internally within the class
        self.infos.append(info)
        
        # At end of episode, print total profit made in this episode and save logs to file 
        if done: 
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
        asset_dict = {}
        for asset in asset_list:
            data_file = '{}{}.csv'.format(data_dir, asset)
            observations = self.getObservations(data_file)
            print('Observations for {} retrieved from {}'.format(asset, data_file))
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
            block = observations[d:t + 1] if d >= 0 else -d * [observations[0]] + observations[0:t + 1] # pad with t0
            res = []
            for i in range(lag - 1):
                    res.append(self.sigmoid(block[i + 1] - block[i]))
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
                read via pandas is value.
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