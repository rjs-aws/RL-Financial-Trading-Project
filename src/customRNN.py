


# Implement a RNN enocoder in case the state is defined based on forecasted horizon
def trainRNN(data, lag, horiz, nfeatures):

            # Set hyperparameters
            nunits = 64            # Number of GRUs in recurrent layer
            nepochs = 100          # Number of epochs
            d = 0.2                # Percent of neurons to drop at each epoch
            optimizer = 'adam'     # Optimization algorithm (also tried rmsprop)
            activ = 'elu'          # Activation function for neurons (elu faster than sigmoid)
            activr = 'hard_sigmoid'# Activation function for recurrent layer
            activd = 'linear'      # Dense layer's activation function
            lossmetric = 'mean_absolute_error'  # Loss function for gradient descent
            verbose = True        # Whether or not to list results of each epoch

            # Prepare data
            df = data
            df["Adjclose"] = df.Close # Moving close price to last column
            df.drop(['Date','Close','Adj Close'], 1, inplace=True)
            df = df.diff() 
            #df = df.replace(numpy.nan, 0)
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
            lags = numpy.array(lags)
            horizons = numpy.array(horizons)
            print("Number of horizons: ", len(horizons))
            lags = numpy.reshape(lags, (lags.shape[0], lags.shape[1], nfeatures))
            print('*** DEBUG *** lags:')
            print(lags)

            # Design RNN architecture
            rnn_in = Input(shape = (lag, nfeatures), dtype = 'float32', name = 'rnn_in')
            print('*** DEBUG *** rnn_in:')
            print(rnn_in)
            rnn_gru = GRU(units = nunits, return_sequences = False, activation = activ, recurrent_activation = activr, dropout = d)(rnn_in)
            rnn_out = Dense(horiz, activation = activd, name = 'rnn_out')(rnn_gru)
            model = Model(inputs = [rnn_in], outputs = [rnn_out])
            model.compile(optimizer = optimizer, loss = lossmetric)

            # Train model
            fcst = model.fit({'rnn_in': lags},{'rnn_out': horizons}, epochs=nepochs,verbose=verbose)

            # Save model
            model.save('./debug-dummy.h5')
  
