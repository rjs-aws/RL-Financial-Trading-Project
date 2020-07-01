Invoking the main script involves passing command line arguments:
The second is the local location of the output, the third is the mode (train or evaluate)

```bash
python run_rl_agent.py <your-output-dir> train > run.log &
python run_rl_agent.py <your-output-dir> evaluate <your-other-outdir> > run_rnnbase_test.log &
```

The main script's role is to start the process of either training or evaluating the model's process, determined by the arguments provided.

In training mode, the training takes place from scratch. 
The entry file ```train-coach.py``` is provided to the estimator object as the entryfile. ```train-coach``` returns the ```preset-trading``` file, 
responsible for the configuration of the RL training job and sets the hyperparameters for the RL algorithms. 

Preset trading also sets the GymVectorEnvironment as the ```trading_env```'s ```TradingEnv``` object. 

**Workflow**: Estimator defined in ```run_rl_agent.py``` calls preset ```preset-trading.py``` which calls custom env ```trading_env.py```.

```trading_env``` configures the OpenAI gym environment. 


OpenAIGym is a toolkit for developing and comparing RL algorithms. It mimics a real-world scenario: given the current state of the 
environment and an action taken by the agent or agents, the simulator processes the impact of the action, and returns the next state and a reward.

Open AI Gym requires the following:
Every environment comes with an action_space and an observation_space. 
These attributes are of type Space, and they describe the format of valid actions and observations.

- ```env.action_space``` — Defines the actions the agent can take, specifies whether each action is continuous or discrete, and specifies the minimum and maximum if the action is continuous.

- ```env.observation_space``` — Defines the observations the agent receives from the environment, as well as minimum and maximum for continuous observations.

- ```env.reset()``` — Initializes a training episode. The reset() function returns the initial state of the environment, and the agent uses the initial state to take its first action. The action is then sent to the step() repeatedly until the episode reaches a terminal state. When step() returns done = True, the episode ends. The RL toolkit re-initializes the environment by calling reset().

- ```step()``` — Takes the agent action as input and outputs the next state of the environment, the reward, whether the episode has terminated, and an info dictionary to communicate debugging information. It is the responsibility of the environment to validate the inputs.


Here, the action spaces contains 3 discrete states: BUY, SELL, SIT: 
```self.action_space = gym.spaces.Discrete(3)```

The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers. 

Here, the observation space contains:
```self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (self.span,), dtype = np.float32)```

Invocation of the ```reset``` function reinitializes member variables to 0/empty (if list), and invokes the ```getState``` method to set
the state for time ```t```.