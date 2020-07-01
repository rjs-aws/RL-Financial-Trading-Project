from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.layers import Dense, Conv2d
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.base_parameters import MiddlewareScheme, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.gym_environment import GymVectorEnvironment, ObservationSpaceType
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.memory import MemoryGranularity

# A preset file configures the RL training jobs and defines the hyperparameters for the RL algorithms. 

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(2000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(10000)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

####################
# Clipped PPOAgent #
####################

'''
agent_params = ClippedPPOAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'tanh'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'tanh'
agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999
agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
# agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000) # Default is constant 1.0
agent_params.algorithm.beta_entropy = 0
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 1
agent_params.algorithm.optimization_epochs = 10
agent_params.algorithm.estimate_state_value_using_gae = True

agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(10000)

agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)

'''

#############
# DQN Agent #
#############

agent_params = DQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.00025                                                                                             
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 200000)


###############
# Environment #
###############

env_params = GymVectorEnvironment(level='trading_env:TradingEnv')

########
# Test #
########

preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
