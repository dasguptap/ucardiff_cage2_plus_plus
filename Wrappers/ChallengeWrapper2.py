#from gym import Env
#PDG: Added next line 20251106 for compatibility with gymnasium in cyborg_plus_plus
from gymnasium import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, RedTableWrapper, EnumActionWrapper

# corrected BlueTableWrapper
from .BlueTableWrapper import BlueTableWrapper


class ChallengeWrapper2(Env, BaseWrapper):
    def __init__(self, agent_name: str, env, agent=None,
                 reward_threshold=None, max_steps=None):
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        env = table_wrapper(env, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self, action=None):
        #PDG: Added next line 20251030 for compatibility with gymnasium in cyborg_plus_plus
        obs, reward, done, truncated, info = self.env.step(action=action)

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            #done = True
            #PDG: Added next 2 lines 20251106 for compatibility with gymnasium in cyborg_plus_plus
            terminated = True
            truncated = True

        return obs, reward, done, truncated, info #PDG: 20251106 compaibility with gymnasium in cyborg_plus_plus

    def reset(self, **kwargs):
        self.step_counter = 0
        #return self.env.reset()
        return self.env.reset(**kwargs) #PDG: 2051106 compaibility with gymnasium in cyborg_plus_plus

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)



