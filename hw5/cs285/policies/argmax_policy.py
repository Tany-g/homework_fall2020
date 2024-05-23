import numpy as np
import pdb


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) <= 3:
            obs = obs[None]

        # return the action that maxinmizes the Q-value
        # at the current observation as the output
        q = self.critic.qa_values(obs)
        acs = q.argmax(1)

        return acs.squeeze()

    ####################################
    ####################################