from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools
from cs285.infrastructure.utils import normalize, unnormalize

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
#         self.action_mean, self.action_std = np.mean(self.action_range), np.std(action_range)
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term (DONE)
#         return entropy
        return torch.exp(self.log_alpha) 

    def get_action(self, obs: np.ndarray, sample=True): #-> np.ndarray:
        # TODO: return sample from distribution if sampling (DONE)
        # if not sampling return the mean of the distribution
#         return action
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        distribution = self(ptu.from_numpy(observation))
        if sample:
            normalized_action = distribution.sample()
        else:
            normalized_action = distribution.mean
        action = normalized_action
        return ptu.to_numpy(action)
        
    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing (DONE)

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        if self.discrete:
            raise NotImplementedError('Not implemented for discrete action space')
#             logits = self.logits_na(observation)
#             action_distribution = distributions.Categorical(logits=logits)
#             return action_distribution
        else:
            batch_mean = self.mean_net(observation)
#             scale_tril = torch.diag(torch.exp(torch.clip(self.logstd,*self.log_std_bounds)))
            scale_tril = torch.exp(torch.clip(self.logstd,*self.log_std_bounds))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1) #, 1)
            action_distribution = sac_utils.SquashedNormal(
                batch_mean,
                batch_scale_tril,
            )
            self.dist = action_distribution
            return action_distribution

    def update(self, ob_no, critic):
        # TODO Update actor network and entropy regularizer (DONE)
        # return losses and alpha value
        ob_no = ptu.from_numpy(ob_no)
        
        distribution = self(ob_no)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(1)
        
        q_n = torch.min(*critic(ob_no, action))
        actor_loss = (self.alpha.detach()*log_prob - q_n).sum()
        alpha_loss = (-self.alpha*(log_prob.detach() + self.target_entropy)).sum()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha
    
#     def normalize(self, action):
#         return normalize(action,self.action_mean,self.action_std)
    
#     def unnormalize(self, action):
#         return unnormalize(action,self.action_mean,self.action_std)
       