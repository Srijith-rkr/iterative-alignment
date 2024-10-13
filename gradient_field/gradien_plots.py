# Uses DPO environment
import torch 
import torch.nn.functional as F
import math
import time 
import numpy as np

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors to device first and then set requires_grad=True
def pytorch_implementation(w,l,r,b,type, time_ = False):
    # SRIJITH - The varaiabls names are misleading, w is policy_winner / ref_winner
    start = time.time()
    winner_log_likelihood = torch.tensor(w, device = device, requires_grad=True)
    loser_log_likelihood = torch.tensor(l, device = device, requires_grad=True)
    reference_log_likelihood = torch.tensor(r, device = device, requires_grad=True)
    beta = torch.tensor(b, device = device, requires_grad=True)
    # Below is dpo loss
    if type == 'mine':
        loss =    - torch.log(  (winner_log_likelihood / reference_log_likelihood)**beta  /  (loser_log_likelihood / reference_log_likelihood)**beta   )
    if type == 'dpo':
        loss =    - torch.log(torch.sigmoid(   beta * torch.log(winner_log_likelihood / reference_log_likelihood) - beta * torch.log(loser_log_likelihood / reference_log_likelihood)    ))
    # Below is Self Play Loss
    if type == 'spin_deravative_of_log':
        loss = -F.logsigmoid(beta * (winner_log_likelihood - loser_log_likelihood))
    if type == 'spin_deravative_of_ratio':
        loss = -F.logsigmoid(beta * (torch.log(winner_log_likelihood) - torch.log(loser_log_likelihood)))
    loss.backward()
    # print(winner_log_likelihood.grad)
    # print(loser_log_likelihood.grad)
    return winner_log_likelihood.grad, loser_log_likelihood.grad
    end = time.time()
    if time_:
        print(f"Time taken: {end - start:.6f} seconds")

# Loss implementation from SPIN
# pi_logratios = policy_real_logps - policy_generated_logps
# ref_logratios = opponent_real_logps - opponent_generated_logps

# if reference_free:
#     ref_logratios = 0

# logits = pi_logratios - ref_logratios

# if self.loss_type == "sigmoid": # we use this
#     losses = -F.logsigmoid(self.beta * logits)
# elif self.loss_type == "hinge":
#     losses = torch.relu(1 - self.beta * logits)
# else:
#     raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

# real_rewards = self.beta * (policy_real_logps - opponent_real_logps).detach()
# generated_rewards = self.beta * (policy_generated_logps - opponent_generated_logps).detach()



def math_implementation(w,l,r,b, time_ = False):
    start = time.time()

    wp = - b * math.pow(l,b)  /  (w * (math.pow(w,b)+ math.pow(l,b)) )
    try:
        lp = b * math.pow(l,b-1)  /  (math.pow(w,b)+ math.pow(l,b)) 
    except:
        lp = 0
    end = time.time()
    if time_:
        print(f"Time taken: {end - start:.6f} seconds")
    return wp, lp
        
# def numdiff_implementation(w,l,r,b, time_ = False):
#     start = time.time()
#     epsilon = 1e-4
#     wp = (math.log(w+epsilon) - math.log(w)) / epsilon
#     lp = (math.log(l+epsilon) - math.log(l)) / epsilon
#     end = time.time()
#     if time_:
#         print(f"Time taken: {end - start:.6f} seconds")
#     return wp, lp
# plt.figure()
# for beta in [1.] :#, 0.5, 1., 2., 5., 10.]:
#     winner_scale = np.arange(1,11)/10
#     loser_scale = np.arange(1,11)/10

#     winner_grid = np.zeros((10,10))
#     loser_grid = np.zeros((10,10))
#     for i in range(winner_grid.shape[0]):
#         for j in range(winner_grid.shape[1]):
#             winner_grid[i,j], loser_grid[i,j] = pytorch_implementation(winner_scale[i],loser_scale[j],1.,beta,'dpo')
#             # SRIJITH : Since we do gradient descent, we multiply by -1
#             winner_grid[i,j] = winner_grid[i,j] * -1
#             loser_grid[i,j] = loser_grid[i,j] * -1
        
#     plt.quiver(winner_scale,loser_scale, winner_grid, loser_grid, angles='xy',pivot='tail',headwidth=5,headlength=2,width= 0.003,color='r', label='Beta = 1')#, scale_units='xy', scale=1)


winner_scale = np.arange(1,31,3)/10
loser_scale = np.arange(1,31,3)/10

winner_grid = np.zeros((10,10))
loser_grid = np.zeros((10,10))
for i in range(winner_grid.shape[0]):
    for j in range(winner_grid.shape[1]):
        winner_grid[i,j], loser_grid[i,j] = pytorch_implementation(winner_scale[i],loser_scale[j],1.,0.1,'spin_deravative_of_log')
        # SRIJITH : Since we do gradient descent, we multiply by -1
        winner_grid[i,j] = winner_grid[i,j] * -1
        loser_grid[i,j] = loser_grid[i,j] * -1
        
plt.quiver(winner_scale,loser_scale, winner_grid, loser_grid, angles='xy',pivot='tail',headwidth=3,headlength=2,width= 0.005,color='b', label='spin_deravative_of_log')#, scale_units='xy', scale=1)

winner_grid = np.zeros((10,10))
loser_grid = np.zeros((10,10))
for i in range(winner_grid.shape[0]):
    for j in range(winner_grid.shape[1]):
        winner_grid[i,j], loser_grid[i,j] = pytorch_implementation(winner_scale[i],loser_scale[j],1.,0.1,'spin_deravative_of_ratio')
        # SRIJITH : Since we do gradient descent, we multiply by -1
        winner_grid[i,j] = winner_grid[i,j] * -1
        loser_grid[i,j] = loser_grid[i,j] * -1
        
plt.quiver(winner_scale,loser_scale, winner_grid, loser_grid, angles='xy',pivot='tail',headwidth=5,headlength=2,width= 0.003,color='r', label='spin_deravative_ratio')#, scale_units='xy', scale=1)




plt.xlabel('Winner response probability')
plt.ylabel('Loser response probability')
plt.title(f'Influence of Beta on Gradient Field')
plt.legend()

plt.savefig(f'temp.jpeg', dpi=500)


    

