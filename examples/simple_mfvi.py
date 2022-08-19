import os, sys
file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import torch as t
from torch.distributions import Normal
import torch.nn as nn
import bayesfunc as bf
import lab as B
import lab.torch
import matplotlib.pyplot as plt
import seaborn as sns
from wbml import plot
from bayesfunc.priors import InsanePrior, NealPrior
from examples.dgp import generate_data
import numpy as np
# Generate data
in_features = 1
out_features = 1
train_batch = 40

# t.manual_seed(0)
t.set_printoptions(precision=10, sci_mode=False)
B.default_dtype = t.float64
B.epsilon = 0.0
key = B.create_random_state(B.default_dtype, seed=0)
B.set_random_seed(0)

# X = t.zeros(train_batch, in_features)
# X[:int(train_batch/2), :] = t.rand(int(train_batch/2), in_features)*2. - 4.
# X[int(train_batch/2):, :] = t.rand(int(train_batch/2), in_features)*2. + 2.
# y = X**3. + 3*t.randn(train_batch, in_features)

# #Rescale the outputs to have unit variance
# scale = y.std()
# y = y/scale

dtype=t.float64
device="cpu"

# key, _, _, X, y, _, _, scale = generate_data(key, dgp=1, size=train_batch, xmin=-4.0, xmax=4.0)

# If comparing to term6, use the following:
X = t.load(os.path.join(file_dir, "data/mfvi_x_tr.pt"), map_location=t.device('cpu'))
y = t.load(os.path.join(file_dir, "data/mfvi_y_tr.pt"), map_location=t.device('cpu'))
# scale = 20.00762265544338
scale = 18.69775412585139

print(f"Scale DGP: {scale}")
print(f"ll_var: {3/scale}")


def plot(net):
    with t.no_grad():
        xs = t.linspace(-6,6,100)[:, None].to(device=device, dtype=dtype)
        print(xs.shape)
        #set sample=100, so we draw 100 different functions
        ys, _, _ = bf.propagate(net, xs.expand(100, -1, -1))
        mean_ys = ys.mean(0)
        std_ys = ys.std(0)
        ax = plt.gca()
        # plt.yticks(np.arange(B.min(ys), B.max(ys)+1, 1.0))
        plt.fill_between(xs[:, 0], mean_ys[:, 0]-2*std_ys[:, 0], mean_ys[:, 0]+2*std_ys[:, 0], alpha=0.5)
        plt.plot(xs, mean_ys)
        plt.scatter(X, y, c='r')
        
        ax.set_axisbelow(True)  # Show grid lines below other elements.
        ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
        plt.show()

def train(net):
    opt = t.optim.Adam(net.parameters(), lr=0.05)
    samples = 10
    for i in range(2000):
        opt.zero_grad()
        X_tr = X.expand(samples, -1, -1).detach().clone()
        output, logpq, _ = bf.propagate(net, X_tr)
        ll = Normal(output, 3/scale).log_prob(y).sum(-1).mean(-1) # Ober
        
        error = B.sqrt(B.mean((output.mean(0) - y)**2)) # rmse
        
        # assert ll.shape == (samples,)
        # assert logpq.shape == (samples,)
        
        # elbo = ll + logpq.mean()/train_batch
        
        elbo = ll + logpq/train_batch
        (-elbo.mean()).backward()
        
        if i == 0 or i % 100 == 0: 

            print(f"{i:4} | elbo: {elbo.mean():13.3f} - ll: {ll.mean():13.3f} - logpq: {-(logpq).mean():13.3f} - error: {error:8.5f}")

            # print("Final layer nz:")
            # print(1/net[-1].weights.post_log_var_scaled.exp())

            print("First layer means:")
            print(net[0].weights.post_mean)
            print("First layer nz:")
            print(1/net[0].weights.post_log_var_scaled.exp())

    
        opt.step()
    print(f"ELBO: {elbo.mean().item()}")
        

net = nn.Sequential(
    bf.FactorisedLinear(in_features=1, out_features=50, bias=True),
    nn.ReLU(),
    bf.FactorisedLinear(in_features=50, out_features=50, bias=True),
    nn.ReLU(),
    bf.FactorisedLinear(in_features=50, out_features=1, bias=True)
)

net = net.to(device=device, dtype=dtype)


train(net)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.scatter(X, y, label="Training data")
# plt.scatter(_z, _yz, label="Pre-training")
# plt.scatter(net[0].inducing_data, net[1][4].weights.u, label="Post-training")
plt.legend()
plt.show()

print(f"Scale DGP: {scale}")
print(f"ll_var: {3/scale}")

print("Final layer nz:")
print(1/net[-1].weights.post_log_var_scaled.exp())

# print("Final layer log var:")
# print(net[-1].weights.post_log_var_scaled)

# _z = net[0].inducing_data.detach().clone()
# _yz = net[1][4].weights.u.detach().clone()
# print("inducing pt locations:")
# print(_z)
# print("inducing pt labels:")
# print(_yz)


plot(net)
plt.show()
