from utils import *
from viz import *

# probabilities
α = 0.5
β = 0.5

# rewards
r_search = 2
r_wait = 1
assert r_search > r_wait

# training parameters
epochs = 1000
steps = 1000

def train(epochs=epochs, steps=steps):
    robot = Robot()
    env = Environment(α, β, r_search, r_wait, robot)
    
    #TODO: finish