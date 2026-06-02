# Implementation-of-MC-prediction-for-estimating-the-state-value-function-

## Aim

To implement the Monte Carlo (MC) Prediction algorithm for estimating the state-value function of an agent interacting with an environment and to calculate and plot the estimated state-value function.

---

## Objective

- To understand Monte Carlo prediction in Reinforcement Learning.
- To estimate the value of each state using sampled episodes.
- To visualize the learned state-value function using a plot.

---

## Theory

Monte Carlo Prediction is a model-free reinforcement learning method used to estimate the value function of a policy. It learns directly from complete episodes of interaction with the environment.

The state-value function is defined as:

\[
V(s) = E[G_t \mid S_t = s]
\]

Where:

- \(V(s)\) = Value of state \(s\)
- \(G_t\) = Return obtained from time step \(t\)

Monte Carlo methods calculate the average return obtained after visiting a state multiple times.

---

## Algorithm

### Monte Carlo Prediction Algorithm

1. Initialize:
   - State-value function \(V(s)\)
   - Returns list for each state

2. Generate an episode using the given policy.

3. For each state appearing in the episode:
   - Calculate the return \(G\)
   - Store the return for that state
   - Update the value function using average return

4. Repeat for many episodes.

5. Plot the estimated state-value function.

---

## Program

```
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.9
episodes = 5000

V = defaultdict(float)

returns = defaultdict(list)

def policy(state):
    return env.action_space.sample()

def generate_episode():
    episode = []

    state, _ = env.reset()
    done = False

    while not done:
        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode.append((state, action, reward))

        state = next_state

    return episode

for ep in range(episodes):

    episode = generate_episode()

    G = 0
    visited_states = set()

    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = gamma * G + reward

        # First-visit MC
        if state not in visited_states:

            returns[state].append(G)

            V[state] = np.mean(returns[state])

            visited_states.add(state)

print("State Value Function:\n")

for s in range(env.observation_space.n):
    print(f"State {s}: {V[s]:.3f}")

value_grid = np.zeros((4,4))

for state in range(16):
    row = state // 4
    col = state % 4

    value_grid[row, col] = V[state]

plt.figure(figsize=(6,6))

plt.imshow(value_grid)

for i in range(4):
    for j in range(4):
        plt.text(j, i,
                 round(value_grid[i,j],2),
                 ha='center',
                 va='center',
                 color='white',
                 fontsize=12)

plt.title("State Value Function Estimate")
plt.colorbar()
plt.show()
```
## Output


<img width="328" height="389" alt="image" src="https://github.com/user-attachments/assets/195ec52a-8762-4520-9f48-69b20d4160b4" />




### Output Graph

<img width="724" height="612" alt="image" src="https://github.com/user-attachments/assets/8f1714a2-d0a8-4d51-a02c-22370a235e19" />

The following heatmap is generated for the estimated state-value function:

- Darker colors represent lower state values.
- Terminal states have value 0.
- States farther from terminal states have larger negative values.



---
## Result

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the state-value function of the environment. The value of each state was calculated using sampled episodes, and the estimated state-value function was plotted successfully using a heatmap.




       

