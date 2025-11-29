# A-Q-learning-agent-for-automated-trading-in-equity-stock-markets


# Introduction

In recent years, a wide range of methods have been developed for market analysis and investment decision-making, including statistical models, mathematical optimization techniques, and supervised and unsupervised machine learning algorithms. Nevertheless, these approaches often face limitations such as the need for labeled data, model stationarity, and a lack of adaptability to the dynamic nature of financial markets. For example, supervised methods require well-defined training sets that are not always available in real-world financial environments, and even when they are, the resulting models typically lack the ability to adjust to new market conditions after the initial training phase. Unsupervised methods, on the other hand, are mainly used to discover hidden structures in data and are not well suited to sequential decision-making problems such as repeated trading or portfolio optimization. Consequently, these approaches often perform poorly in terms of adapting to real-time changes, responding to emerging market conditions, and supporting long-term decision-making.

An ideal trading strategy must be dynamic and capable of continuously updating its trading decisions in line with changes in stock market trends. Hence, instead of relying on predefined, static strategies, it is necessary to employ a self-improving trading strategy based on the current behavior of the market. Reinforcement learning (RL) can be used to develop such a dynamic and adaptive strategy, because its learning process is driven by rewards received from the environment. Unlike traditional approaches, RL does not require predefined target outputs and instead derives an optimal policy through experience and ongoing interaction with the environment. This property has made RL a promising candidate for modeling complex, time-varying, and long-horizon problems such as financial trading, where sequential decisions and continuous feedback play a central role.

Accordingly, in this project, similar to the study “Designing a Q-learning-based agent for automated stock market trading” \citep{1}, the aim is to develop a model using the Q-learning algorithm—which is a model-free reinforcement learning method—that can generate a dynamic trading strategy, adapt to current market conditions, and achieve near-optimal performance.

# Methodology

This section provides a brief overview of the Markov Decision Process (MDP), reinforcement learning, Q-learning algorithms, and stock technical analysis.﻿

## Markov Decision Process
A Markov Decision Process (MDP) is a stochastic process used to define a sequential decision-making problem under uncertainty. This process can be either discrete or continuous: if the variable whose state we are examining is observed at specific points in time, the process is discrete; otherwise, when the state can be observed at any arbitrary time, the process is continuous. The Markov property states that the next state depends only on the current state. An MDP formulates the problem as a tuple (S, A, P, R, γ), such that:

S – State space, the set of all possible states.﻿

A – A finite set of actions.﻿

P: S×A×S → [0,1] – State transition probability function.

R: S×A → R – Reward function.﻿​

γ ∈ [0,1] – Discount factor.

## Types of rewards
Immediate reward ($R_{t+1}$): The reward received immediately after taking an action in a particular state. It reflects the direct value of that action at the moment it is taken.﻿

Cumulative return ($G_t$): The total discounted reward from time t onward, representing the sum of rewards the agent receives from that point forward, given by the equation below:﻿

$$
		 	G_t = R_{t+1} + \gamma R_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

Furthermore, its recursive form is given as follows:﻿

$$
		G_t = R_{t+1} + \gamma G_{t+1}
$$

where $\gamma \in [0,1]$ , and it is used to discount future rewards and to prevent the return from becoming infinite in cyclic Markov processes. The closer $\gamma$ is to zero, the less importance is placed on future rewards (short-term view), and the closer it is to one, the more importance is placed on future rewards (long-term view).


## Policy:
The agent’s action selection, or its strategy, is modeled by a function known as the policy. The policy can be either stochastic or deterministic:﻿

Stochastic policy: A function $\pi : A \times S \rightarrow [0, 1]$ that specifies the probability of choosing action $A_t = a$ in state $S_t = s$.

Deterministic policy: A function $\pi : S \rightarrow A$ that maps each state directly to a specific action.﻿

## State–action value function $Q(S,A)$ :

The expected return at time t, when starting from state s, taking action a, and then following policy π, is given by the state–action value function $Q_{\pi}(s, a)$.﻿

$$
		Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \mid s_t = s, a_t = a]
$$

## Optimal state–action value function $Q_{\pi}^{*}(s, a)$ :

This function represents the supremum of the state–action value function taken over all admissible policies.﻿

$$
Q_*(s, a) = \max_{\pi} Q_{\pi}(s, a)
$$

This function provides the maximum expected return for state s and action a, assuming that an optimal policy is followed from that point onward.

The relationship between the optimal policy and the optimal state–action value function is given as follows:

$$ 
{\pi}_*(s) = \arg\max_{a} Q_*(s, a) 
$$

where the optimal policy $\(\pi^*\)$ selects the action that maximizes the optimal Q-value at each state.

## Reinforcement Learning:
Reinforcement learning (RL) is a branch of machine learning that operates based on interaction with an environment and the receipt of rewards. In RL, an agent observes the state of the environment, takes actions, and receives rewards or penalties as feedback for the consequences of those actions. Through this trial‑and‑error process, the agent gradually learns an optimal policy that maximizes its overall cumulative reward.﻿

In many cases, the agent aims to reach a specific configuration of the environment, known as the goal state. Initially, the environment may be in a particular configuration called the initial state. The agent interacts with the environment according to its policy $\pi$; after each action, the environment transitions to a new state and the agent receives a reward. This process begins from the initial state and continues until the environment reaches the goal state. In some problems and algorithms, to prevent excessively long interactions, the agent–environment interaction is limited to a fixed number of episodes, where an episode is defined as a sequence of states, actions, and rewards that starts from an initial state and terminates upon reaching a goal or terminal state.

Each episode can terminate in three ways:﻿

1- Reaching the goal state (Goal state).﻿

2- Reaching a terminal state (Terminal State), such as failure or resource exhaustion.﻿

3- Truncated: the episode is ended artificially due to limits like a maximum number of steps (max-step) or a time limit, before reaching a goal or terminal state.﻿

While the agent interacts with the environment, the policy is updated using the rewards the agent receives for actions taken in different states.﻿

At the outset, the policy is initialized randomly (e.g., with a uniform distribution) or arbitrarily, and then gradually refined and optimized.﻿
The learner’s objective is to obtain an optimal policy—one that yields the highest possible cumulative return compared to all other policies.﻿

## Q-learning Algorithm:

To begin with, it is necessary to describe the two main categories of policy-learning methods.

Off-policy: In this approach, the policy used to select actions (the behavior policy) is different from the policy that the model is actually learning (the target policy). In off-policy learning, the agent can learn an optimal policy from experiences generated by another policy, even a purely random one.

On-policy: In this case, the behavior policy is exactly the same as the target policy. In other words, the agent learns only from data that are generated by the very policy it is trying to optimize.

Q-learning is a model-free, off-policy reinforcement learning method that aims to obtain the optimal state–action value function $Q_{*}(s,a)$ through interaction with the environment. In this approach, the values $Q(s,a)$ for all state–action pairs $(s,a)$, where s ∈ S and $a \in \mathcal{A} $, are stored in a table known as the Q-table. There are various ways to initialize the Q-table; in this work, we assume that all its entries are set to zero at the beginning. The Q-learning algorithm then updates the entries of this table using a temporal-difference (TD) update rule:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right)
$$

where $\alpha$ is the learning rate and $\gamma \in [0,1]$. The optimal state–action value function $Q_{*}(s,a)$ is obtained as a result of the convergence of this iterative update.

<div justify-content: center; >
<img src="https://leimao.github.io/images/blog/2019-03-14-RL-On-Policy-VS-Off-Policy/q-learning.png" alt="Girl in a jacket" width="800" height="900">
</div>

## $\varepsilon$-greedy Policy :

The $\varepsilon$-greedy policy helps the learning agent decide which action to take, based on the current estimates of the Q-value function:﻿

1- Exploitation: In state S, with probability $1-\varepsilon$ , the agent selects the action that has the highest Q-value. This means the agent is exploiting its current knowledge in order to obtain the greatest possible reward.

2- Exploration: With probability $\varepsilon$, the agent selects a random action to explore new possibilities and determine whether there is a better way to obtain reward. This helps the agent discover new strategies and improve its decision-making over time.﻿

Note that, as mentioned earlier, all entries of the Q-table are initially set to zero; therefore, in the exploitation phase and also in update of the Q-learning algorithm, the first action in the table for each state is selected at every time step t. Consequently, in order to see the effect of other actions and to allow the Q-table values to converge to their optimal values, we employ the $\varepsilon$-greedy policy. At the beginning, we set $\varepsilon$=1, which encourages the agent to explore different paths and to discover an approximately optimal trajectory, but as time passes we gradually decrease $\varepsilon$, because choosing actions that have previously been suboptimal (and may still be suboptimal) too frequently can lead to substantial losses. When the table has converged, $\varepsilon$ is chosen to be a value close to zero; this is because, over time, the environment may change and, as a result, the Q-table entries may also change, so the agent needs to keep trying new paths in order to update the table values and maintain near-optimal performance.

** Our trading strategy is based on the paper “A Q-learning Agent for Automated Trading in Equity Stock Markets”[^1]. The detailed definition of the state representation, action space, reward function, and other implementation choices is provided and documented within the accompanying notebook file.

# Refrences:

[^1]: [A Q-learning Agent for Automated Trading in Equity Stock Markets](https://www.sciencedirect.com/science/article/abs/pii/S0957417420305856)


