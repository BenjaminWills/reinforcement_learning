# Markov Decision Processes (MDPs)

[[Policy Iteration]]

> A Markov Decision Process (MDP) is a framework to describe sequential decision making processes, that is: processes in which an outcome depends on one or more previous bit of history.

In `reinforcement learning` we call a *decision maker* the **agent**, their *decisions* **actions** that they execute within their **environment** or **state**.

An MDP is **not** deterministic, meaning that there is a probability distribution of outcomes. More specifically an MDP works via *stochastic non-determinism*.

> [!faq]- A process is called *stochastic* if it evolves randomly over time. An example of a stochastic process is counting the number of coin flips.

## Formal definition

A MDP is a **fully observable** (meaning we know *all* that goes on in the process) and **probabilistic** (meaning random!) state model. A common formulation is a **discounted-reward Markov Decision Process**. A DR-MDP is then completely defined as a tuple:

$$
(S, s_0, A, P, r, \gamma)
$$
Where:

* The `state space`, $S$
* The `initial state` (within the state space), $s_0 \in S$
* `Actions` $A(s) \subset A$  applicable in each state $s\in S$ that our agent can execute 
* `Transition probabilities` $P_a(s'|s)$ for $s\in S$ and $a \in A(s)$ (i.e. the probability that given action $a$ we go from state $s$ to state $s'$)
* `Rewards` $r(s,a,s')$, positive or negative of transitioning from state $s$ to state $s'$ using action $a$
* A discount factor $0 \leq \gamma < 1$ which will dictate how large or small a reward is, and how much we go back in time when assigning reward

A **state** is a situation that the agent can be in, each state contains information using which allows for a decision to be made. The **state space** is the set of all possible states in a given environment.

**Actions** allow agents to affect the environment / state. **Transition probabilities** tell us the effect of each action. 

**Rewards** tell us the benefit of each action for a particular state, and finally the **discount factor** tells us how much a future reward should be discounted related to a current reward - it allows us to control the importance of history in a model.

If our *agent* receives a set of rewards $\{r_i\}_{i\in \mathbb{N}}$ then the `discounted reward` is:

$$
V = r_1 + \gamma r_2 + \gamma^2r_3 + \dots = \sum^N_{i=1}\gamma^{i-1}r_i
$$
If $V_t$ is the reward at time $t$ then we can show that $V_t = r_t + \gamma V_{t+1}$

But how do we make our decisions? We use a policy, more on this below.
## What is a policy?

A `policy` in RL refers to the way that a RL model makes decisions. Usually it is defined using the greek letter `pi` ($\pi$). A policy is a function that takes in the current **state** of the environment and returns information about which **action** to take from that state. 

An example of policy is in the game Blackjack, the state the players current score and if they are currently holding an ace - the policy will then recommend *hitting* or *sticking*.

A policy can be either **deterministic** - will return a single **action** given a state, or **stochastic** (a fancy word for controlled randomness) - will return a probability distribution of the **actions** to take which should be sampled from.

### Deterministic Policies

A *deterministic policy* is a function that maps *states* to *actions*, i.e. $\pi: S \rightarrow A$. So our *agent* should map some state, $S$, to some action, $a$, such that $\pi(s) = a$.

### Stochastic Policies

A *stochastic policy* is a probability distribution based on *states* and *actions*, i.e. $\pi: S \times A \rightarrow \mathbb{R}$, so $\pi(s,a)$ is the probability that you should take action $a$ when you're in state $s$.

## Optimal Solutions for MDPs

To define a solution for an MDP we need to think about how we can assess an MDP. The solution is through the rewards, we ideally want to define a policy, $\pi$, that maximises our reward and therefore achieves the best outcomes possible. A way to do this is to maximise the *expected discounted reward*, i.e. maximises the average reward when using the policy.

$$
V_\pi(s) = E_\pi[\sum_i\gamma^ir(s_i,a_i,s_{i+1})| s_0 = s, a_i = \pi(s_i)]
$$
Here $V_\pi(s)$ is the expected discounted reward of following policy $\pi$ from an initial state $s \in S$.

### The Bellman equation

This is a **very** famous equation in RL, developed by `Richard Bellman`, shows how we can optimise any policy - in fact Bellman defined a condition that **must** hold for a policy to be optimal.

$$
V(s) = \max_{a\in A(s)}\sum_{s'\in S}P_a(s'|s)[r(s,a,s') + \gamma V(s')]
$$
In plain english this says: $V$ is optimal if for all states $V(s)$ describes he total discounted reward for taking the action with the highest reward over an indefinite time horizon. To break the equation down further:

1. $max_{a \in A(s)}$ says that we're looking for the action ($a$) in the space of all possible actions given some state ($A(s)$) that maximises the sum
2. $P_a(s'|s)$ the probability of going to state $s'$ from state $s$ given action $a$ (the policy!)
3. $r(s,a,s')$ is the immediate reward for going to state $s'$ from state $s$ by doing action $a$
4. $V(s')$ is the value of the subsequent state $s'$
5. Finally, the sum itself is the expected reward for doing action $a$ in state $s$

Notice how this equation is recursive since it calls its self within itself! $V(s)$ is the value of the action which will maximise the expected reward for that given state.

Right now we have a value function that depends only on the state of our environment, $s$, it tells us the expected reward of being in state $s$ and acting optimally according to our policy. It would be more intuitive to abstract this notion to optimise some `action-value` function that we'll call $Q$ which will give us the expected reward given a state $s$ and an action $a$.

$$
Q(s,a) = \sum_{s'\in S}P_a(s'|s)[r(s,a,s' + \gamma V(s'))]
$$
So we can massively simplify the bellman equation to be:

$$
V(s) = \max_{a\in A(s)}Q(s,a)
$$
### Policy extraction

We can use the bellman equations to find an optimal deterministic policy for any state $s$, in theory we could also find a non-deterministic policy by ranking the actions by reward and weighting their probabilities that way.

$$
\pi(s) = argmax_{a\in A(s)} Q(a,s)
$$
Or in the non-deterministic sense we can use the soft-max function to create weighted probabilities based on rewards, if we define the softmax function to be $\lambda(\vec{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{dim(x)}e^{x_j}}$.