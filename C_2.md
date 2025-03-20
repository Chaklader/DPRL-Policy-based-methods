# C-2: Policy Gradient Methods

1. Policy Gradient Fundamentals
    - The Policy Gradient Theorem
    - Likelihood Ratio Trick
    - REINFORCE Algorithm
    - Monte Carlo Policy Gradient
2. Gradient Estimation and Optimization
    - Sample-Based Estimates
    - Trajectory Collection
    - Policy Update Mechanisms
    - Credit Assignment Problem
3. Variance Reduction Techniques
    - Baseline Subtraction
    - Rewards Normalization
    - Future Rewards vs Total Returns
    - Importance Sampling

#### 1. Policy Gradient Fundamentals

Policy gradient methods represent a principled approach to directly optimize policy parameters using gradient-based
techniques. Unlike simpler methods like hill climbing, policy gradients compute exact gradient information to update the
policy in the direction of steepest ascent of expected return.

##### The Policy Gradient Theorem

The policy gradient theorem provides the mathematical foundation for calculating the gradient of expected return with
respect to policy parameters:

$$\nabla_\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$$

This theorem shows that the gradient can be estimated by sampling trajectories and using the log derivative of the
policy multiplied by the action-value function.

This image illustrates how policy gradient methods work. For actions that led to positive returns (WON), we increase
their probability, while for actions that led to negative returns (LOST), we decrease their probability. This intuitive
update rule emerges naturally from the mathematics of the policy gradient.

##### Likelihood Ratio Trick

The derivation of the policy gradient relies on a fundamental mathematical technique known as the likelihood ratio trick
(or REINFORCE trick):

$$\nabla_\theta \log P(\tau; \theta) = \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)}$$

This transforms the gradient of probabilities into a more tractable form involving the gradient of log probabilities,
which is easier to compute and has better numerical properties.

The full derivation proceeds as follows:

$$\nabla_\theta U(\theta) = \nabla_\theta \sum_\tau P(\tau; \theta)R(\tau) = \sum_\tau \nabla_\theta P(\tau; \theta)R(\tau)$$

$$= \sum_\tau P(\tau; \theta)\frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)}R(\tau) = \sum_\tau P(\tau; \theta)\nabla_\theta \log P(\tau; \theta)R(\tau)$$

This gives us the likelihood ratio policy gradient:

$$\nabla_\theta U(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[\nabla_\theta \log P(\tau; \theta)R(\tau)]$$

##### REINFORCE Algorithm

REINFORCE (Monte Carlo Policy Gradient) is the canonical policy gradient algorithm:

The pseudocode for REINFORCE is as follows:

1. Use the policy $\pi_\theta$ to collect m trajectories ${\tau^{(1)}, \tau^{(2)}, ..., \tau^{(m)}}$ with horizon H.
   Each trajectory $\tau^{(i)} = (s_0^{(i)}, a_0^{(i)}, ..., s_H^{(i)}, a_H^{(i)}, s_{H+1}^{(i)})$
2. Estimate the gradient $\nabla_\theta U(\theta)$:
   $$\nabla_\theta U(\theta) \approx \hat{g} = \frac{1}{m}\sum_{i=1}^m\sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})$$
3. Update the weights of the policy: $$\theta \leftarrow \theta + \alpha\hat{g}$$
4. Loop over steps 1-3.

##### Monte Carlo Policy Gradient

The Monte Carlo aspect of REINFORCE comes from using complete trajectory returns to estimate the expected return. This
approach:

- Provides unbiased gradient estimates
- Works with both continuous and discrete action spaces
- Directly optimizes the policy without intermediate value functions
- Naturally learns stochastic policies

#### 2. Gradient Estimation and Optimization

##### Sample-Based Estimates

In practice, we cannot compute the true gradient over all possible trajectories, so we use sampling to estimate it:

The image shows how we sample multiple trajectories and use them to estimate the policy gradient. By averaging across
trajectories, we reduce the variance of our gradient estimate, leading to more stable learning.

For a single trajectory, the gradient estimate is:

$$\nabla_\theta U(\theta) \approx \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t|s_t)R(\tau)$$

When using multiple trajectories (m), we average the gradients:

$$\nabla_\theta U(\theta) \approx \frac{1}{m}\sum_{i=1}^m\sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})$$

##### Trajectory Collection

Collecting meaningful trajectories is crucial for effective policy gradient estimation. When implementing REINFORCE,
trajectories are typically collected by:

1. Initializing an episode state
2. Sampling actions from the current policy
3. Executing actions and recording states, actions, rewards
4. Continuing until episode termination
5. Computing the return for the trajectory

```python
def collect_trajectories(policy, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        while not done:
            action_probs = policy(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        trajectories.append((states, actions, rewards))
    return trajectories
```

##### Policy Update Mechanisms

Once we have estimated the gradient, we update the policy parameters using gradient ascent:

$$\theta \leftarrow \theta + \alpha\nabla_\theta U(\theta)$$

Where:

- $\alpha$ is the learning rate
- $\nabla_\theta U(\theta)$ is the estimated policy gradient

The learning rate is a critical hyperparameter that controls the step size. Too large, and the updates may overshoot
optima; too small, and learning becomes inefficiently slow.

##### Credit Assignment Problem

One fundamental challenge in reinforcement learning is the credit assignment problem: determining which actions in a
trajectory contributed to the observed return. In basic REINFORCE, all actions receive credit proportional to the entire
trajectory return.

This approach can be inefficient because early actions might receive inappropriate credit for rewards that came much
later and were unrelated. We'll address this issue in the next section on variance reduction techniques.

#### 3. Variance Reduction Techniques

Policy gradient methods, while theoretically sound, often suffer from high variance in gradient estimates. Several
techniques have been developed to address this issue.

##### Baseline Subtraction

One of the most common variance reduction techniques is to subtract a baseline from the returns:

$$\nabla_\theta U(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)(R_t - b(s_t))$$

Where $b(s_t)$ is a state-dependent baseline, often implemented as a value function $V(s_t)$.

This technique reduces variance without introducing bias because the expected value of the baseline term is zero:

$$\mathbb{E}*{a \sim \pi*\theta}[\nabla_\theta \log \pi_\theta(a|s)b(s)] = b(s)\mathbb{E}*{a \sim \pi*\theta}[\nabla_\theta \log \pi_\theta(a|s)] = b(s) \cdot 0 = 0$$

##### Rewards Normalization

Another simple yet effective technique is to normalize rewards across trajectories:

$$R_i \leftarrow \frac{R_i - \mu}{\sigma}$$

Where:

- $\mu = \frac{1}{N}\sum_i R_i$ is the mean return
- $\sigma = \sqrt{\frac{1}{N}\sum_i(R_i - \mu)^2}$ is the standard deviation

This normalization helps stabilize learning by making the scale of updates more consistent across training.

##### Future Rewards vs Total Returns

A significant insight for variance reduction comes from using future rewards instead of total returns. The key
observation is that actions at time t can only influence future rewards, not past ones.

For a time step t, we can decompose the return into:

$$R_t^{\text{past}} + R_t^{\text{future}}$$

Where $R_t^{\text{past}}$ includes rewards up to time t-1, and $R_t^{\text{future}}$ includes rewards from time t
onward.

Since past rewards cannot be affected by the current action, a more efficient policy gradient uses only future rewards:

$$g = \sum_t R_t^{\text{future}}\nabla_\theta \log \pi_\theta(a_t|s_t)$$

This modification reduces variance without introducing bias, as the expected gradient remains the same.

##### Importance Sampling

Importance sampling provides a way to reuse trajectories collected from an older policy to update a newer one:

$$\nabla_\theta U(\theta') \approx \frac{P(\tau;\theta')}{P(\tau;\theta)} \sum_t \frac{\nabla_{\theta'}\pi_{\theta'}(a_t|s_t)}{\pi_{\theta'}(a_t|s_t)} R_t^{\text{future}}$$

This technique improves sample efficiency by allowing multiple policy updates from the same set of trajectories.
However, care must be taken to ensure the old and new policies aren't too different, as this can lead to high variance
in the importance weights.

The re-weighting factor is:

$$\frac{P(\tau;\theta')}{P(\tau;\theta)} = \frac{\pi_{\theta'}(a_1|s_1) \pi_{\theta'}(a_2|s_2) \ldots}{\pi_{\theta}(a_1|s_1) \pi_{\theta}(a_2|s_2) \ldots}$$

When the policies are similar, this ratio stays close to 1, enabling effective reuse of trajectories.
