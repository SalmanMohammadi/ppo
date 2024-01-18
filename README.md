# PPO
An implementation of Proximal Policy Optimization (PPO) using Generalized Advantage Estimation (GAE) and multi-processing.

## Installing

`python3 -m venv venv` to set up a virtual environment 

`cd pip install .` to install, or `pip install -e .` for development.

`python src/run_ppo.py config/pendulum.yaml` trains PPO for a given config file. Examples for different environments with hyperparameters I've found that work well can be found in `config/`.

## Paper Notes
My notes as I've been learning about these concepts.

### PPO
PPO is an on-policy reinforcement learning algorithm which constrains updates to model weights $\pi_\theta(a_t,s_t)$ and $V_\phi(s_t)$ using a modification to the Trust Region Policy Optimization (TRPO) algorithm.
It involves estimating a pessimistic lower bound on the constrained policy update:

```math
\hat{\mathbb{E}} [min(\frac{\pi_\theta(a_t,s_t)}{\pi_{\theta_{old}}(a_t,s_t)}\hat{A}_t, clip(\frac{\pi_\theta(a_t,s_t)}{\pi_{\theta_{old}}(a_t,s_{t)}}, 1 - \epsilon, 1 + \epsilon)\hat{A}_t]
```
In this manner, when the new policy diverges from the old policy, the policy update is constrained using $1+\epsilon$ or $1-\epsilon$, depending on whether the advantage function is positive or negative, respectively. 
When the new policy places lower density on actions compared to the previous policy, i.e. may be more conservative, the advantage update is smaller. When the new policy is greedier, i.e. places higher density on actions compared to the previous policy, the advantage update is *also* smaller. In this manner, PPO is considered to place a lower, pessimistic bound on updates.

PPO is usually used alongside Generalised Advantage Estimation (GAE) It follows the technique for advantage estimation from [Minh. et al. 2016](https://arxiv.org/pdf/1602.01783.pdf) which uses collected rewards across a trajectory of $T$ steps:

```math
\hat{A_t}=\delta_{t}+(\gamma \lambda)\delta_{t+1} + ... +(\gamma \lambda)^{{T-t-1}}\delta_{T-1}
```

which includes an exponential discount factor $\gamma$ and bias-variance trade-off parameter $\lambda$. For $\lambda=0$ we have the one-step temporal difference error:

```math
\hat{A_t}=\delta_{t}=r_{t}+\gamma V(s_{t+1})-V(s_t)
```

### GAE

GAE helps reduce variance in estimations of the advantage function using the following formula:

```math
\hat{A_{t}}=R_t(\lambda) - V(s_t)
```
which can be derived by considering the return 
```math
R_t=\sum_{l=0}^{T-t} \gamma^{l}r_{t+l}
```
as an unbiased sample of the expected return from a given state. 

However, in practice, variance in the dynamics of the policy and training result in stochasticity in the return, resulting in variance in return estimation. One method to reduce variance involves estimating the return over *n*-steps instead (similar to how we sample mini-batches in SGD). This can introduce bias in the smaller sample sizes, but may reduce overall variance from environment and agent dynamics. The remaining steps can be bootstrapped by approximating from the value function. When used with e.g. PPO, which helps reduce variance in policy and value functions, this can result in stable advantage estimations which are robust to variation in training dynamics.
Given *n* steps, the return is now:

```math
R_{t}^{(n)} = \sum_{l=0}^{n-1}[\gamma^{l}r_{t+l}]+\gamma^{n}V(S_{t+n})
```
i.e. estimate the return as usual but only up to $n$ time steps, and then use the expected value from $n$ steps on. For example, for $n=1$, the return is simply the Q-value function ($R_{t}^{1}=r_{t}+\gamma V(S_{t+n}$), and $n=\inf$ is the unbiased Monte-Carlo return estimation. This represents the trade-off between bias and variance in return estimation: the Q-value has very low variance but high bias, whereas the unbiased Monte-Carlo return has very high variance and low bias. $n$ directly controls this trade-off.

$\lambda$-returns also allow for controlling the trade-off between bias and variance by exponentially weighting future rewards with a decay parameter $\lambda$. While $\gamma$ controls the influence of longer time scales in the reward, $\lambda$ can be interpreted as the portion of future timesteps which the temporal difference is influenced by:

```math
R_t(\lambda)=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}R_{t}^{(n)}+\lambda^{T-t-1} R_{t}^{T-t}
```
This involves taking an exponentially weighted average over $n$ returns, i.e, returns estimated over $n$ steps for $n=1,...T-t-1$, where $T$ is the episode length (an exponentially-weighted moving average). $\lambda=1$ returns the unbiased Monte-Carlo return $R_t$, and $\lambda=0$ returns the single-step return $R_{t}^{(1)}$ (i.e. the Q-value).

(see [DeepMimic paper - Appendix A](https://arxiv.org/pdf/1804.02717.pdf#appendix.A))
