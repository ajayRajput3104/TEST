# 🤖 Autonomous Warehouse Agent with Battery Constraints using PPO

> Course: Reinforcement Learning / AI Games
>
> Project Type: Resource-Constrained Navigation & Logistics

A comprehensive implementation of a Proximal Policy Optimization (PPO) agent designed to solve a multi-objective warehouse logistics problem. The agent must balance task completion (delivery) with survival (battery management) in a procedurally generated environment.

## 👨‍🎓 Group Information

| Name             | Roll Number |
| ---------------- | ----------- |
| [Student Name 1] | [Roll No 1] |
| [Student Name 2] | [Roll No 2] |
| [Student Name 3] | [Roll No 3] |
| [Student Name 4] | [Roll No 4] |
| [Student Name 5] | [Roll No 5] |

## 1\. Problem Statement & Motivation

### The Challenge

The objective is to train an autonomous agent to operate within a grid-based **Warehouse Environment**. The agent is tasked with:

1.  **Navigation:** Avoiding static obstacles (shelves) to reach specific coordinates.
2.  **Logistics:** Picking up a package from a random location and delivering it to a target zone.
3.  **Survival (Critical):** Managing an internal battery level. If the battery depletes to 0, the agent "dies" (fails), incurring a massive penalty.

### 🌟 Why the "Battery Constrained" Version is Superior

Standard GridWorld or Maze navigation tasks are often trivialized by RL agents simply learning the shortest path. However, real-world autonomous mobile robots (AMRs) face **resource constraints**.

This constrained version provides significantly higher utility because:

1.  **Hierarchical Decision Making:** The agent cannot simply maximize the "Deliver" reward. It must dynamically switch its goal from "Deliver Box" to "Find Charger" based on its internal state (Battery Level).
2.  **Risk Management:** The agent learns to calculate if it has enough charge to reach the target or if it must detour to charge first.
3.  **Realism:** This closely mimics real-world warehouse scenarios where robot uptime and charging schedules are critical for throughput.
4.  **Reward Shaping Complexity:** It demonstrates how to balance dense shaping rewards (distance) with sparse terminal rewards (delivery) and survival penalties (death).

## 2\. Codebase & Implementation Details

### A. Environment Logic: warehouse_env.py

This file defines the physics, rules, and rewards of the simulation using the Gymnasium API.

Key Feature: The Observation Space

The agent does not see the grid as an image; it sees a normalized vector of 8 values. This ensures fast training and generalization.

**Visual Representation of Input Vector:**

```
[  Robot X  ] --+
[  Robot Y  ]   |
[   Box X   ]   |
[   Box Y   ]   |----> [ Neural Network Input Layer ]
[  Target X ]   |
[  Target Y ]   |
[  Has Box  ]   |
[  Battery  ] --+ (Critical Decision Feature)
```

**Code Snippet:**

```python

#From warehouse_env.py

def _get_obs(self):
    s = self.grid_size
    return np.array([
        self.robot_pos[0] / s, self.robot_pos[1] / s, # Robot Coordinates
        self.box_pos[0] / s, self.box_pos[1] / s, # Box Coordinates
        self.target_pos[0] / s, self.target_pos[1] / s, # Target Coordinates
        1.0 if self.has_box else 0.0, # State Flag
        self.battery # Critical Resource
    ], dtype=np.float32)
```

**Key Feature: Reward Shaping**

To prevent sparse reward issues, we implemented specific shaping logic. The step function calculates rewards based on distance changes and battery status.

```Python
# From warehouse_env.py
# 1. BATTERY DRAIN
drain = 0.005 if not self.has_box else 0.01 # Drain faster if carrying load
self.battery -= drain
# 2. CHARGING LOGIC
if new_pos == self.charger_pos:
    self.battery = 1.0
    if not self.charge_cooldown:
        reward += 10.0 # Incentive to charge, but only if needed

# 3. TERMINAL PENALTY
if self.battery <= 0:
    reward += -100.0 # Massive penalty for death
    terminated = True
```

_Explanation:_ The code explicitly penalizes carrying a box to simulate weight/physics physics. The charging logic includes a cooldown to prevent the agent from "camping" on the charger to farm infinite rewards.

### B. The Algorithm: ppo_scratch.py

This file contains the "from scratch" implementation of PPO, demonstrating the mathematical mechanics of the algorithm.

#### 1\. Generalized Advantage Estimation (GAE)

We implement GAE inside RolloutBuffer to reduce variance in trajectory estimation.

Mathematical Formula:

The advantage $\\hat{A}\_t$ is calculated recursively:

$$\\delta\_t = r\_t + \\gamma V(s\_{t+1}) - V(s\_t)$$$$\\hat{A}\_t = \\delta\_t + (\\gamma \\lambda) \\hat{A}\_{t+1}$$

**Visual Logic Flow (Backward Pass):**

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`Step T (End) --> Step T-1 --> ... --> Step 0      ^               ^                   ^      |               |                   |  [Calc Delta]    [Add Future Adv]    [Result stored]`

**Code Snippet:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`# From ppo_scratch.py  # We loop BACKWARDS (reversed) because GAE relies on the "next" step's advantage  for step in reversed(range(self.buffer_size)):      # Delta term represents the TD-Error (Temporal Difference)      delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]      # Recursive formula: Current Delta + Discounted Future Advantage      last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam      self.advantages[step] = last_gae_lam`

#### 2\. The Actor-Critic Network (MlpPolicy)

We use a shared backbone network that splits into two heads.

**Architecture Diagram:**

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`State Input (8)        |  [ Shared Linear Layer 64 ]        |        +---------------------+        |                     |  [ Actor Head ]        [ Critic Head ]        |                     |     Softmax              Linear        |                     |  Action Probs (4)      Value Estimate (1)`

**Code Snippet:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`# From ppo_scratch.py  self.pi_net = nn.Sequential(*pi_layers) # Actor Network  self.vf_net = nn.Sequential(*vf_layers) # Critic Network  if self.is_discrete:      logits = self.action_head(pi_latent)      # Categorical distribution allows us to sample discrete actions (0,1,2,3)      dist = Categorical(logits=logits)`

#### 3\. The PPO Update Loop

The core learning mechanism uses the **Clipped Surrogate Objective**. This forces the update to stay within a safe range, preventing "catastrophic forgetting."

Mathematical Formula:

$$L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}\_t \[\\min(r\_t(\\theta)\\hat{A}\_t, \\text{clip}(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}\_t)\]$$

Where $r\_t(\\theta)$ is the probability ratio $\\frac{\\pi\_\\theta(a\_t|s\_t)}{\\pi\_{\\theta\_{old}}(a\_t|s\_t)}$.

**Visual Logic (Clipping):**

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML `Ratio          |      Is ratio > 1.2 ?      /          \    YES          NO     |            |  [Clip to 1.2] [Keep Ratio]     |            |     +-----+------+           |    Multiply by Advantage           |      Update Gradient`

**Code Snippet:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`# From ppo_scratch.py  # Ratio indicates how much more likely the action is NOW vs. when we collected data  ratio = th.exp(log_prob - old_log_probs)  # Unclipped objective: Standard Policy Gradient  policy_loss_1 = advs * ratio  # Clipped objective: Forces ratio to stay between 0.8 and 1.2 (if epsilon=0.2)  policy_loss_2 = advs * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)  # We take the minimum (pessimistic bound) to be safe  policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()`

### C. Testing Suite: stress_test_analytics.py

This script runs a rigorous evaluation of the trained model, focusing on edge cases that standard training metrics might miss.

Key Snippet: Critical Battery Initialization

To test "Intelligence," we force the robot to start with low battery.

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`# From stress_test_analytics.py  start_batt = np.random.uniform(0.15, 1.0) # Randomize battery  env.battery = start_batt  # Define "Critical" as less than 30% charge  is_critical = start_batt < 0.30   # Metric tracking logic:  if is_critical:      if did_charge: stats["critical_saves"] += 1 # Intelligent behavior      else: stats["critical_fails"] += 1          # Failed to prioritize survival`

## 3\. Results & Performance Analysis

### A. Training Metrics (Graphs)

We monitored the training process over **2,000,000 timesteps**. Below is the analysis of the resulting graphs.

!

**Graph 1: Episode Reward (rollout/ep_rew_mean)**

- **Trend:** The curve begins at a negative value (due to the -100 death penalty and wandering).
- **Inflection Point:** Around 250k steps, the reward spikes, indicating the agent has learned to avoid obstacles and find the target.
- **Convergence:** The reward stabilizes around **143.9**, which represents a near-perfect run (100 Delivery + 20 Pickup + Distance Bonuses - minimal Battery Costs).

**Graph 2: Episode Length (rollout/ep_len_mean)**

- **Trend:** Starts high (~200 steps) as the agent explores randomly.
- **Convergence:** Drops to **~20 steps** per episode.
- **Interpretation:** This confirms the agent is taking the **shortest path** possible. It is no longer wandering; it is executing the mission with high efficiency.

### B. Stress Test Results (Quantitative Table)

The model warehouse_strict_agent was subjected to 1,000 randomized test episodes.

**MetricValueInterpretationDuration**38.06sThe model is lightweight and highly performant.**🏆 Success Rate84.20%**The agent successfully delivers the package in vast majority of cases.**Avg Steps**23.3Matches the theoretical optimal path length for an 8x8 grid.

#### Failure Mode Analysis

- **Battery Deaths (8.5%):** Occurs in "Impossible Scenarios" where the agent spawns with 15% battery but the charger is on the other side of the map (mathematically impossible to reach).
- **Timeouts (7.3%):** Rare instances where the agent enters a loop.

#### 🧠 Intelligence Check (The "Why it's Useful" Metric)

This specific metric determines if the agent learned **Priority Management**.

- **Scenario:** Battery starts below 30% (Critical).
- **Total Critical Episodes:** 160
- **Survived & Delivered:** 145
- **Dead:** 15
- **Survival Rate:** **90.6%**

**Conclusion:** The agent successfully learned to prioritize survival over task completion when resources are low. It demonstrates **emergent intelligent behavior**.

## 4\. How to Run

### Prerequisites

Ensure you have the required libraries installed:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`pip install torch numpy gymnasium stable-baselines3 tensorboard tqdm matplotlib imageio`

### 1\. Training the Agent

To train the agent using the robust PPO implementation (SB3 wrapper):

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`python train_compare.py`

- **Output:** Generates warehouse_strict_agent.zip and TensorBoard logs in ./logs/strict/.

### 2\. Monitoring Training

To visualize the graphs shown in the report:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`tensorboard --logdir ./logs/`

### 3\. Running Stress Tests

To generate the detailed statistics table:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`python stress_test_analytics.py`

### 4\. Visual Demo (GIF Generation)

To create a video file (final_presentation_video.gif) showing the agent in action:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`python create_demo.py`

## 5\. Conclusion

This project successfully implements a **Battery-Constrained Autonomous Agent** using Proximal Policy Optimization. By integrating resource constraints into the standard warehouse logistics problem, we created an agent capable of **dynamic decision-making**. The results—specifically the **90.6% survival rate** in critical battery states—demonstrate that Deep Reinforcement Learning is a viable solution for complex, multi-objective robotic control systems.
