"""
AI-Based Intelligent Power and Propulsion Management for Satellites
Q-Learning Simulator
Author: Vedhiga V B, Avinashilingam University
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import json
import os

# ─────────────────────────────────────────────
# ENVIRONMENT PARAMETERS
# ─────────────────────────────────────────────
BATTERY_MAX        = 100.0   # % 
BATTERY_MIN        = 15.0    # % (unsafe threshold)
BATTERY_INIT       = 75.0    # % starting charge
ORBIT_ERROR_MAX    = 10.0    # km max deviation
ORBIT_ERROR_INIT   = 5.0     # km starting deviation
SOLAR_CHARGE_RATE  = 2.5     # % per step (daylight)
ECLIPSE_PROB       = 0.25    # probability of eclipse (no solar)

# Cost weights (α, β, γ)
ALPHA = 0.4   # power usage weight
BETA  = 0.4   # orbit error weight
GAMMA = 0.2   # battery risk weight

# ─────────────────────────────────────────────
# ACTIONS
# 0 = Idle             (low power, no correction)
# 1 = Minor Thruster   (medium power, small correction)
# 2 = Full Thruster    (high power, large correction)
# 3 = Charge Mode      (minimal ops, prioritise battery)
# ─────────────────────────────────────────────
ACTIONS = {
    0: {"name": "Idle",           "power": 1.0, "correction": 0.1},
    1: {"name": "Minor Thrust",   "power": 4.0, "correction": 1.5},
    2: {"name": "Full Thrust",    "power": 9.0, "correction": 3.5},
    3: {"name": "Charge Mode",    "power": 0.5, "correction": 0.0},
}
N_ACTIONS = len(ACTIONS)

# ─────────────────────────────────────────────
# STATE DISCRETISATION
# ─────────────────────────────────────────────
# Battery: 5 bins  [0-20, 20-40, 40-60, 60-80, 80-100]
# Orbit:   5 bins  [0-2, 2-4, 4-6, 6-8, 8-10]
BATTERY_BINS = np.array([0, 20, 40, 60, 80, 100])
ORBIT_BINS   = np.array([0, 2, 4, 6, 8, 10])
N_BAT_STATES = len(BATTERY_BINS) - 1   # 5
N_ORB_STATES = len(ORBIT_BINS)   - 1   # 5
N_STATES     = N_BAT_STATES * N_ORB_STATES  # 25

def discretise(battery, orbit_error):
    b = np.digitize(battery,     BATTERY_BINS[1:], right=True)
    o = np.digitize(orbit_error, ORBIT_BINS[1:],   right=True)
    b = np.clip(b, 0, N_BAT_STATES - 1)
    o = np.clip(o, 0, N_ORB_STATES - 1)
    return b * N_ORB_STATES + o

# ─────────────────────────────────────────────
# REWARD / COST FUNCTION
# ─────────────────────────────────────────────
def compute_reward(battery, orbit_error, action_id, next_battery):
    power_usage   = ACTIONS[action_id]["power"] / 9.0          # normalise 0-1
    orbit_norm    = orbit_error / ORBIT_ERROR_MAX               # normalise 0-1
    battery_risk  = max(0, (BATTERY_MIN - next_battery) / BATTERY_MIN)

    cost = ALPHA * power_usage + BETA * orbit_norm + GAMMA * battery_risk

    # Extra penalty for critically low battery
    if next_battery < BATTERY_MIN:
        cost += 1.5
    # Bonus for good orbit + healthy battery
    if orbit_error < 1.0 and battery > 60:
        cost -= 0.3

    return -cost   # reward = negative cost

# ─────────────────────────────────────────────
# ENVIRONMENT STEP
# ─────────────────────────────────────────────
def step(battery, orbit_error, action_id):
    act = ACTIONS[action_id]

    # Power drain
    new_battery = battery - act["power"]

    # Solar charging (eclipse probabilistic)
    if np.random.random() > ECLIPSE_PROB:
        new_battery += SOLAR_CHARGE_RATE
    else:
        new_battery += 0.5  # minimal trickle in eclipse

    new_battery = np.clip(new_battery, 0.0, BATTERY_MAX)

    # Orbit correction + natural drift
    drift = np.random.uniform(0.1, 0.5)
    new_orbit = max(0.0, orbit_error - act["correction"] + drift)
    new_orbit = min(new_orbit, ORBIT_ERROR_MAX)

    reward = compute_reward(battery, orbit_error, action_id, new_battery)
    done   = new_battery < 5.0  # terminal: critically depleted

    return new_battery, new_orbit, reward, done

# ─────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0,
                 eps_decay=0.995, eps_min=0.05):
        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min   = eps_min
        self.Q = np.zeros((N_STATES, N_ACTIONS))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train(n_episodes=600, max_steps=200):
    agent = QLearningAgent()

    ep_rewards      = []
    ep_battery_avg  = []
    ep_orbit_avg    = []
    ep_epsilon      = []
    action_log      = []       # for last episode
    battery_log     = []
    orbit_log       = []
    reward_log      = []

    for ep in range(n_episodes):
        battery     = BATTERY_INIT + np.random.uniform(-5, 5)
        orbit_error = ORBIT_ERROR_INIT + np.random.uniform(-1, 1)
        total_reward = 0
        batteries, orbits, rewards, actions = [], [], [], []

        for step_i in range(max_steps):
            state  = discretise(battery, orbit_error)
            action = agent.choose_action(state)

            new_bat, new_orb, reward, done = step(battery, orbit_error, action)
            next_state = discretise(new_bat, new_orb)
            agent.update(state, action, reward, next_state, done)

            battery, orbit_error = new_bat, new_orb
            total_reward += reward

            batteries.append(battery)
            orbits.append(orbit_error)
            rewards.append(reward)
            actions.append(action)

            if done:
                break

        agent.decay_epsilon()
        ep_rewards.append(total_reward)
        ep_battery_avg.append(np.mean(batteries))
        ep_orbit_avg.append(np.mean(orbits))
        ep_epsilon.append(agent.epsilon)

        # Save last episode trajectory for plotting
        if ep == n_episodes - 1:
            action_log  = actions
            battery_log = batteries
            orbit_log   = orbits
            reward_log  = rewards

    return {
        "agent":          agent,
        "ep_rewards":     ep_rewards,
        "ep_battery_avg": ep_battery_avg,
        "ep_orbit_avg":   ep_orbit_avg,
        "ep_epsilon":     ep_epsilon,
        "last_actions":   action_log,
        "last_battery":   battery_log,
        "last_orbit":     orbit_log,
        "last_reward":    reward_log,
        "Q_table":        agent.Q.tolist(),
    }

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
COLORS = {
    "bg":      "#0B0F1A",
    "panel":   "#111827",
    "accent1": "#00D4FF",
    "accent2": "#FF6B35",
    "accent3": "#7CFC00",
    "accent4": "#FFD700",
    "text":    "#E2E8F0",
    "grid":    "#1E293B",
}

def smooth(arr, w=20):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

def plot_training_results(data, save_path="/mnt/user-data/outputs/training_results.png"):
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    plt.rcParams.update({
        "text.color":       COLORS["text"],
        "axes.facecolor":   COLORS["panel"],
        "axes.edgecolor":   COLORS["grid"],
        "axes.labelcolor":  COLORS["text"],
        "xtick.color":      COLORS["text"],
        "ytick.color":      COLORS["text"],
        "grid.color":       COLORS["grid"],
        "grid.linewidth":   0.5,
        "font.family":      "monospace",
    })

    episodes = np.arange(len(data["ep_rewards"]))
    steps    = np.arange(len(data["last_battery"]))

    # ── 1. Cumulative Reward per Episode ──────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, data["ep_rewards"], color=COLORS["accent1"], alpha=0.25, lw=0.8)
    sm = smooth(data["ep_rewards"])
    ax1.plot(np.arange(len(sm)), sm, color=COLORS["accent1"], lw=2)
    ax1.set_title("Cumulative Reward per Episode (Learning Curve)", color=COLORS["text"], fontsize=11, pad=8)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.4); ax1.axhline(0, color="white", lw=0.4, ls="--")
    ax1.fill_between(np.arange(len(sm)), sm, alpha=0.08, color=COLORS["accent1"])

    # ── 2. Epsilon Decay ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(episodes, data["ep_epsilon"], color=COLORS["accent4"], lw=2)
    ax2.set_title("Exploration Rate (ε Decay)", color=COLORS["text"], fontsize=11, pad=8)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Epsilon")
    ax2.grid(True, alpha=0.4)
    ax2.fill_between(episodes, data["ep_epsilon"], alpha=0.1, color=COLORS["accent4"])

    # ── 3. Battery Level — Last Episode ──────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(steps, data["last_battery"], color=COLORS["accent3"], lw=1.5)
    ax3.axhline(BATTERY_MIN, color=COLORS["accent2"], lw=1.2, ls="--", label=f"Min safe ({BATTERY_MIN}%)")
    ax3.axhline(80, color=COLORS["accent1"], lw=0.8, ls=":", label="Optimal zone (80%)")
    ax3.fill_between(steps, data["last_battery"], BATTERY_MIN,
                     where=np.array(data["last_battery"]) < BATTERY_MIN,
                     color=COLORS["accent2"], alpha=0.3, label="Danger Zone")
    ax3.set_title("Battery Level — Final Trained Episode", color=COLORS["text"], fontsize=11, pad=8)
    ax3.set_xlabel("Time Step"); ax3.set_ylabel("Battery (%)")
    ax3.set_ylim(0, 105); ax3.grid(True, alpha=0.4)
    ax3.legend(fontsize=8, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

    # ── 4. Orbit Error — Last Episode ────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(steps, data["last_orbit"], color=COLORS["accent2"], lw=1.5)
    ax4.axhline(1.0, color=COLORS["accent3"], lw=1, ls="--", label="Target (<1 km)")
    ax4.set_title("Orbit Deviation — Final Episode", color=COLORS["text"], fontsize=11, pad=8)
    ax4.set_xlabel("Time Step"); ax4.set_ylabel("Error (km)")
    ax4.set_ylim(0, ORBIT_ERROR_MAX + 0.5); ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=8, facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    ax4.fill_between(steps, data["last_orbit"], alpha=0.1, color=COLORS["accent2"])

    # ── 5. Action Distribution — Last Episode ────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    action_names  = [ACTIONS[i]["name"] for i in range(N_ACTIONS)]
    action_counts = [data["last_actions"].count(i) for i in range(N_ACTIONS)]
    bars = ax5.bar(action_names, action_counts,
                   color=[COLORS["accent1"], COLORS["accent3"], COLORS["accent2"], COLORS["accent4"]])
    ax5.set_title("Action Distribution\n(Final Episode)", color=COLORS["text"], fontsize=10, pad=8)
    ax5.set_ylabel("Count"); ax5.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, action_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=9, color=COLORS["text"])
    plt.setp(ax5.get_xticklabels(), fontsize=8)

    # ── 6. Avg Battery Trend ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes, data["ep_battery_avg"], color=COLORS["accent3"], lw=1.5)
    sm_bat = smooth(data["ep_battery_avg"])
    ax6.plot(np.arange(len(sm_bat)), sm_bat, color="white", lw=1.5, ls="--")
    ax6.set_title("Avg Battery/Episode\n(Training Trend)", color=COLORS["text"], fontsize=10, pad=8)
    ax6.set_xlabel("Episode"); ax6.set_ylabel("Avg Battery (%)")
    ax6.axhline(BATTERY_MIN, color=COLORS["accent2"], lw=0.8, ls=":")
    ax6.grid(True, alpha=0.4)

    # ── 7. Avg Orbit Error Trend ─────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(episodes, data["ep_orbit_avg"], color=COLORS["accent4"], lw=1.5)
    sm_orb = smooth(data["ep_orbit_avg"])
    ax7.plot(np.arange(len(sm_orb)), sm_orb, color="white", lw=1.5, ls="--")
    ax7.set_title("Avg Orbit Error/Episode\n(Training Trend)", color=COLORS["text"], fontsize=10, pad=8)
    ax7.set_xlabel("Episode"); ax7.set_ylabel("Avg Error (km)")
    ax7.grid(True, alpha=0.4)

    # Title
    fig.text(0.5, 0.97,
             "Satellite Power & Propulsion RL Simulator — Q-Learning Results",
             ha="center", va="top", fontsize=14, color=COLORS["accent1"],
             fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.945,
             "Vedhiga V B | Avinashilingam University | α=0.4  β=0.4  γ=0.2",
             ha="center", va="top", fontsize=9, color=COLORS["text"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_q_heatmap(Q_table, save_path="/mnt/user-data/outputs/q_table_heatmap.png"):
    Q = np.array(Q_table)
    fig, axes = plt.subplots(1, N_ACTIONS, figsize=(16, 5), facecolor=COLORS["bg"])
    plt.rcParams.update({"font.family": "monospace", "text.color": COLORS["text"]})

    Q_grid = Q.reshape(N_BAT_STATES, N_ORB_STATES, N_ACTIONS)
    bat_labels = ["0-20%","20-40%","40-60%","60-80%","80-100%"]
    orb_labels = ["0-2","2-4","4-6","6-8","8-10"]

    for a in range(N_ACTIONS):
        ax = axes[a]
        im = ax.imshow(Q_grid[:, :, a], cmap="plasma", aspect="auto")
        ax.set_title(f"Action {a}: {ACTIONS[a]['name']}", color=COLORS["text"], fontsize=9)
        ax.set_xticks(range(N_ORB_STATES)); ax.set_xticklabels(orb_labels, fontsize=7, color=COLORS["text"])
        ax.set_yticks(range(N_BAT_STATES)); ax.set_yticklabels(bat_labels, fontsize=7, color=COLORS["text"])
        ax.set_xlabel("Orbit Error (km)", color=COLORS["text"], fontsize=8)
        if a == 0: ax.set_ylabel("Battery Level", color=COLORS["text"], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(N_BAT_STATES):
            for j in range(N_ORB_STATES):
                ax.text(j, i, f"{Q_grid[i,j,a]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white")

    fig.text(0.5, 1.01, "Q-Table Heatmap — Learned State-Action Values",
             ha="center", fontsize=12, color=COLORS["accent1"], fontweight="bold")
    fig.patch.set_facecolor(COLORS["bg"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {save_path}")

# ─────────────────────────────────────────────
# STATISTICS SUMMARY
# ─────────────────────────────────────────────
def print_stats(data):
    rewards = data["ep_rewards"]
    first50 = np.mean(rewards[:50])
    last50  = np.mean(rewards[-50:])
    improvement = ((last50 - first50) / abs(first50)) * 100 if first50 != 0 else 0

    print("\n" + "="*55)
    print("  SIMULATION STATISTICS")
    print("="*55)
    print(f"  Episodes trained       : {len(rewards)}")
    print(f"  Avg reward (first 50)  : {first50:.3f}")
    print(f"  Avg reward (last 50)   : {last50:.3f}")
    print(f"  Improvement            : {improvement:.1f}%")
    print(f"  Best episode reward    : {max(rewards):.3f}")
    print(f"  Final epsilon          : {data['ep_epsilon'][-1]:.4f}")
    print(f"  Avg battery (last ep)  : {np.mean(data['last_battery']):.1f}%")
    print(f"  Avg orbit err (last ep): {np.mean(data['last_orbit']):.2f} km")
    print(f"  Battery violations     : {sum(1 for b in data['last_battery'] if b < BATTERY_MIN)}")
    print("="*55)

    stats = {
        "episodes": len(rewards),
        "avg_reward_first50": round(first50, 4),
        "avg_reward_last50":  round(last50, 4),
        "improvement_pct":    round(improvement, 2),
        "best_reward":        round(max(rewards), 4),
        "final_epsilon":      round(data["ep_epsilon"][-1], 4),
        "avg_battery_last_ep": round(float(np.mean(data["last_battery"])), 2),
        "avg_orbit_last_ep":   round(float(np.mean(data["last_orbit"])), 2),
        "battery_violations":  int(sum(1 for b in data["last_battery"] if b < BATTERY_MIN)),
    }
    with open("/mnt/user-data/outputs/sim_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("✅ Saved: sim_stats.json")
    return stats

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("🚀 Starting Q-Learning Satellite Simulator...")
    print("   Episodes: 600 | Max steps/ep: 200 | States: 25 | Actions: 4")
    print()

    data = train(n_episodes=600, max_steps=200)

    print("📊 Generating plots...")
    plot_training_results(data)
    plot_q_heatmap(data["Q_table"])
    stats = print_stats(data)

    print("\n✅ All outputs saved to ./outputs/")
    print("   → training_results.png")
    print("   → q_table_heatmap.png")
    print("   → sim_stats.json")
