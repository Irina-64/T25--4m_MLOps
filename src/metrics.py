from prometheus_client import Counter, Histogram, Gauge

# -----------------------------
# RL шаги
# -----------------------------

RL_STEPS_TOTAL = Counter(
    "rl_steps_total",
    "Total number of RL agent steps"
)

# -----------------------------
# Распределение действий
# -----------------------------

RL_ACTIONS_TOTAL = Counter(
    "rl_actions_total",
    "Distribution of RL actions",
    ["action"]
)

# -----------------------------
# Latency хода
# -----------------------------

RL_STEP_LATENCY = Histogram(
    "rl_step_latency_seconds",
    "Latency of RL agent decision",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5)
)

# -----------------------------
# Reward
# -----------------------------

RL_REWARD = Histogram(
    "rl_reward",
    "Distribution of RL rewards",
    buckets=(-10, -5, -1, -0.5, 0, 0.5, 1, 5, 10)
)

# -----------------------------
# Победы
# -----------------------------

RL_WINS_TOTAL = Counter(
    "rl_wins_total",
    "Total RL wins"
)
