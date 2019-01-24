from matplotlib import pyplot as plt
import re

RETURN_VALUE_RE = r"Episode: [0-9]+ Return: ([0-9.]*)"

episode_returns = []
with open("simulation_results/returns") as returns_file:
    lines = returns_file.readlines()
    for line in lines:
        result = re.search(RETURN_VALUE_RE, line)
        try:
            episode_return = float(result.group(1))
        except:
            episode_return = 0.0
        episode_returns.append(episode_return)

plt.plot(episode_returns)
plt.xlabel("Episode number")
plt.ylabel("Return (no discounting)")
plt.show()
