import matplotlib.pyplot as plt

# Project data
tasks = [
    "Dataset Collection & Data Preprocessing",
    "Model Training & Development",
    "Model Evaluation & Optimization",
    "Integration & System Testing",
    # "Dashboard Development",
    # "Client Acceptance Testing (CAT)"
]

start_weeks = [1, 5, 8, 10]
durations = [1, 5, 4, 3, 3]
colors = ["lightcoral", "#6ec5c0", "#67b7d1", "#a9cbbd", "#eadba6", "#c9a0cf"]

# Create figure
fig, ax = plt.subplots(figsize=(14.1, 8.06))
fig.patch.set_facecolor("#f0f0f0")
ax.set_facecolor("#f5f5f5")

# Reverse order so first task appears at the top
tasks_plot = tasks[::-1]
start_plot = start_weeks[::-1]
durations_plot = durations[::-1]
colors_plot = colors[::-1]

# Draw horizontal bars
for i, (task, start, duration, color) in enumerate(zip(tasks_plot, start_plot, durations_plot, colors_plot)):
    ax.barh(y=i, width=duration, left=start, height=0.8, color=color, edgecolor="#333333", linewidth=1.2)

    # Add duration label inside each bar
    ax.text(
        start + duration / 2, i, f"{duration}w", ha="center", va="center", fontsize=12, fontweight="bold", color="black"
    )

# Axis labels and title
ax.set_yticks(range(len(tasks_plot)))
ax.set_yticklabels(tasks_plot, fontsize=12)
ax.set_xlabel("Week", fontsize=18, fontweight="bold")
ax.set_ylabel("Project Stages", fontsize=18, fontweight="bold")
ax.set_title("Project Timeline - Total Project Duration: 15 weeks", fontsize=22, fontweight="bold", pad=20)

# X axis formatting
ax.set_xlim(0, 19)
ax.set_xticks(range(1, 14))
ax.tick_params(axis="x", labelsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Keep visible spines
for spine in ax.spines.values():
    spine.set_color("black")

plt.tight_layout()
plt.show()

# Optional save
# plt.savefig("project_timeline.png", dpi=300, bbox_inches="tight")
