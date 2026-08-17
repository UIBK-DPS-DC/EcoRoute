import re
from glob import glob
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from plots import (
    get_metrics_query,
    plot_basic_metrics_scatter,
    plot_model_utilization,
    plot_output_quality_analysis,
    plot_task_classification,
)


def get_log_entries(logfile: str):
    pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
        r"\s+\|\s+INFO\s+\|\s+Routed\s+"
        r"(?P<id>[a-f0-9-]+)\s+to\s+"
        r"(?P<location>\S+)"
    )

    rows = []

    with open(logfile) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rows.append(match.groupdict())

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f")

    print(df.head())
    return df


site = "edge"

directory = f"../../deployed-db/{site}"

base_dir = Path(directory)
pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")

l = sorted([d for d in base_dir.iterdir() if d.is_dir() and pattern.search(d.name)])


df_to_select = 1

latest_dir = max(
    (d for d in base_dir.iterdir() if d.is_dir() and pattern.search(d.name)),
    key=lambda d: pattern.search(d.name).group(1),
    default=None,
)

if df_to_select is not None:
    latest_dir = l[-df_to_select]

print(latest_dir)

con = duckdb.connect(f"{latest_dir}/metrics-router-{site}.duckdb")

query = """
SELECT *
FROM metrics m
JOIN reference r ON m.query_id = r.query_id
"""

query = """
SELECT *
FROM reference
"""

query = f"""
SELECT
            m.task,
            AVG(m.reward) AS mean_reward, 
            AVG(m.response_time) AS mean_response_time,
            AVG(m.energy) AS mean_energy
        FROM llm l
        JOIN metrics m USING (query_id)
        WHERE m.processed = true AND m.site = '{site}'
        GROUP BY l.id, m.task
"""


# """
#     CREATE TABLE IF NOT EXISTS metrics (
#         query_id VARCHAR,
#         timestamp TIMESTAMP,
#         llm_id VARCHAR,
#         site VARCHAR,
#         task VARCHAR,
#         response VARCHAR,
#         reward DOUBLE DEFAULT NULL,
#         response_time DOUBLE,
#         network_time DOUBLE,
#         routing_time DOUBLE,
#         execution_time DOUBLE,
#         energy DOUBLE,
#         routing_confidence DOUBLE,
#         processed BOOLEAN DEFAULT FALSE
#     )
#     """
# query_id, timestamp, llm_id, site, task, response, reward, response_time, network_time, routing_time, execution_time, energy, routing_confidence


# query = f"""
# SELECT *
# FROM metrics m
# ORDER BY m.timestamp desc
# """
# query = f"""
# SELECT
#     m.task,
#     r.dataset
# FROM metrics m
# JOIN reference r using (query_id)
# WHERE m.processed = true AND m.site = '{site}'
# """


datasets = {
    "squad": "qa",
    "cnn_dailymail": "summarization",
    "xsum": "summarization",
    "glue_mnli": "classification",
    "glue_qqp": "classification",
    "glue_sst2": "classification",
    "mbpp": "coding",
    "gsm8k": "reasoning",
    "natural_questions": "qa",
}
# query = "SHOW DATABASES;"
# query = "SELECT current_database();"
# query = "SELECT * FROM duckdb_tables();"
# query = "SHOW TABLES;"

query = get_metrics_query()

df = con.execute(query).fetchdf()

# print(df['reward'].idxmax())
print(df.iloc[df["reward"].idxmax()])

plot_basic_metrics_scatter(df)
# plot_output_quality_analysis(df)
# plot_task_classification(df)
plot_model_utilization(df)
assert False

# c_not_found = 0
# c_found = 0
# for idx_tacc, log in logentries.iterrows():
#     query_id = log["id"]
#     found = False
#     for idx_edge, row in df.iterrows():
#         if(row["query_id"] == query_id):
#             difference = (row['timestamp'] - log['timestamp']).seconds / 60
#             # print(f"{row_edge['timestamp']} | {row_tacc['timestamp']} | {difference}")
#             found = True
#             c_found += 1
#             print(f"Found {query_id} | {difference}")
#     if(not found):
#         c_not_found += 1
#         # print(f"Not found for {query_id} routed to {log['location']}")
# print(f"{c_found}/{c_not_found} entries found/not found")

# assert False


# print(df)
print(df[["query_id", "timestamp", "llm_id", "task", "network_time"]])

# print(df["llm_id"].unique())
# assert False

con = duckdb.connect(f"{directory}/2026-07-08_18-13-55/metrics-router-{site}.duckdb")

df_1_per_sec = con.execute(query).fetchdf()

import matplotlib.pyplot as plt
import pandas as pd

# Convert timestamp to datetime if needed
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_1_per_sec["timestamp"] = pd.to_datetime(df_1_per_sec["timestamp"])

# Sort by time
df = df.sort_values("timestamp")
df_1_per_sec = df_1_per_sec.sort_values("timestamp")

fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

# # First right axis
# ax_quality = axes[0].twinx()

# # Second right axis
# ax_norm_quality = axes[0].twinx()
# ax_norm_quality.spines["right"].set_position(("outward", 60))

for llm_id, group in df.groupby("llm_id"):
    axes[0].scatter(group["timestamp"], group["reward"], label=llm_id, alpha=0.7)
    axes[1].scatter(group["timestamp"], group["output_quality"], alpha=0.7)
    axes[2].scatter(group["timestamp"], group["normalized_output_quality"], alpha=0.7)
    axes[3].scatter(group["timestamp"], group["response_time"], alpha=0.7)
    axes[4].scatter(group["timestamp"], group["energy"], alpha=0.7)
    axes[5].scatter(group["timestamp"], group["network_time"], alpha=0.7)

# # Plot output quality metrics on the right axis
# ax_quality.scatter(
#     df["timestamp"],
#     df["output_quality"],
#     color="black",
#     marker="v",
#     label="Output Quality"
# )

# ax_norm_quality.scatter(
#     df["timestamp"],
#     df["normalized_output_quality"],
#     color="red",
#     marker="x",
#     label="Normalized Output Quality"
# )
# ax_quality.set_ylabel("Output Quality", color="black")
# ax_norm_quality.set_ylabel("Normalized Output Quality", color="red")
# # Match tick colors
# ax_quality.tick_params(axis="y", colors="black")
# ax_norm_quality.tick_params(axis="y", colors="red")

# handles, labels = [], []
# for ax in [axes[0], ax_quality, ax_norm_quality]:
#     h, l = ax.get_legend_handles_labels()
#     handles.extend(h)
#     labels.extend(l)

# axes[0].legend(handles, labels, loc="upper left")

# axes[0].legend(
#     handles, labels, loc="upper left"
# )

# Reward
# axes[0].scatter(df["timestamp"], df["reward"])
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward over Time")
axes[0].grid(True)
axes[0].legend(title="LLM")

axes[1].set_ylabel("Output quality")
axes[1].set_title("Output quality over Time")
axes[1].grid(True)

axes[2].set_ylabel("Normalized output quality")
axes[2].set_title("Normalized output quality over Time")
axes[2].grid(True)


# Response time
# axes[1].scatter(df["timestamp"], df["response_time"])
axes[3].set_ylabel("Response Time (s)")
axes[3].set_title("Response Time over Time")
axes[3].grid(True)

# Energy
# axes[2].scatter(df["timestamp"], df["energy"])
axes[4].set_ylabel("Energy")
axes[4].set_xlabel("Time")
axes[4].set_title("Energy over Time")
axes[4].grid(True)

plt.tight_layout()
# plt.title(f"Queries per second: {run_config["queries_per_second"]}")
plt.show()

# =====================================================================================


# df["timestamp"] = pd.to_datetime(df["timestamp"])

# agg = (
#     df.set_index("timestamp")
#       .resample("0.5min")[["reward", "response_time", "energy"]]
#       .mean()
# )

# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# for ax, col in zip(axes, agg.columns):
#     ax.plot(agg.index, agg[col])
#     ax.set_title(col.replace("_", " ").title())
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

# =====================================================================================


# df["timestamp"] = pd.to_datetime(df["timestamp"])

# # Count requests per minute (change to '5min', '1H', etc. as needed)
# utilization = (
#     df.set_index("timestamp")
#       .groupby("llm_id")
#       .resample("1min")
#       .size()
#       .unstack(level=0, fill_value=0)
# )

# utilization.plot(figsize=(12, 6))

# plt.xlabel("Time")
# plt.ylabel("Number of Requests")
# plt.title("LLM Utilization Over Time")
# plt.grid(True)
# plt.legend(title="LLM")
# plt.tight_layout()
# plt.show()


# utilization.plot.area(figsize=(12, 6))

# plt.xlabel("Time")
# plt.ylabel("Requests")
# plt.title("LLM Utilization Over Time")
# plt.tight_layout()
# plt.show()


# utilization_pct = utilization.div(utilization.sum(axis=1), axis=0)

# utilization_pct.plot.area(figsize=(12, 6))

# plt.ylabel("Fraction of Requests")
# plt.xlabel("Time")
# plt.title("LLM Routing Share Over Time")
# plt.ylim(0, 1)
# plt.tight_layout()
# plt.show()


# utilization = (
#     df.set_index("timestamp")
#       .groupby("llm_id")["execution_time"]
#       .resample("1min")
#       .sum()
#       .unstack(level=0, fill_value=0)
# )

# utilization.plot(figsize=(12, 6))
# plt.ylabel("Execution Time (s)")
# plt.title("LLM Compute Utilization")
# plt.grid(True)
# plt.show()

# =====================================================================================

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Count requests
# task_llm = pd.crosstab(df["task"], df["llm_id"])

# plt.figure(figsize=(10, 6))
# sns.heatmap(task_llm, annot=True, fmt="d", cmap="Blues")

# plt.xlabel("LLM")
# plt.ylabel("Task")
# plt.title("Number of Tasks Executed by Each LLM")
# plt.show()

# =====================================================================================

# # Aggregate by task
# task_metrics = (
#     df.groupby("task")[["reward", "response_time", "energy"]]
#       .mean()      # or .median()
#       .sort_values("reward", ascending=False)
# )

# fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# # Reward
# task_metrics["reward"].plot(kind="bar", ax=axes[0])
# axes[0].set_title("Average Reward by Task")
# axes[0].set_ylabel("Reward")
# axes[0].tick_params(axis="x", rotation=45)

# # Response Time
# task_metrics["response_time"].plot(kind="bar", ax=axes[1])
# axes[1].set_title("Average Response Time by Task")
# axes[1].set_ylabel("Seconds")
# axes[1].tick_params(axis="x", rotation=45)

# # Energy
# task_metrics["energy"].plot(kind="bar", ax=axes[2])
# axes[2].set_title("Average Energy by Task")
# axes[2].set_ylabel("Energy")
# axes[2].tick_params(axis="x", rotation=45)

# plt.tight_layout()
# plt.show()

# =====================================================================================


task_metrics = pd.concat(
    {
        "1 query per second": df_1_per_sec.groupby("task").mean(numeric_only=True),
        "5 query per second": df.groupby("task").mean(numeric_only=True),
    }
)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, metric in zip(axes, ["reward", "response_time", "energy"]):
    task_metrics[metric].unstack(0).plot(kind="bar", ax=ax)
    ax.set_title(f"Average {metric.replace('_', ' ').title()} by Task")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# for ax, metric in zip(axes, ["reward", "response_time", "energy"]):
#     pivot = (
#         df.set_index("timestamp")
#           .groupby("llm_id")[metric]
#           .resample("1min")
#           .mean()
#           .unstack(0)
#     )

#     pivot.plot(ax=ax)

#     ax.set_title(f"{metric.replace('_', ' ').title()} Over Time by LLM")
#     ax.set_ylabel(metric.replace("_", " ").title())
#     ax.grid(True, alpha=0.3)

# axes[-1].set_xlabel("Time")
# plt.tight_layout()
# plt.show()


# =====================================================================================


# avg = (
#     df.groupby("llm_id")[["network_time", "routing_time", "execution_time"]]
#       .mean()
# )

# ax = avg.plot(
#     kind="bar",
#     stacked=True,
#     figsize=(10, 6)
# )

# # Plot total response time as black dots
# response = df.groupby("llm_id")["response_time"].mean()
# ax.scatter(
#     range(len(response)),
#     response,
#     color="black",
#     marker="o",
#     s=80,
#     label="Response Time"
# )

# ax.set_ylabel("Time (s)")
# ax.set_title("Average Response Time Breakdown by LLM")
# ax.legend()
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df["expected"] = df["dataset"].map(datasets)
# # Count requests
# cm = pd.crosstab(
#     df["expected"],
#     df["task"],
#     rownames=["Expected"],
#     colnames=["Predicted"]
# )

# plt.figure(figsize=(10, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

# # plt.xlabel("Expected task")
# # plt.ylabel("Actual task")
# plt.title("Task classification")
# plt.show()

import seaborn as sns

plt.figure(figsize=(14, 6))

sns.boxplot(data=df, x="task", y="output_quality", hue="llm_id")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
