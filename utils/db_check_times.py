import duckdb
import re
from glob import glob
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.dates as mdates

def compute_time(df, col_later, col_earlier):
    return (df[col_later] - df[col_earlier]).dt.seconds





site = "edge"


directory = f"../deployed-db/{site}"

base_dir = Path(directory)
pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")

l = sorted([d for d in base_dir.iterdir() if d.is_dir() and pattern.search(d.name)])



df_to_select = 1

latest_dir = max(
    (
        d for d in base_dir.iterdir()
        if d.is_dir() and pattern.search(d.name)
    ),
    key=lambda d: pattern.search(d.name).group(1),
    default=None,
)

if(df_to_select is not None):
    latest_dir = l[-df_to_select]

print(latest_dir)

with open(f"{latest_dir}/run_config.yml", "r") as f:
    run_config = yaml.safe_load(f)



con = duckdb.connect(f"{latest_dir}/metrics-router-{site}.duckdb")


# CREATE TABLE IF NOT EXISTS request_time (
#         query_id VARCHAR,
#         timestamp TIMESTAMP,
#         llm_id VARCHAR,
#         request_received TIMSTAMP DEFAULT NULL,
#         request_published TIMSTAMP DEFAULT NULL,
#         model_message_received TIMSTAMP DEFAULT NULL,
#         model_inference_started TIMSTAMP DEFAULT NULL,
#         model_inference_finished TIMSTAMP DEFAULT NULL,
#         model_response_published TIMSTAMP DEFAULT NULL,
#         router_response_received TIMSTAMP DEFAULT NULL,
#         router_future_completed TIMSTAMP DEFAULT NULL,   
#     )


query = f"""
SELECT *
FROM request_time t
ORDER BY t.timestamp desc
"""


df = con.execute(query).fetchdf()


import pandas as pd
import matplotlib.pyplot as plt

# Convert timestamp to datetime if needed
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(df[["model_message_received", "model_inference_started", "model_inference_finished", "model_response_published"]])

for col in ["timestamp", "request_received", "request_published", "model_message_received", "model_inference_started", "model_inference_finished", "model_response_published", "router_response_received", "router_future_completed", ]:
    df[col] = pd.to_datetime(df[col])

print(df)

df["nats_request_delay"] = (df["model_message_received"] - df["request_published"]).dt.seconds
df["model_waiting_delay"] = (df["model_inference_started"] - df["model_message_received"]).dt.seconds
df["inference_time"] = (df["model_inference_finished"] - df["model_inference_started"]).dt.seconds
df["nats_response_delay"] = (df["router_response_received"] - df["model_response_published"]).dt.seconds
df["response_time"] = compute_time(df, "router_future_completed", "request_received")
# Sort by time
df = df.sort_values("timestamp")

fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)


for llm_id, group in df.groupby("llm_id"):
    axes[0].scatter(group["timestamp"], group["nats_request_delay"], label=llm_id, alpha=0.7)
    axes[1].scatter(group["timestamp"], group["model_waiting_delay"], alpha=0.7)
    axes[2].scatter(group["timestamp"], group["inference_time"], alpha=0.7)
    axes[3].scatter(group["timestamp"], group["nats_response_delay"], alpha=0.7)
    axes[4].scatter(group["timestamp"], group["response_time"], alpha=0.7)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))

    # Optional: format the timestamps
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # Draw gridlines at the major ticks
    ax.grid(axis="x", which="major")

axes[0].set_ylabel("Delay (s)")
axes[0].set_title("NATS request delay")
axes[0].grid(True)
axes[0].legend(title="LLM")

axes[1].set_ylabel("Model wait delay (s)")
axes[1].set_title("Model wait delay")
axes[1].grid(True)

axes[2].set_ylabel("Inference time (s)")
axes[2].set_title("Inference time")
axes[2].grid(True)

axes[3].set_ylabel("NATS response delay (s)")
axes[3].set_title("NATS response delay")
axes[3].grid(True)

axes[4].set_ylabel("Response time (s)")
axes[4].set_title("Response time")
axes[4].grid(True)

fig.autofmt_xdate()

plt.tight_layout()
# plt.title(f"Queries per second: {run_config["queries_per_second"]}")
plt.show()
