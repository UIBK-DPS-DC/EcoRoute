import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset_task_map = {
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


def get_metrics_query():
    return f"""
SELECT m.*, t.request_received as req_rec, r.dataset, r.subset
FROM metrics m
JOIN request_time t using (query_id)
LEFT JOIN reference r using (query_id)
ORDER BY t.request_received desc
"""


def plot_basic_metrics_scatter(df):
    # Convert timestamp to datetime if needed
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["req_rec"] = pd.to_datetime(df["req_rec"])
    df["elapsed"] = df["req_rec"] - df["req_rec"].min()
    df["elapsed"] = df["elapsed"].dt.total_seconds()

    # Sort by time
    df = df.sort_values("req_rec")

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # print(df[df["response"] == None])
    print(df[df["response"] != None])

    # for e in df["response"]:
    #     print(e)

    # assert False

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Assign one color per llm_id
    llm_ids = sorted(df["llm_id"].unique())
    color_map = {llm_id: colors[i % len(colors)] for i, llm_id in enumerate(llm_ids)}

    markers = ["o", "v"]
    alphas = [0.7, 0.3]
    labels = ["successful query", "failed query"]
    dfs = [df[df["response"].notna()], df[df["response"].isna()]]

    for _df, marker, alpha, label in zip(dfs, markers, alphas, labels):
        for llm_id, group in _df.groupby("llm_id"):
            color = color_map[llm_id]

            axes[0].scatter(
                group["elapsed"],
                group["reward"],
                marker=marker,
                color=color,
                alpha=alpha,
            )
            axes[1].scatter(
                group["elapsed"],
                group["output_quality"],
                marker=marker,
                color=color,
                alpha=alpha,
            )
            axes[2].scatter(
                group["elapsed"],
                group["normalized_output_quality"],
                marker=marker,
                color=color,
                label=f"{llm_id} {label}",
                alpha=alpha,
            )
            axes[3].scatter(
                group["elapsed"],
                group["response_time"],
                marker=marker,
                color=color,
                alpha=alpha,
            )
            axes[4].scatter(
                group["elapsed"],
                group["energy"],
                marker=marker,
                color=color,
                alpha=alpha,
            )

    # Reward
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward over Time")
    axes[0].grid(True)

    axes[1].set_ylabel("Output quality")
    axes[1].set_title("Output quality over Time")
    axes[1].grid(True)

    axes[2].set_ylabel("Normalized output quality")
    axes[2].set_title("Normalized output quality over Time")
    axes[2].grid(True)
    axes[2].legend(title="LLM")

    # Response time
    axes[3].set_ylabel("Response Time (s)")
    axes[3].set_title("Response Time over Time")
    axes[3].grid(True)

    # Energy
    axes[4].set_ylabel("Energy")
    axes[4].set_xlabel("Time")
    axes[4].set_title("Energy over Time")
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()


def plot_output_quality_analysis(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x="task", y="output_quality")

    sns.stripplot(
        data=df, x="task", y="output_quality", color="black", alpha=0.4, jitter=True
    )

    plt.xlabel("Task")
    plt.ylabel("Output Quality")
    plt.title("Output Quality by Task")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_task_classification(df):
    df_eval = df[df["dataset"].notna()].copy()
    df_eval["expected_task"] = df_eval["dataset"].map(dataset_task_map)

    df_eval["correct"] = df_eval["task"] == df_eval["expected_task"]

    heatmap_data = pd.crosstab(
        df_eval["dataset"], df_eval["task"], values=df_eval["correct"], aggfunc="mean"
    )

    heatmap_data = pd.crosstab(
        df.loc[df["dataset"].notna(), "dataset"], df.loc[df["dataset"].notna(), "task"]
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
    )

    plt.xlabel("Classified Task")
    plt.ylabel("Dataset")
    plt.title("Task Classification Accuracy")
    plt.tight_layout()
    plt.show()


def plot_model_utilization(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Count requests per minute (change to '5min', '1H', etc. as needed)
    utilization = (
        df.set_index("timestamp")
        .groupby("llm_id")
        .resample("1min")
        .size()
        .unstack(level=0, fill_value=0)
    )

    utilization_pct = utilization.div(utilization.sum(axis=1), axis=0)

    utilization_pct.plot.area(figsize=(12, 6))

    plt.ylabel("Fraction of Requests")
    plt.xlabel("Time")
    plt.title("LLM Routing Share Over Time")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
