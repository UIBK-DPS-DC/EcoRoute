import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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
    SELECT m.*, t.request_received as req_rec, r.dataset, r.subset, l.pending as pending
    FROM metrics m
    JOIN request_time t using (query_id)
    
    LEFT JOIN reference r using (query_id)
    LEFT JOIN llm l ON m.query_id = l.query_id AND m.llm_id = l.id
    ORDER BY t.request_received desc
    """

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


def plot_moving_avg(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    agg = (
        df.set_index("timestamp")
        .groupby("task")[["reward", "response_time", "energy"]]
        .resample("0.5min")
        .mean()
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for task in df["task"].dropna().unique():
        task_data = agg.loc[task]

        for ax, col in zip(axes, ["reward", "response_time", "energy"]):
            ax.plot(task_data.index, task_data[col], label=task)

    for ax, col in zip(axes, ["reward", "response_time", "energy"]):
        ax.set_title(col.replace("_", " ").title())
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # for ax, col in zip(axes, agg.columns):
    #     ax.plot(agg.index, agg[col])
    #     ax.set_title(col.replace("_", " ").title())
    #     ax.grid(True)

    # plt.tight_layout()
    # plt.show()


def plot_output_quality_analysis(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x="task", y="output_quality", hue="llm_id", showmeans=True)

    sns.stripplot(
        data=df,
        x="task",
        y="output_quality",
        hue="llm_id",
        dodge=True,
        color="black",
        alpha=0.4,
        jitter=True,
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


def plot_task_model_utilization(df):
    # Count requests
    task_llm = pd.crosstab(df["task"], df["llm_id"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(task_llm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("LLM")
    plt.ylabel("Task")
    plt.title("Number of Tasks Executed by Each LLM")
    plt.show()


def plot_combined(df):
    # Convert timestamp to datetime if needed
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["req_rec"] = pd.to_datetime(df["req_rec"])
    df["elapsed"] = df["req_rec"] - df["req_rec"].min()
    df["elapsed"] = df["elapsed"].dt.total_seconds()
    # df["llm_id"] = df["llm_id"].str[:11]

    # Sort by time
    df = df.sort_values("req_rec")

    # fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    fig = plt.figure(figsize=(20, 16))

    nrows = 5
    ncols = 3

    gs = fig.add_gridspec(
        nrows, ncols, width_ratios=[2, 1, 1], height_ratios=[1, 1, 1, 1, 3]
    )

    axes = np.empty((nrows, ncols), dtype=object)

    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[2, 0] = fig.add_subplot(gs[2, 0])
    axes[3, 0] = fig.add_subplot(gs[3, 0])
    axes[4, 0] = fig.add_subplot(gs[4, :])

    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[2, 1] = fig.add_subplot(gs[2, 1])
    axes[3, 1] = fig.add_subplot(gs[3, 1])

    axes[0, 2] = fig.add_subplot(gs[:2, 2])
    # axes[2, 2] = fig.add_subplot(gs[2, 2])
    axes[1, 2] = fig.add_subplot(gs[2:4, 2])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Assign one color per llm_id
    llm_ids = sorted(df["llm_id"].unique())
    color_map = {llm_id: colors[i % len(colors)] for i, llm_id in enumerate(llm_ids)}

    markers = ["o", "v"]
    marker_size = 14
    alphas = [0.7, 0.3]
    labels = ["successful query", "failed query"]
    dfs = [df[df["response"].notna()], df[df["response"].isna()]]

    for _df, marker, alpha, label in zip(dfs, markers, alphas, labels):
        for llm_id, group in _df.groupby("llm_id"):
            color = color_map[llm_id]

            axes[0, 0].scatter(
                group["elapsed"],
                group["reward"],
                marker=marker,
                color=color,
                alpha=alpha,
                s=marker_size,
            )
            axes[1, 0].scatter(
                group["elapsed"],
                group["output_quality"],
                marker=marker,
                color=color,
                alpha=alpha,
                s=marker_size,
            )
            axes[2, 0].scatter(
                group["elapsed"],
                group["response_time"],
                marker=marker,
                color=color,
                alpha=alpha,
                s=marker_size,
            )
            axes[3, 0].scatter(
                group["elapsed"],
                group["energy"],
                marker=marker,
                color=color,
                alpha=alpha,
                label=f"{llm_id} {label}",
                s=marker_size,
            )
            axes[1, 1].scatter(
                group["elapsed"],
                group["pending"],
                marker=marker,
                color=color,
                alpha=alpha,
                s=marker_size,
            )

    # Reward
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Reward over Time")
    axes[0, 0].grid(True)

    axes[1, 0].set_ylabel("Output quality")
    axes[1, 0].set_title("Output quality over Time")
    axes[1, 0].grid(True)

    # Response time
    axes[2, 0].set_ylabel("Response Time (s)")
    axes[2, 0].set_title("Response Time over Time")
    axes[2, 0].grid(True)

    # Energy
    axes[3, 0].set_ylabel("Energy")
    axes[3, 0].set_xlabel("Time")
    axes[3, 0].set_title("Energy over Time")
    axes[3, 0].legend(title="LLM")
    axes[3, 0].grid(True)

    axes[1, 1].set_ylabel("Pending requests")
    axes[1, 1].set_title("Pending requests over Time")
    axes[1, 1].grid(True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    agg_total = (
        df.set_index("timestamp")
        .groupby("task")[["elapsed", "reward", "response_time", "energy"]]
        .resample("0.5min")
        .mean()
    )

    window_size = int((len(df["reward"]) // df["elapsed"].max()) * 30)
    print(f"{window_size=}")

    axes[0, 1].plot(
        df["elapsed"], df["reward"].rolling(window=window_size).mean(), label="total"
    )
    axes[2, 1].plot(
        df["elapsed"],
        df["response_time"].rolling(window=window_size).mean(),
        label="total",
    )
    axes[3, 1].plot(
        df["elapsed"], df["energy"].rolling(window=window_size).mean(), label="total"
    )

    agg = (
        df.set_index("timestamp")
        .groupby("task")[["elapsed", "reward", "response_time", "energy"]]
        .resample("0.5min")
        .mean()
    )

    for task in df["task"].dropna().unique():
        task_data = agg.loc[task]

        axes[0, 1].plot(task_data["elapsed"], task_data["reward"], label=task)
        axes[0, 1].set_title("Moving AVG reward")
        axes[0, 1].grid(True)
        # axes[1, 1].axis("off")
        axes[2, 1].plot(task_data["elapsed"], task_data["response_time"], label=task)
        axes[2, 1].set_title("Moving AVG response time")
        axes[2, 1].grid(True)
        axes[3, 1].plot(task_data["elapsed"], task_data["energy"], label=task)
        axes[3, 1].set_title("Moving AVG energy")
        axes[3, 1].grid(True)
        axes[3, 1].legend()

    sns.boxplot(
        data=df,
        x="task",
        y="output_quality",
        hue="llm_id",
        palette=color_map,
        showmeans=True,
        ax=axes[4, 0],
    )

    # Vertical separators between tasks
    for x in range(len(df["task"].unique()) - 1):
        axes[4, 0].axvline(x + 0.5, color="gray", linestyle="--", alpha=0.5)

    # sns.stripplot(
    #     data=df,
    #     x="task",
    #     y="output_quality",
    #     hue="llm_id",
    #     dodge=True,
    #     color="black",
    #     alpha=0.4,
    #     jitter=True,
    #     ax=axes[0, 2],
    # )
    axes[4, 0].set_xlabel("Task")
    axes[4, 0].set_ylabel("Output Quality")
    axes[4, 0].set_title("Output Quality by Task")
    axes[4, 0].tick_params(axis="x", labelrotation=45)

    # df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Count requests per minute (change to '5min', '1H', etc. as needed)
    # utilization = (
    #     df.set_index("timestamp")
    #     .groupby("llm_id")
    #     .resample("0.1min")
    #     .size()
    #     .unstack(level=0, fill_value=0)
    # )

    time_frame_min = 0.1

    utilization = (
        df.assign(elapsed_min=(df["elapsed"] // (60 * time_frame_min)).astype(int))
        .groupby(["elapsed_min", "llm_id"])
        .size()
        .unstack(level=1, fill_value=0)
    )

    utilization_pct = utilization.div(utilization.sum(axis=1), axis=0)

    utilization_pct.index = utilization_pct.index * (60 * time_frame_min)

    utilization_pct.plot.area(ax=axes[0, 2])

    axes[0, 2].set_ylabel("Fraction of Requests")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_title("LLM Routing Share Over Time")
    axes[0, 2].set_ylim(0, 1)

    # Count requests
    task_llm = pd.crosstab(df["task"], df["llm_id"])

    sns.heatmap(task_llm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 2])

    axes[1, 2].set_xlabel("LLM")
    axes[1, 2].set_ylabel("Task")
    axes[1, 2].set_title("Number of Tasks Executed by Each LLM")
    axes[1, 2].tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    plt.show()
