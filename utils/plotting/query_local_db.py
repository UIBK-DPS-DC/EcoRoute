import argparse
import re
from glob import glob
from pathlib import Path
from pprint import pprint

import duckdb
import matplotlib.dates as mdates
import numpy as np
from plots import (
    get_metrics_query,
    plot_basic_metrics_scatter,
    plot_combined,
    plot_model_utilization,
    plot_moving_avg,
    plot_output_quality_analysis,
    plot_task_classification,
    plot_task_model_utilization,
)


def compute_time(df, col_later, col_earlier):
    return (df[col_later] - df[col_earlier]).dt.seconds


parser = argparse.ArgumentParser(prog="Trace generator")

parser.add_argument("directory", type=str, help="directory of DuckDB database files")

parser.add_argument(
    "-s",
    "--site",
    type=str,
    default="uc",
)

args = parser.parse_args()

con = duckdb.connect(f"./local/tmp/{args.directory}/metrics-router-{args.site}.duckdb")

query = get_metrics_query()

df = con.execute(query).fetchdf()


plot_combined(df)
