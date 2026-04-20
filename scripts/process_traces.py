"""
process_traces.py
==============================================================
Convert production cluster traces into the (inter_arrival_time,
duration) .dat format that the C++ simulator reads via --trace.

Two sources are supported :

  1. Google Borg cluster trace, 2019 release.
     The simulator treats each task's SCHEDULE event as its
     arrival and each FINISH event as the end of its service
     period; the duration is the difference between the two.

  2. Alibaba cluster trace, 2025 disaggregated DLRM release.
     Has named columns (creation_time, scheduled_time,
     deletion_time); arrival = creation_time, duration =
     deletion_time - scheduled_time.

Both parsers emit the same plain-text format, one job per line:

    <inter_arrival_time> <duration>

which is what the simulator's trace-reading code in
Simulation::load_trace() expects.

Usage :
    python scripts/process_traces.py --source google  IN.csv OUT.dat
    python scripts/process_traces.py --source alibaba IN.csv OUT.dat
    python scripts/process_traces.py --source google  IN.csv OUT.dat --limit 200000
"""

import argparse
import csv
from pathlib import Path
import pandas as pd

# ==========================================================
# GOOGLE BORG TRACE (2019)
# ==========================================================
# Event codes from the Borg v2.1 schema. Arrival is SCHEDULE,
# service completion is FINISH; every other event type is
# ignored.
EVENT_SUBMIT   = 0
EVENT_SCHEDULE = 1    # we treat this as the task's "arrival"
EVENT_EVICT    = 2
EVENT_FAIL     = 3
EVENT_FINISH   = 4    # and this as the task's "completion"
EVENT_KILL     = 5
EVENT_LOST     = 6


def process_google(input_csv, output_dat, max_tasks=100_000):
    """
    Stream through a Google Borg task_events CSV, pair SCHEDULE
    with FINISH events by (job_id, task_idx), and write at most
    max_tasks (inter_arrival, duration) pairs in seconds.
    """
    print(f"Reading {input_csv}...")

    # (job_id, task_idx) -> schedule_timestamp_in_microseconds
    active_tasks = {}

    # (start_time_sec, duration_sec) for every fully-paired task
    valid_tasks = []

    with open(input_csv, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            try:
                # Schema : [timestamp, missing, job_id, task_idx, ..., event_type, ...]
                timestamp_us = float(row[0])
                job_id       = row[2]
                task_idx     = row[3]
                event        = int(row[5])

                key = (job_id, task_idx)

                # SCHEDULE : remember when this task started running
                if event == EVENT_SCHEDULE:
                    active_tasks[key] = timestamp_us

                # FINISH : pair with the matching SCHEDULE, if any
                elif event == EVENT_FINISH and key in active_tasks:
                    start_us    = active_tasks.pop(key)
                    duration_us = timestamp_us - start_us

                    if duration_us > 0:
                        start_sec    = start_us    / 1_000_000.0
                        duration_sec = duration_us / 1_000_000.0
                        valid_tasks.append((start_sec, duration_sec))

                        if len(valid_tasks) >= max_tasks:
                            break
            except (ValueError, IndexError):
                # Skip malformed rows silently
                continue

    print(f"Found {len(valid_tasks)} paired SCHEDULE/FINISH tasks.")
    if not valid_tasks:
        print("Error : no usable tasks found.")
        return

    # The raw Borg file is roughly but not perfectly sorted by
    # start time, so sort explicitly. Otherwise the inter-arrival
    # computation below could produce negative gaps.
    print("Sorting tasks by start time...")
    valid_tasks.sort(key=lambda x: x[0])

    # Shift time so the first job arrives at t = 0.
    write_dat_file(output_dat, valid_tasks)


# ==========================================================
# ALIBABA DLRM TRACE (2025)
# ==========================================================

def process_alibaba(input_csv, output_dat):
    """
    Parse the disaggregated DLRM CSV (which has named columns)
    and write the .dat file. Uses pandas since the trace is
    small enough to fit in memory and named-column access
    keeps the code short.
    """
       

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Drop rows missing any of the three timestamps we rely on
    df = df.dropna(subset=["creation_time", "scheduled_time", "deletion_time"])

    # Arrival  = when the job appeared in the system
    # Duration = how long it actually ran on a server
    df["arrival"]  = df["creation_time"]
    df["duration"] = df["deletion_time"] - df["scheduled_time"]

    # Throw away zero / negative durations (malformed rows)
    df = df[df["duration"] > 0]

    # Sort by arrival so inter-arrival times come out non-negative
    df = df.sort_values("arrival")

    # List of (arrival_time, duration) pairs, in arrival order
    valid_tasks = list(zip(df["arrival"].tolist(),
                           df["duration"].tolist()))

    write_dat_file(output_dat, valid_tasks)


# ==========================================================
# COMMON .DAT WRITER
# ==========================================================

def write_dat_file(output_dat, valid_tasks):
    """
    valid_tasks is a list of (arrival_time, duration) pairs
    sorted by arrival. Compute inter-arrival times and write
    one "<dt> <duration>" line per job.
    """
    out_path = Path(output_dat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(valid_tasks)} jobs to {out_path}...")
    with open(out_path, "w") as out:
        # First job: inter-arrival = 0 (it's the start of the trace)
        first_arrival, first_duration = valid_tasks[0]
        out.write(f"0.000000 {first_duration:.6f}\n")

        previous_arrival = first_arrival
        for arrival, duration in valid_tasks[1:]:
            dt = arrival - previous_arrival
            if dt < 0:            # floating-point safety net
                dt = 0.0
            out.write(f"{dt:.6f} {duration:.6f}\n")
            previous_arrival = arrival

    print("Done. Trace is ready.")


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Google or Alibaba cluster traces into the "
                    "simulator's .dat trace format.")
    parser.add_argument("--source", choices=["google", "alibaba"], required=True,
                        help="Which trace format the input CSV uses")
    parser.add_argument("input_csv",  help="Path to the raw trace CSV")
    parser.add_argument("output_dat", help="Path to write the .dat trace")
    parser.add_argument("--limit", type=int, default=100_000,
                        help="(Google only) maximum number of tasks to extract")
    args = parser.parse_args()

    if args.source == "google":
        process_google(args.input_csv, args.output_dat, args.limit)
    else:
        process_alibaba(args.input_csv, args.output_dat)


if __name__ == "__main__":
    main()
