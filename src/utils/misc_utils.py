import subprocess

import numpy as np


def mk_digraphwave_name(*, kemb, radius, arctan_tranform=True, aggregate=True, transpose=False, normalize=False,
                        num_gpus=0):
    name_extra = f"{kemb}_{radius}_"
    name_extra = name_extra + "a" if aggregate else name_extra
    name_extra = name_extra + "t" if transpose else name_extra
    name_extra = name_extra + "n" if normalize else name_extra
    name_extra = name_extra + "_no_transform" if not arctan_tranform else name_extra
    name_extra = "_" + name_extra if name_extra else ""
    name = f"digraphwave{name_extra}_cpu" if num_gpus == 0 else f"digraphwave{name_extra}_cuda[{num_gpus}]"
    return name


def mk_graphwave_name(*, kemb, radius, arctan_tranform=True, aggregate=True, num_gpus=0):
    name_extra = f"{kemb}_{radius}_"
    name_extra = name_extra + "a" if aggregate else name_extra
    name_extra = name_extra + "_no_transform" if not arctan_tranform else name_extra
    name_extra = "_" + name_extra if name_extra else ""
    name = f"graphwave{name_extra}_cpu" if num_gpus == 0 else f"graphwave{name_extra}_cuda[{num_gpus}]"
    return name


def get_duration_from_stdout(stdout):
    output = stdout.split("\n")
    duration = None
    for line in output[:-4:-1]:  # Check last 3 lines
        if line.lower().startswith("completed"):
            duration = float(line.split("::")[-1])
    return duration


def run_command(command, timeout_time: float):
    error_out = ""
    try:
        result = subprocess.run(command, capture_output=True, universal_newlines=True, timeout=timeout_time)
        if result.returncode != 0:
            outcome = "fail"
            duration = np.nan
            # print(result.returncode, result.stdout, result.stderr)
            if result.stderr:
                error_out = result.stderr.strip().split("\n")[-10:]
                extensionsToCheck = {"memoryerror", "out of memory", "killed"}
                for msg in error_out[::-1]:
                    if any(ext in msg.lower() for ext in extensionsToCheck):
                        outcome = "oom"
                        break
        else:
            duration = get_duration_from_stdout(result.stdout)
            outcome = "completed"
            # print(duration)
    except subprocess.TimeoutExpired:
        # print("timed out")
        outcome = "timeout"
        duration = timeout_time
    return duration, outcome, error_out
