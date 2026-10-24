#!/bin/sh
# Demo beat 4: hard-kill the supervisor mid-run, then rerun run_demo.py without --fresh.
pkill -9 -f run_demo.py && echo "supervisor killed (SIGKILL)"
