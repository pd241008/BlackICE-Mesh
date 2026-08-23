#!/bin/bash
# Periodic health monitor for the seed sweep. Appends a timestamped snapshot
# to logs/watchdog.log every CHECK_INTERVAL seconds and flags:
#   - training process death before the sweep is complete
#   - OOM strings (CUDA out of memory / Killed / RuntimeError) in job logs
#   - low free RAM or GPU memory pressure
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
TOTAL_RUNS=27   # 9 seeds x 3 methods

count_ckpts() {
    ls models/unified/model_adv_{hardened,curriculum,rsc}_cicids2017_seed{46,47,48,49,50,51,52,53,54}.pth 2>/dev/null | wc -l
}

while true; do
    ts="$(date '+%F %T')"
    ckpts=$(count_ckpts)
    train_pid=$(pgrep -f "train_unified.py" | head -1)
    eval_pid=$(pgrep -f "eval_unified.py" | head -1)

    ram_line=$(free -m | awk '/^Mem:/ {printf "%dMB used / %dMB total (%d free)", $3, $2, $4}')
    gpu_line=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")

    # Newest activity line from whatever job log is freshest.
    newest_log=$(ls -t logs/train_*.log logs/eval_seed*.log 2>/dev/null | head -1)
    last_line=""
    [ -n "$newest_log" ] && last_line=$(tail -n 1 "$newest_log" | cut -c1-120)

    # Scan today's job logs for fatal patterns (only report the newest hit).
    err_hit=""
    for f in $(ls -t logs/train_*.log logs/eval_seed*.log 2>/dev/null); do
        err_hit=$(grep -m1 -E "CUDA out of memory|RuntimeError|Killed|Cannot allocate memory" "$f" 2>/dev/null)
        if [ -n "$err_hit" ]; then
            err_hit="IN ${f}: ${err_hit}"
            break
        fi
    done

    {
        echo "=== ${ts} ==="
        echo "checkpoints: ${ckpts}/${TOTAL_RUNS} | RAM: ${ram_line} | GPU: ${gpu_line}"
        if [ -n "$train_pid" ]; then
            rss=$(ps -o rss= -p "$train_pid" | tr -d ' ')
            echo "training PID ${train_pid} alive (RSS ${rss}KB) | last: ${last_line}"
        elif [ -n "$eval_pid" ]; then
            echo "evaluating PID ${eval_pid} alive | last: ${last_line}"
        else
            if [ "$ckpts" -ge "$TOTAL_RUNS" ]; then
                echo "no training process — sweep appears COMPLETE"
            else
                echo "*** no train/eval/driver process with ${ckpts}/${TOTAL_RUNS} checkpoints — RESPAWNING driver ***"
                setsid nohup bash scripts/run_seeds_46_54.sh < /dev/null >> logs/sweep_driver.log 2>&1 &
                echo "[WATCHDOG] ${ts} respawned sweep driver (PID $!)"
            fi
        fi
        [ -n "$err_hit" ] && echo "*** ERROR DETECTED ${err_hit} ***"
        echo ""
    } >> logs/watchdog.log

    sleep "$CHECK_INTERVAL"
done
