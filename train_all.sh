#!/usr/bin/env bash
# Train an encoder for every combination of sampler, max size, alphabet size, reward type and p
# (3 x 2 x 2 x 2 x 2 = 48 runs). Arguments are passed on to every train.py call, e.g.
#
#   ./train_all.sh --save-dir storage --seed 0 --gamma 0.9
#
# Runs that already have a checkpoint are skipped (train.py refuses to overwrite them unless --overwrite
# is passed), so re-running the script after an interruption resumes the sweep.

set -o pipefail
cd "$(dirname "$0")"

samplers=(RAD R RA)
max_sizes=(10 5)
n_tokens_list=(10 5)
binary_rewards=(False True)
ps=(0.5 None)

total=$(( ${#samplers[@]} * ${#max_sizes[@]} * ${#n_tokens_list[@]} * ${#binary_rewards[@]} * ${#ps[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for sampler in "${samplers[@]}"; do
    for max_size in "${max_sizes[@]}"; do
        for n_tokens in "${n_tokens_list[@]}"; do
            for binary_reward in "${binary_rewards[@]}"; do
                for p in "${ps[@]}"; do
                    i=$((i + 1))
                    run="sampler=$sampler max_size=$max_size n_tokens=$n_tokens binary_reward=$binary_reward p=$p"
                    echo "=== [$i/$total] $run"

                    args=(--sampler "$sampler" --max-size "$max_size" --n-tokens "$n_tokens" --p "$p")
                    if [ "$binary_reward" = True ]; then
                        args+=(--binary-reward)
                    fi

                    if uv run train.py "${args[@]}" "$@" 2>&1 | tee "$out"; then
                        n_trained=$((n_trained + 1))
                    elif grep -q "FileExistsError" "$out"; then
                        echo "--- already trained, skipping"
                        n_skipped=$((n_skipped + 1))
                    else
                        failed+=("$run")
                    fi
                done
            done
        done
    done
done

echo
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
