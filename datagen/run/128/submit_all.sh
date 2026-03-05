#!/bin/bash
# Submit the next pending batch of 400 volumes (one batch at a time).
# MaxSubmit=500 per user (normal QOS) → 400-task batch fits safely.
# Re-run this script after each batch finishes to queue the next one.
#
# Batches: 0–399 | 400–799 | 800–1199 | 1200–1599 | 1600–1999
# Usage:   ./submit_all.sh          # submits only the next needed batch
#          ./submit_all.sh --all    # chains all remaining batches (careful: counts toward MaxSubmit)

SCRIPT="$(dirname "$0")/gen_labels_array.sbatch"
BASE_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels"

BATCHES=("0 399" "400 799" "800 1199" "1200 1599" "1600 1999")

count_done() {
    local lo=$1 hi=$2 n=0
    for i in $(seq $lo $hi); do
        local s; if [[ $i -lt 1700 ]]; then s="train"; elif [[ $i -lt 1900 ]]; then s="val"; else s="test"; fi
        [[ -f "$BASE_DIR/$s/vol$(printf '%04d' $i)_label.nii.gz" ]] && ((n++))
    done
    echo $n
}

ALL_MODE=0; [[ "${1:-}" == "--all" ]] && ALL_MODE=1
PREV_JID=""

for BATCH in "${BATCHES[@]}"; do
    LO=${BATCH%% *}; HI=${BATCH##* }
    TOTAL=$(( HI - LO + 1 ))
    DONE=$(count_done $LO $HI)

    if [[ $DONE -eq $TOTAL ]]; then
        echo "Batch $LO–$HI: complete ($DONE/$TOTAL) — skipping"; continue
    fi

    echo "Batch $LO–$HI: $DONE/$TOTAL done — submitting..."
    DEP_FLAG=""; [[ -n "$PREV_JID" ]] && DEP_FLAG="--dependency=afterany:$PREV_JID"

    JID=$(sbatch --array=${LO}-${HI}%50 $DEP_FLAG "$SCRIPT" | awk '{print $NF}')
    if [[ -z "$JID" ]]; then echo "  ERROR: submission failed"; exit 1; fi
    echo "  → job $JID  (${TOTAL} tasks, 50 at a time)"

    if [[ $ALL_MODE -eq 0 ]]; then
        echo ""
        echo "Re-run './submit_all.sh' after job $JID finishes to queue the next batch."
        echo "Monitor: squeue -u $USER --format='%.10i %.8T %.5C %j'"
        exit 0
    fi
    PREV_JID="$JID"
done

echo ""; echo "All batches submitted."
echo "Count done: find $BASE_DIR -name '*_label.nii.gz' | wc -l"
