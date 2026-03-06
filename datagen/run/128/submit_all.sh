#!/bin/bash
# Submit the next pending batch, using sparse --array for partial batches.
# Batches: 0–399 | 400–799 | 800–1199 | 1200–1599 | 1600–1999

SCRIPT="$(dirname "$0")/gen_labels_array.sbatch"
BASE_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels"
BATCHES=("0 399" "400 799" "800 1199" "1200 1599" "1600 1999")

get_split() {
    local i=$1
    if [[ $i -lt 1700 ]]; then echo "train"
    elif [[ $i -lt 1900 ]]; then echo "val"
    else echo "test"; fi
}

missing_ids() {
    local lo=$1 hi=$2 ids=""
    for i in $(seq $lo $hi); do
        local s=$(get_split $i)
        if [[ ! -f "$BASE_DIR/$s/vol$(printf '%04d' $i)_label.nii.gz" ]]; then
            ids="${ids}${i},"
        fi
    done
    echo "${ids%,}"  # strip trailing comma
}

ALL_MODE=0; [[ "${1:-}" == "--all" ]] && ALL_MODE=1
PREV_JID=""

for BATCH in "${BATCHES[@]}"; do
    LO=${BATCH%% *}; HI=${BATCH##* }
    TOTAL=$(( HI - LO + 1 ))
    MISSING=$(missing_ids $LO $HI)

    if [[ -z "$MISSING" ]]; then
        echo "Batch $LO–$HI: complete — skipping"; continue
    fi

    N_MISSING=$(echo "$MISSING" | tr ',' '\n' | wc -l)
    echo "Batch $LO–$HI: $N_MISSING missing — submitting sparse array..."
    DEP_FLAG=""; [[ -n "$PREV_JID" ]] && DEP_FLAG="--dependency=afterany:$PREV_JID"

    JID=$(sbatch --array=${MISSING}%4 $DEP_FLAG "$SCRIPT" | awk '{print $NF}')
    if [[ -z "$JID" ]]; then echo "  ERROR: submission failed"; exit 1; fi
    echo "  → job $JID  ($N_MISSING tasks)"

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
