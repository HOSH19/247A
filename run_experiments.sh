#!/bin/bash
# Auto-resume experiment pipeline. Skips already-completed experiments.

cd /home/hanchoi/247A
source .venv/bin/activate

LOG=/home/hanchoi/247A/win_experiment.log
BASE="model=lstm_ctc transforms=log_spectrogram_hop48 trainer.max_epochs=40 +trainer.gradient_clip_val=1.0"

run_if_not_done() {
    local tag=$1
    local outlog=$2
    shift 2
    if grep -q "^${tag} done$" "$LOG" 2>/dev/null; then
        echo "$(date): Skipping ${tag} (already done)" >> "$LOG"
        return
    fi
    echo "$(date): Starting ${tag}" >> "$LOG"
    python -m emg2qwerty.train $BASE "$@" > "$outlog" 2>&1
    echo "${tag} done" >> "$LOG"
}

run_if_not_done "win4000"        win4000_hop48.log      datamodule.window_length=4000
run_if_not_done "win16000"       win16000_hop48.log     datamodule.window_length=16000
run_if_not_done "pad900"         pad900_hop48.log       "datamodule.padding=[900,100]"
run_if_not_done "pad3600"        pad3600_hop48.log      "datamodule.padding=[3600,400]"
run_if_not_done "win4000_pad3600" win4000_pad3600.log   datamodule.window_length=4000 "datamodule.padding=[3600,400]"
run_if_not_done "win16000_pad900" win16000_pad900.log   datamodule.window_length=16000 "datamodule.padding=[900,100]"
run_if_not_done "win16000_pad3600" win16000_pad3600.log datamodule.window_length=16000 "datamodule.padding=[3600,400]"

run_if_not_done "win16000_pad900_150ep" win16000_pad900_150ep.log \
    datamodule.window_length=16000 "datamodule.padding=[900,100]" \
    trainer.max_epochs=150

# GRU experiments
run_if_not_done "gru_screening" gru_screening.log \
    model=gru_ctc

run_if_not_done "gru_best" gru_best.log \
    model=gru_ctc datamodule.window_length=16000 "datamodule.padding=[900,100]" \
    trainer.max_epochs=150

run_if_not_done "conv_gru_conv_best" conv_gru_conv_best.log \
    model=conv_gru_conv_ctc trainer.max_epochs=150

echo "$(date): All done!" >> "$LOG"
