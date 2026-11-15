# Training on Colab from the Raspberry Pi (`colab` CLI)

Operator guide for running round-2 training from the Pi over ssh with `~/.local/bin/colab`.
The notebook route (`docs/colab.md`) still works; this is the scripted route.

## Rules

- Before starting a paid runtime, say what will be started (GPU, high-mem, what will run, how long).
- Record `colab usage` before and after every run.
- Always `colab stop -s train` after a run, also after a failure.

## One-time files on Drive (`MyDrive/BeatSaberMaker/`)

`bsmaker-train-data.tar` (base), `bsmaker-train-delta.tar`, `bsmaker-code.zip`. Build on the PC:

    .venv/Scripts/python scripts/make_colab_bundle.py
    .venv/Scripts/python scripts/make_colab_bundle.py --delta --base-index BASE/events/index.json \
        --data-dir F:\BeatSaberMakerData --out-dir F:\BeatSaberMakerData\upload

The delta holds all of `events/` and the features of maps absent from the base index. It does not
hold `dataset_stats.json`: the code zip is its only source (tracked at HEAD, bundled by the first command).

Checklist: after `data build-events`, commit `dataset_stats.json` before building the code zip.

## Session

    colab usage                                  # record before
    colab new -s train --gpu A100 --high-mem
    colab drivemount -s train                    # interactive: only the user can run this
    colab status -s train

All commands below are run in a shell ON THE PI (over ssh), not on the PC and not on the VM.

First copy the two scripts from the repo (PC) to the Pi, once and again after changing them:

    scp scripts/colab/vm_setup.py scripts/colab/vm_train.py raspi:~/bsm-colab/

`colab exec -f FILE` sends the file's content to the VM kernel; it takes no arguments, so the script
arguments travel in the environment variable `BSM_ARGS` (`--env`), which the scripts read instead of `sys.argv`.
The exit status of `colab exec` is not relied on: the last output line and `console.log` on Drive are the record.

Setup (unpacks about 40 GB, takes a while):

    ~/.local/bin/colab exec -s train -f ~/bsm-colab/vm_setup.py --timeout 0       --env BSM_ARGS="--drive-dir /content/drive/MyDrive/BeatSaberMaker --work /content"

Without a delta tar add `--no-delta` inside `BSM_ARGS`. Setup is safe to run twice: finished steps are
skipped (`.base.done`, `.delta.done`), an interrupted base unpack (`/content/data.partial`) is redone.
The delta is re-applied when its size or modification time differs from the one recorded in `.delta.done`.
After unpacking, setup checks that every `map_key` in `events/index.json` has a feature file; if not it prints
the count and the first 10 keys and exits with status 1 before the pip step.
It prints feature file count, index entries, bundle commit and free disk.

`vm_train.py` exits with 2 (`vm_train: drive dir not found: ...`) when `--drive-dir` is not a directory
(Drive not mounted); `--dry-run` skips that check.

Prove the mechanism first (nothing is trained): the output must show the command line and `vm_train: exit code 0`.

    ~/.local/bin/colab exec -s train -f ~/bsm-colab/vm_train.py --timeout 0       --env BSM_ARGS="--model pattern --name pat-s --drive-dir /content/drive/MyDrive/BeatSaberMaker --work /content --dry-run -- --size s --corrupt-p 0.1"

## Training

Checkpoints and `console.log` go to `<drive>/checkpoints/gen2/<RUN>/`. Options after `--` go to
`bsmaker train` (not `--out`). Run it inside `tmux` on the Pi as an active `colab exec --timeout 0`:
a background job on the VM survives detaching but leaves the kernel idle.

    tmux new -s train
    ~/.local/bin/colab exec -s train -f ~/bsm-colab/vm_train.py --timeout 0       --env BSM_ARGS="--model pattern --name pat-s --drive-dir /content/drive/MyDrive/BeatSaberMaker --work /content -- --size s --corrupt-p 0.1"
    # detach with Ctrl-b d; reattach with: tmux attach -t train

The last output line is `vm_train: exit code N`. Run the timing model the same way with `--model timing`.

## Resuming

If the connection drops or the runtime dies, check `colab status -s train` (start a new session and
rerun the setup if it is gone), then re-run exactly the same training command: training resumes from
`last.pt` in the run folder on Drive. Use the same `--name`.

## Finishing

    colab download -s train /content/drive/MyDrive/BeatSaberMaker/checkpoints/gen2/t1/console.log ./t1.log   # optional
    colab stop -s train
    colab sessions                               # confirm nothing is left running
    colab usage                                  # record after

Checkpoints are already on Drive; fetch them to the PC from there.
