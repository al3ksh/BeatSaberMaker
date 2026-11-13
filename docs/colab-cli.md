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

The delta holds all of `events/`, `dataset_stats.json` and the features of maps absent from the base index.

## Session

    colab usage                                  # record before
    colab new -s train --gpu A100 --high-mem
    colab drivemount -s train                    # interactive: only the user can run this
    colab status -s train

Put the two VM scripts on the VM (re-upload after changing them):

    colab upload -s train scripts/colab/vm_setup.py /content/vm_setup.py
    colab upload -s train scripts/colab/vm_train.py /content/vm_train.py

`colab exec -f FILE` sends a local file's content to the kernel (no command-line arguments), so to
run a script with arguments, exec a one-line snippet from stdin. Setup (unpacks about 40 GB, takes a while):

    echo 'import subprocess, sys; subprocess.run([sys.executable, "/content/vm_setup.py", "--drive-dir", "/content/drive/MyDrive/BeatSaberMaker", "--work", "/content"], check=False)' \
      | colab exec -s train --timeout 0

Without a delta tar add `"--no-delta"` to the list. Setup is safe to run twice: finished steps are skipped
(`.base.done`, `.delta.done`), an interrupted base unpack (`/content/data.partial`) is redone from scratch.
It prints feature file count, index entries, bundle commit and free disk.

## Training

Checkpoints and `console.log` go to `<drive>/checkpoints/gen2/<RUN>/`. Options after `"--"` go to
`bsmaker train` (not `--out`). Check the command first with `"--dry-run"` added.

    tmux new -s train
    echo 'import subprocess, sys; subprocess.run([sys.executable, "/content/vm_train.py", "--model", "timing", "--name", "t1", "--", "--max-steps", "40000"], check=False)' \
      | colab exec -s train --timeout 0
    # detach tmux with Ctrl-b d; reattach with: tmux attach -t train

Run it inside `tmux` as an active `colab exec --timeout 0`: a background job on the VM survives
detaching but leaves the kernel idle, and an idle runtime gets reclaimed. The last output line is
`vm_train: exit code N`. Then run the pattern model the same way with `"--model", "pattern"`.

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
