# Running the courier benchmark on another machine

Everything in this repository is text. The city the courier walks is not: it is
2.6 GB of baked photographs that cannot go in git. This document is what you
need to get from a fresh machine to a training run.

Two things to install, in this order: the **photographs** (one archive, below)
and the **two Python environments** (further down). Nothing runs without both.

---

## 1. The photographs

One archive, `courier-city.tar`, 2.6 GB. It holds the compiled map and all five
albums under a single directory, so unpacking it is the whole install.

**Where to put it.** Anywhere with 6 GB free — 2.6 for the archive plus 2.6 for
what comes out of it, and you can delete the archive afterwards. Prefer a disk
you are not sharing with a training run's checkpoints. Then export
`COURIER_DATA` pointing at the unpacked directory; everything below reads that
one variable.

```bash
export COURIER_DATA=/scratch/courier-city          # or wherever you like
mkdir -p "$(dirname "$COURIER_DATA")" && cd "$(dirname "$COURIER_DATA")"
# ... put courier-city.tar here ...
sha256sum -c SHA256SUMS
tar -xf courier-city.tar                           # creates courier-city/
```

You should end up with:

```
$COURIER_DATA/
  citycore-paris/                            the compiled map: nodes, streets, addresses
  paris_streets_v2/citycore-paris/           the view down each street            (always)
  paris_signals_kerb/citycore-paris/         pedestrian lamps, one frame a phase  (hazards)
  paris_obstacles/citycore-paris/            barriers and crowded pavements       (hazards)
  paris_streets_pavement/citycore-paris/     the same views from the pavement     (on foot)
  paris_obstacles_pavement/citycore-paris/   obstacles from the pavement          (on foot)
```

**The sidecar JSONs matter.** `signal_visibility.json` lists the approaches
whose lamp the album can actually show, and the environment charges for
crossing on red *only* on those. An album without its sidecar charges nothing,
which is the intended failure mode — silence is not consent — but it means half
the benchmark quietly switches off. Check they came through:

```bash
ls "$COURIER_DATA"/paris_signals_kerb/citycore-paris/signal_visibility.json
ls "$COURIER_DATA"/paris_obstacles/citycore-paris/obstacle_visibility.json
```

Then point the repo at them:

```bash
cd /path/to/simworld_nav
python scripts/point_at_albums.py "$COURIER_DATA"
```

That rewrites the album paths in `embodiedbench/training/vagen/*.yaml`, copies
nothing, prints every change, and refuses to write a path that is not there.
`--check` reports without writing.

## 2. The two environments

They are separate on purpose. Training needs vLLM and a CUDA build of torch
pinned to what verl expects; evaluation needs neither and must stay importable
on a machine with no GPU, because the test suite runs there.

### Evaluation and tests — Python 3.11

```bash
conda create -n courier python=3.11 -y && conda activate courier
pip install numpy pillow cairosvg networkx pytest pyyaml requests
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is fine
```

Verify — this needs no GPU, no server, and no model:

```bash
cd /path/to/simworld_nav
PYTHONPATH=. pytest tests -q          # ~6 min, 1081 tests
```

If the albums are mounted, roughly 40 of those tests exercise the real
photographs; if not, they skip and say so.

### Training — Python 3.12

```bash
conda create -n courier-rl python=3.12 -y && conda activate courier-rl
pip install torch==2.8.0
pip install vllm==0.11.0 ray==2.53.0 transformers==4.57.1 accelerate==1.12.0
pip install cairosvg pillow numpy pyyaml
```

`cairosvg` is easy to forget and fails silently: the phone's map is rendered
from SVG, and without it the map is simply never sent while the environment
goes on telling the courier to read the route off it. That defect ran
undetected here for a full training run.

Then the two vendored checkouts, which are gitignored and pinned:

```bash
mkdir -p vendor && cd vendor
git clone https://github.com/ymzhang0303/VAGEN.git vagen
cd vagen && git checkout b93deaa
git clone https://github.com/JamesKrW/verl.git verl
cd verl && git checkout 3fe0a29
cd ../.. && pip install -e vendor/vagen -e vendor/vagen/verl
```

Copy the map in so VAGEN can find it:

```bash
mkdir -p vendor/vagen/vagen/envs/deliverybench/maps
cp -r "$COURIER_DATA/citycore-paris" vendor/vagen/vagen/envs/deliverybench/maps/
```

### The model

```bash
export HF_HOME=/somewhere/with/40GB
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
```

---

## 3. Running it

### A quick check that the world works — no GPU

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.courier_oracle import run_reference_courier
import os
maps = Path('vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris')
net = build_road_network(maps, map_name='citycore-paris')
env = CourierEnv(net, seed=0, difficulty='solo', stride='block'); env.reset()
run_reference_courier(env, 0, max_steps=400)
print('reference courier delivered', env.summary()['delivered'], 'of', env.summary()['orders_issued'])
"
```

It should say `1 of 1`. The reference courier reads the geometry directly, so
this proves the map and the graph are sound. It does not prove the photographs
are, which is what the next step is for.

### Evaluating a model

Serve it, then run 40 held-out shifts:

```bash
# terminal 1
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image":16}' --port 8500

# terminal 2
PYTHONPATH=. python docs/review/tools/run_vlm.py \
  --model qwen3-vl-4b --port 8500 --seeds 40 \
  --tier solo --stride block --max-turns 40 --max-tokens 1024 \
  --out artifacts/vlm/mine.json
```

`--limit-mm-per-prompt` matters: a junction can offer six streets, their lamps
and the map, and a server that refuses the request returns a 400 that is not a
model output. The harness records those separately and excludes them from the
denominator — read `aborted_by_infrastructure` in the output before believing
any rate.

For pass@k on earnings rather than a binary:

```bash
PYTHONPATH=. python docs/review/tools/passk_earnings.py \
  --model qwen3-vl-4b --port 8500 --seeds 10 --k 8 --temperature 1.0
```

### Training with GRPO

One GPU is not enough; four 24 GB cards is what this has been run on. The
launcher picks cards by live free memory at launch time rather than from a
snapshot, because on a shared machine a snapshot taken when you edit the config
is already wrong by the time vLLM starts.

```bash
export HF_HOME=/somewhere/with/40GB
export MODEL_PATH=$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<hash>
export EXPERIMENT_DIR=/somewhere/for/checkpoints

bash scripts/train_grpo.sh
```

`scripts/train_grpo.sh` picks the GPUs, computes the vLLM memory fraction from
the tightest chosen card, opens every one of them before launching (a card
released seconds ago is not yet CUDA-openable, which fails as an unrelated
assertion deep in ray), and then calls
`embodiedbench/training/vagen/train_grpo_courier.sh`.

Knobs worth knowing, all environment variables:

| Variable | Default | Why you might change it |
|---|---|---|
| `WANT_GPUS` | 4 | Batch is 24 rollouts and must divide evenly: 2, 3, 4 or 6. |
| `TRAIN_BS` | 6 | Prompts per step. With `rollout.n=4` that is 24 episodes. |
| `ACTOR_LR` | 1e-6 | Reward scale and learning rate multiply. Raising both at once cost a run. |
| `KL_COEF` | 0.005 | Do not set to 0. With no anchor, entropy doubled every step and the policy collapsed. |
| `TEST_FREQ` | 10 | Validation every N steps. |
| `MAX_ATTEMPTS` | 40 | The supervisor restarts on death; each attempt writes its own log. |

Watch it with:

```bash
tail -f "$EXPERIMENT_DIR"/../grpo_run_*.log | grep -E "val-core|critic/score/mean"
```

### Reading the result — do not compare two means

Validation is 64 greedy episodes on fixed seeds. Run three times on the *same
untrained model* it gave delivery rates of 23.4%, 14.1% and 17.2% — a standard
deviation of about 5 points, with only a few tokens of prompt differing between
runs. Anything under roughly 10 points is unreadable from the means alone, and
two conclusions were reported here before that was measured.

The episodes are paired by construction — same seeds, same order — so compare
them that way:

```bash
python docs/review/tools/paired_validation.py "$EXPERIMENT_DIR/validation_data" --all
```

It reports which seeds started earning, which stopped, McNemar's exact p on
those counts alone, and the paired difference in earnings with a bootstrap
interval. Most of the variance between two validations is *which seeds are
hard*, and that cancels when each seed is compared with itself.

---

## 4. What breaks first

Ordered by how often it has actually happened here.

**`Free memory on device (5.75/23.55 GiB) is less than desired`.** Someone else
took the card between your config and vLLM starting. `scripts/train_grpo.sh`
recomputes at launch; if you launch by hand, do the same.

**`device >= 0 && device < num_gpus`.** Not a verl bug, though it was diagnosed
as one three times. A GPU released seconds earlier is not yet openable. The
launcher opens each card before starting; a fifteen-second wait does the same.

**NCCL collective timeout, a broadcast hanging for 600 s.** `NCCL_P2P_DISABLE=1`.

**`param_offload=True` doing nothing.** That is FSDP1's name. Under
`strategy=fsdp2` the parameter is `offload_policy=True`, and the wrong one is
accepted in silence.

**The map never appearing.** `cairosvg` missing in the training environment
while present in the evaluation one. The environment goes on saying "your phone
is showing the route" and no image is attached. Check with
`python -c "import cairosvg"` in *both* environments.

**A delivery rate that looks impossibly low.** Check
`aborted_by_infrastructure` in the evaluation output first. Sixteen episodes
that ended on turn 1 against a server that had stopped answering once sat in a
denominator here and turned 33% into 20%.

---

## 5. What the courier is actually asked to do

Worth reading before you interpret any number, because the answer changed
recently and it is the point of the benchmark.

Three sources of information, and no one of them is enough:

* **The map, which is a picture**, is the only place the direction to walk
  exists. Nothing the environment says in words names a compass point — a test
  asserts this over every sentence it speaks. At the 320 px the harness sends,
  the route line, the position dot, the destination and the compass are legible
  and the street labels are not, deliberately.
* **The junction, which is text**, is the only place street names come from.
  `walk_to("Rue de Grenelle", "east")`. Only the streets listed exist this
  turn; a street further along the route cannot be walked from here.
* **The photographs** are the only place a red light or a barrier appears.

Before this, the phone read the route out leg by leg — "Take Rue de Grenelle,
east, 1 junction, 18 m" — and a policy holding that text never had to look at
anything. Removing it took delivery from 6/40 to 3/39 on the same model, which
is the size of the benchmark that was previously being solved by reading.
