# Harness engineering log: what helped, what did not, and how it was done

Every number here is Qwen3-VL-4B on the same forty seeds of `solo`/`block`, up
to forty turns, run through `docs/review/tools/run_vlm.py`. One change per
round, measured before the next was made. The harness rounds are on twenty
common seeds because those runs were still going when this was written; they
are marked.

The short version: **the harness was worth more than the environment, the
prompt, or the model.** Six of the eight rounds moved something, and the single
largest effect — a 4× change in delivery rate — came from noticing that the
harness had never sent the model its own conversation.

---

## The table

| # | Change | Layer | Delivered | Collected | Refused actions |
|---|--------|-------|-----------|-----------|-----------------|
| 0 | *(first run)* | — | 1/40 | 8/40 | 40 + 78 "format errors" |
| 1 | Serving image limit 6 → 10; transport errors no longer parsed as replies | harness | 1/40 | 10/40 | 153 |
| 2 | Mark barriers the courier has seen; name dead ends; state the slip's street; state the legal indices | env + prompt | 2/40 | 11/40 | **20** |
| 3 | Ordered decision procedure in the system prompt | prompt | 1/40 | **20/40** | 115 |
| 4 | Dead-end refusal says backtracking is the move | env | **3/40** | 16/40 | 87 |
| 5 | `model_io.py`: requery, reasoning split, truncation, budget clamp | harness | 2/20 † | — | — |
| 6 | **Conversation history with image compaction** | harness | **8/20** † | — | — |

† twenty common seeds, not forty.

---

## What helped

**Conversation history (round 6) — 2/20 → 8/20.** The eval harness sent
`[system, current observation]` every turn and nothing else. The policy was
stateless: it could not remember where it had been, what it had tried, or that
it was already standing on the right street. Five seeds succeeded only with
history and none only without, so this is not sampling noise. It also finished
in fewer turns.

This one is worth dwelling on because of what it invalidated. Before it, the
measured failure profile was: 26 of 41 moves returning to an already-visited
junction, 81 departures from the target street, and a right-direction rate of
60% once on it. That was written up as *the model lacks cross-turn spatial
memory* — a capability claim. It was a harness claim. The model was never given
a memory to fail to use, and the runbook telling it to notice it was circling
was advice about information the harness had thrown away.

**The ordered decision procedure (round 3) — collection 11/40 → 20/40.** The
existing runbooks each answered one situation well and none said which
situation the courier was in. Adding a five-step "stop at the first line that
fits" nearly doubled collections and halved the median closest approach, from
72 m to 36 m. It cost 800 tokens of prompt, and the budget test now says what
that buys and what it displaces: at ~380 tokens a frame, 800 tokens is two
photographs the turn cannot carry.

**Showing the courier the barrier it walked into (round 2) — refusals 153 →
20.** A barrier refusal did not move the courier, recorded nothing, and
re-rendered the identical menu with the blocked street still listed first and
unmarked. 96 of 153 refusals were `way_blocked`, and 52 of 93 were the same
street tried again immediately. The code's justification — "a survey does not
learn" — is right about the phone and wrong about the courier, who is standing
at the barrier and can see it.

**The dead-end refusal (round 4) — deliveries 2 → 3, refusals 130 → 87.**
Naming the legal street was not enough: three episodes spent themselves asking
for street 2 at a junction with only street 1, thirty-three, twenty-six and
twenty-five times each, because the only legal move went back the way they
came. Saying *going back is the move* ended it.

**Requery instead of charge (round 5).** mini-swe-agent re-prompts on an
unparseable reply and never lets the environment see it. Charging a format
error as a world step measures a model's luck at formatting rather than its
competence at the task. The count is reported, so leniency is visible rather
than free.

## What did not help, or helped less than it looked

**Round 3 raised collections and *lowered* deliveries** (2 → 1) while refusals
went 20 → 115. Telling the model to `navigate()` when off the slip's street
took its `navigate` share from 1.1% to 29.3%; it spent the clock asking for
directions it already had. A prompt instruction is not free — it competes for
the same turns as the task.

**Nothing in the environment moved the ceiling.** Rounds 2 and 4 removed almost
every wasted action — refusals fell from 153 to about 20 — and deliveries went
from 1 to 3 out of 40. Those failures were real defects and they were noise,
not the binding constraint. It is worth saying plainly: if the goal had been a
better number, the environment work was the wrong place to spend four rounds.

**My own adaptive mechanisms each introduced a fresh fault.** Raising
`max_tokens` on truncation walked into the context limit and 400d every turn
(episodes ended after a mean of 2.6 turns with *zero* format errors). Raising
the serving image limit from 6 to 10 was still short of the 13 a six-street
junction needs. Each fix was one guess about a number the server already knew.

## What was measurement error rather than a result

Three findings had to be withdrawn after the harness was fixed, and all three
had already been believed once:

| Reported as | Actually |
|---|---|
| "9.7% format errors, 26 of 40 episodes killed by malformed replies" | HTTP 400s from a six-image serving limit, fed to the parser as if the model had said them |
| "Qwen3.5-9B: 82.5% format errors" | It rehearses calls inside `<think>`; the parser found two fenced blocks and refused replies that were all well formed |
| "held-out format score 0.125" (RL) | `max_new_tokens=48` cut the reasoning off before the fenced call |

All three present as *the model cannot follow the reply contract*. That is the
signature of a harness measuring its own configuration, and it is why the
adaptation now lives in one module instead of in whoever is reading the logs.

---

## How the harness engineering was done

**One change per round, same seeds, compare before deciding the next.** Two of
the rounds above would have been misread otherwise: round 3 looks like a clear
win on collections and is a small loss on deliveries, and the history result
would have been invisible if it had been bundled with the prompt change.

**Read the mainstream harnesses first.** The decisive idea — a *model call* and
a *world step* are different things, so an unparseable reply is requeried
rather than charged — is mini-swe-agent's, and it is also why that project is
model-agnostic by construction: same prompt, same tools, no per-vendor
instructions. The context-window discipline (compaction, keeping the newest
turn rich and older turns cheap) is the standard multi-turn pattern. Neither
was invented here.

**Make the server tell you its limits.** Every ceiling this harness guessed per
model was wrong at least once. The rule that survived is: read the number out
of the refusal. `maximum context length is N tokens` and `prompt contains M
tokens` are both in the 400 body, so the budget clamps itself and the clamp
sticks for the rest of the episode.

**Never let infrastructure look like behaviour.** A failed request raises with
the server's own message and is recorded as `infra_error`; it is never handed
to the parser. `finish_reason == "length"` is a configuration fault, not a bad
answer. A 4xx is not retried, because it is a request the harness built wrongly.

**Test the harness without a model.** `tests/test_model_io.py` runs all of it
against canned responses — no GPU, no server, under a second. Every case pinned
is a mistake this benchmark actually made. One of those tests failed against
correct code because the stub server refused requests it had just said would
fit; a stub that does not model the thing it stands in for tests a problem
nobody has.

**Keep the leniency visible.** Requeries, truncations, budget clamps and
transport retries are all counted into the episode summary. A model that needs
three attempts a turn should still look worse than one that needs none.

---

## Where this leaves the benchmark

The harness is now model-agnostic in the way that matters: it handles reasoning
models, models that need a large generation budget, and models served behind
different limits, without per-model code. That was not true a day ago, and a
benchmark that cannot evaluate reasoning models cannot claim to measure much in
2026.

What has not moved is the task. Best delivery rate is 8 of 20 with history, and
the environment work — which removed 87% of wasted actions — moved deliveries
by two. The next question is not another harness round; it is where the
remaining failures actually are, measured on the fixed harness rather than on
the one that was reporting its own configuration.


---

# Part two: the RL harness

The evaluation harness above was fixed first, and then the same class of defect
turned up in the training path -- three times, each one silently changing what
was being measured rather than announcing itself.

## The three bugs

**Training ran in a different city from evaluation.** The adapter derived the
hazard albums as `album_root.parent/"signals"/name`, a directory that has never
existed -- the bakes live in `paris_signals_kerb`, `paris_obstacles`,
`paris_streets_pavement` -- and an `if path.exists()` guard skipped each one
without a word. So training had no pedestrian lamps, no barrier frames, no
pavement views (an on-foot courier shown the carriageway, which is the exact
defect the pavement bake exists to fix) and `enforce_signals` quietly False,
while every evaluation number was measured with all of them. Two environments,
one set of conclusions drawn across both.

**traj_success could not be non-zero.** VAGEN takes trajectory success from
`info["success"]` or `info["is_success"]`; this environment's info dict had
neither, so `extract_success` returned False on every turn of every episode. It
was reported as 0.0 throughout, and read here -- repeatedly, in writing -- as
"the model never delivers". The moment the key was added it read 0.172, against
20% measured independently in evaluation. The metric had never been measuring
the policy. It also gates the agent loop's early exit, so a completed delivery
could not end its own episode.

**The caption promised photographs that were never sent.** The harness writes
"[1], [2] — the view down each of those streets, in that order" for every frame
a turn offers, and the image cap then sends one. A model reasoning about "the
photograph of street 2" was reasoning about an image it had never received.
This one has a retrospective cost: `earn@1 = 0.942` and `earn@8 = 2.549`, the
numbers the whole case for an earnings objective rests on, were measured that
way. The 2.7x headroom survives -- both halves were handicapped identically --
but the absolute figures came from a policy being misled and should be
re-measured before being quoted as a ceiling.

## What the optimiser needed

Separately from the bugs, three configuration facts were established by
watching the run fail:

* **The reward scale and the learning rate multiply.** Switching the basis from
  `env_return` to earnings raised the scale 2.5x; raising the learning rate 5x
  at the same time made the effective step 12x larger, and held-out earnings
  fell 1.42 -> 0.63 in twenty steps.
* **Nothing was anchoring the policy.** `use_kl_loss=False`, `kl_coef=0.0`,
  `entropy_coeff=0.0`, all copied from a reference script tuned elsewhere.
  Entropy doubled every step (0.30, 0.61, 1.14) while the score collapsed to
  0.01 -- textbook policy collapse with no reference to fall back on. A KL
  coefficient of 0.005 stopped it dead: entropy has since sat between 0.18 and
  0.23 for eight steps.
* **The shaping outweighed the money.** Training score averaged 3.8 against
  held-out earnings of 1.07, so roughly 70% of the gradient signal was "walk
  closer" rather than "earn". Group variance was 8-11 where earlier runs saw
  0.6-2. Dropping `progress_weight` to 0.2 brought variance back to 0.3-2.9.

## What this cost, and the rule that follows

Fifteen launches died on `Free memory on device ... less than desired`, because
GPUs were being chosen from a snapshot taken when the config was edited and the
server starts two or three minutes later. On a shared machine that snapshot is
already wrong. The launcher now picks cards and computes the memory fraction at
launch time, from the tightest chosen card minus a margin.

Prefer emptier cards to more cards. The rollout is data parallel, so every GPU
holds a whole vLLM replica and each one's KV cache comes out of its own budget,
while `gpu_memory_utilization` is global -- so adding a card someone else is
using cuts the budget on the empty ones too. Three empty cards at 0.88 gave
6.11 GiB of KV; six cards including shared ones at 0.72 gave 2.34.

## The honest state

Reward has not yet been shown to rise. The one run that reached a second
validation fell (1.42 -> 0.63), and it was running the wrong environment, with
no KL anchor, at twelve times the intended step size, reporting a success
metric that could not leave zero. Every conclusion drawn before those were
fixed is void. The current run is the first with all four corrected, and its
baseline -- earnings 0.769, success 17.2% -- is the first number here that is
directly comparable to an evaluation figure.
