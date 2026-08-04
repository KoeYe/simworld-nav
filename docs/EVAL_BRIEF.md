# Brief for a reviewing agent

You are being asked to do two things that are easy to confuse:

1. **Play the environment as the courier.** Read the prompt and the photographs,
   choose actions, try to deliver. You are the policy.
2. **Report what is wrong with it.** Not with your play — with the environment.

The second is the deliverable. The first is how you earn the right to it.

Read `docs/RUNNING.md` first; it tells you how to start an episode.

## Ground rules while you play

**Read the pictures before you act.** The text will never tell you the colour of
a light, what is standing in a street, or what a shopfront says. If you act from
the text alone you will walk into barriers — the person who built this did it
twice in eleven turns and lost 45 s each time.

**You may not read the source while playing.** No peeking at `courier_env.py`,
no calling `env.light_here()`, no `route_nodes`. Those are privileged and using
them makes your play worthless as evidence. Play from `session.observe()` and
nothing else.

**One turn per process, or keep a transcript.** A model does not carry Python
objects between calls; it carries text. If you hold a live session across your
whole run you will accidentally use state a real policy would not have.

## What to look for

Play at least two full episodes — one `solo`, one `pair` or `triple`, at
`stride="block"` — and write down, every single turn, any moment where you were:

- **missing information** you needed to choose
- **given contradictory information** by two parts of the observation
- **misled** — you did the sensible thing and the world punished it
- **unable to express** what you wanted to do with the nine verbs
- **unsure what a message meant**

Those five categories are the finding. A confusion you had and worked around is
still a defect: the next policy may not work around it.

Some already-known ones, so you do not spend your time re-finding them:

- Two streets leaving a corner on nearly the same bearing get near-identical
  photographs; only the name separates them.
- On turn 1 there is no facing yet, so the candidate list gives a compass
  heading and no left/right, although the prompt promises both.
- Views at the edge of the map end in flat grey under sky. The street does not
  actually end there.
- The numbered streets are re-derived every turn. `walk_to(1)` twice is not
  "keep going straight".
- 7.5% of edges are under 5 m, so a whole turn can buy you two metres, and the
  compass bearing of a 2 m stub is noise.

## What would be a serious finding

In rough order of how much I would want to know:

1. **A way to succeed without looking.** If you can deliver reliably while
   ignoring every photograph, the benchmark is not measuring perception and that
   is the most important thing you could tell me.
2. **A way to succeed by exploiting the harness** rather than navigating — a
   parse trick, a tool that returns more than it should, a refusal that leaks
   the answer. `collect()`'s refusal used to state the exact distance to the
   door, which made it a free rangefinder.
3. **An unwinnable situation** the courier can be put in through no fault of its
   own. Livelocks especially: the phone routes through a barrier it cannot see,
   and if going round is impossible rather than merely hard, that is a bug.
4. **A number in the observation that is not true** — a distance that disagrees
   with what walking it costs, a heading that points the wrong way, a door that
   cannot be reached from the kerb it names.
5. **Anything in the text that gives away what only a picture should.**

## What is deliberate, so do not report it

- The phone cannot see and never learns. It will route you down a street with a
  barrier in it every time you ask. Going round is your job.
- `navigate()` costs 15 s and the map then stays on screen; the route is frozen
  where it was drawn and only your position updates.
- Deadlines are tight and the deep tiers are meant to be unfinishable at the
  reference level. Failing to deliver everything is not a bug.
- `no_phone` and `visual` conditions score 0. They are declared and explicitly
  not validated.

## How to report

For each finding: what you were shown (paste the observation), what you did,
what happened, and what you expected. A finding without the observation attached
cannot be reproduced, and one that cannot be reproduced cannot be fixed.

Then run the checks and report the numbers:

```bash
python3 -m pytest tests/ -q                       # expect 950 passed, 4 skipped
python3 -m embodiedbench.tools.migration_check    # expect exit 0
```

Finally, please state plainly how your own run went — delivered, turns,
simulated minutes, barriers hit, red lights crossed. If you did badly, say so
and say why. The reference policies deliver 5/6 and 8/12 on the easy tiers; the
privileged router delivers everything. Landing below the reference is
informative and worth reporting honestly rather than quietly re-running until a
seed goes well.
