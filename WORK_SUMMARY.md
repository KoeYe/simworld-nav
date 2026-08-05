# 这一轮做了什么、怎么做的、怎么用、还差什么

面向 Paris courier benchmark（`embodiedbench`）。目标是把它打磨到「embodied agent 的 SWE-bench」那个水准：
一个 agent 拿到文字 + 图片，就能完成派送；环境不泄露答案，也不藏必要信息。

下面四节分别对应：**做了什么 / 怎么做的 / code 怎么用 / 还没做**。

---

## 1. 做了什么

### 1.1 全量审查 + 修 bug

两个 agent 独立跑完所有 tier / stride / condition，各自出一份问题清单，合并进
`agent1_debug_report.md`（DBG-001..020 + A2-01..20 + 对账表）。**只修两份报告都判定 100% 是 bug 的条目**，
涉及设计取舍的没动。已修的主要几条：

| 问题 | 改动 | 验证结果 |
|---|---|---|
| 失败的工具调用泄露到目标的精确米数 | refusal 只说「你不在门口」，不再给距离 | refusal 文本无数字 |
| 图片文件名泄露目标 | 新增 `agent/courier/frame_alias.py`，帧名改成内容寻址的不透明串 | 泄露率 **39.9% → 0%** |
| 门牌号在街上不单调，跳变很大 | `road_network.py` 改成全街共享游标、按 side 定奇偶 | N 与 N+1 最远间隔 **12 → 5** 个路口 |
| prompt 里出现不可调用的工具（幻觉工具） | 构造期扫描整段 prompt，发现就报错 | 顺带查出 `no_phone` 下仍在宣传 `check_order()` 的第二处泄露 |
| block stride 下 route/junction/距离语义互相矛盾 | 新增 `_block_preview()`，候选行改成「61 m on, 4 junctions, to the next choice」 | 300+ 处不一致清零 |
| observation 冗余 | 删掉从不影响决策的 `Streets you have seen`，照片 caption 压成一行 | observation **161 → 134** 行 |

时间预算 `TIME_BUDGET_MULTIPLE` 从 7.0 调到 **3.5**：跑了 6 个取值的 sweep，3.5 是「有视觉」相对「盲走」
增益最大的点（+11.2，4/4 格子视觉获胜）。太松的预算会让盲走也能完成，benchmark 就测不出感知能力了。

### 1.2 人行道视角 + embodiment

原来所有街景都是**马路中央**拍的 —— 一个「走路」的 agent 看到的却是车道视角，这不成立。

- 新烘焙 **856 帧人行道街景** + **240 帧人行道障碍帧**（`tools/ue/bake_pavement_views.py`，
  相机在行进方向右侧 `width/2 + 60cm`）。
- 新增 `runtime/city/embodiment.py`：`human_on_foot` / `human_on_scooter` / `human_in_car` 三种人类形态，
  各自有速度、体力上限、体力消耗率（按距离算）、停车开销、视角（人行道 or 车道）。
- 新增 `rest()` 动作（`REST_SECONDS = 60`）恢复体力。体力耗尽会真的拖慢行程，不是摆设。
- `robot_dog` / `humanoid_robot` 目前是**空 config**，调用 `require_defined()` 会直接抛错，
  而不是偷偷用人的参数跑 —— 按你的要求先占位。

途中我自己引入过一个泄露：人行道街景配车道障碍帧，光看视角就能猜出哪里有障碍（19.5%）。
现在构造期就会拒绝这种不匹配的组合。

### 1.3 vLLM 起来了

`/data/murray/serve_q3vl.sh`，跑 **Qwen3-VL-4B-Instruct**，端口 8200。

关键是版本组合：`vllm 0.12.0 + torch 2.9.0+cu128 + flashinfer 0.5.3`。
之前一直失败的根因是 vllm 0.26 会拉 torch 2.11+cu130，flashinfer JIT 编译时撞上 CUDA 13 删掉的 CUB `FlagHeads`。
这个组合是照抄同机器上另一个用户已经跑通的环境。

**8B 在单张 24 GB 卡上放不下**（权重 + vision tower 之后 KV cache 不够，已实测三档
`gpu-memory-utilization` 都失败），所以用的 4B。

### 1.4 真的拿 VLM 跑了 benchmark

Qwen3-VL-4B，solo/block，6 个 seed：

```
delivered 0/6, 106 turns, format errors 9 (8.5%), rejected 3 (2.8%)
动作分布：walk_to 87, navigate 3, wait 2, check_map 2, look 1, check_order 1, collect 1
```

对比 Qwen2.5-VL-7B 的 19.1% format error —— 新模型格式能力明显更好，**但一次都没送到**。

失败模式很清楚：它 82% 的动作是 `walk_to`，几乎不看地图（2 次）、不看订单（1 次）。
它会正确认出自己在 Rue Saint-Antoine 上，但不会用门牌号收敛到 9 号。

**我专门验证了这不是环境的锅**：每回合 observation 都写着「You are on X, outside number N」，
而且全图 **477 个地址，477 个在自己门口都能读到自己的门牌号**（0 个读不到）。
信息是全的，是 policy 不会用。

### 1.5 RL 跑通了

`training/courier_rollout.py`（把 courier episode 变成训练样本）+ `training/train_courier.py`（训练循环）。
REINFORCE + batch mean baseline，masked policy loss。

设计上有两条硬线：

- **训练目标和 benchmark 分数是两列**。`env_return` 是 benchmark 记的分，永远不加 shaping；
  `reward` 是优化器看的，可以带 shaping 项。两个字段一路分开传到日志，避免把带 shaping 的数字当成 benchmark 成绩报出去。
- **只信 held-out**。第一版训练看着从 0.25 涨到 1.0，我用 `--learning-rate 0` 跑同样配置，
  复现了同样的曲线（0.425 vs 0.350）—— 那「学习」其实是抽到什么 seed。
  现在训练 seed 是固定池，报的数来自**每轮完全相同的 held-out 集**、no_grad、greedy。

实测结果（都是 held-out）：

| 模型 | 方式 | 起点 → 终点 |
|---|---|---|
| Qwen2-VL-2B | full FT | 0.2812 → **0.8125**（lr=0 对照组：0.2812，纹丝不动） |
| Qwen2-VL-2B | LoRA | 0.2812 → **1.0**（第 3 轮到顶） |
| Qwen2.5-VL-7B | LoRA | 0.5 → **0.6875**，env_return 非零（0.0125） |
| Qwen3-VL-4B | LoRA + progress | **没学到**，见下 |

**Qwen3-VL-4B 那次要如实记一笔**：held-out progress 跑完 14 轮是
`0.647 → 0.727 → 0.333 → 0.831 → 0.823 → 0.596 → 0.625 → 0.547` ——
**围绕 0.65 的噪声，没有趋势**，`delivered` 全程 0。
中途取第 8 轮的 0.823 说「在涨」是挑点，不成立。
这也是把主线换到 verl/GRPO 的直接原因。

一路上真正卡住的坑：optimizer 每次调用都被重建，state 全丢 —— 这是之前学不动的主因。
其余是 OOM 链：参数快照放 GPU→改放 CPU、没开 gradient checkpointing→开、
pixel_values 常驻显存→挪到 CPU、`parameter_digest` float64 upcast 一个 7B tensor 吃 4 GB→改成分块。

**这轮新加的：progress shaping（第二级课程）。**
Qwen3-VL-4B 在 format 这一级上**开局就是 1.0** —— 满分意味着 batch 内每个 episode 得分一样，
advantage 全 0，优化器拿到的是 no-op（实测连续 4 轮 `step: skipped`）。
所以加了一项「离目标又近了多少米」：

- 用 `route_length_cm`（环境自己的记账，从不出现在 observation 里）；
- **potential-based**（Ng, Harada & Russell 1999），所以不改变最优策略，只是把「送到了」这件事提前说出来；
- 收件后目标会从 pickup 跳到 dropoff，两者隔着几条街，所以**目标切换的那一回合不计分**，
  否则会白送一大笔它没走出来的奖励；
- 永远不进 `env_return`，并有测试钉住这一点。

加上之后 Qwen3-VL-4B 的 `eval_progress` 基线是 0.647 且跨 seed 有方差，优化器终于有东西可推。
这一轮训练正在跑（`artifacts/training/q3vl4b_lora_progress.json`）。

### 1.6 报告

- `docs/review/TEST_PLAN.md` —— **动手前**写的测试计划，定义 S1–S5 严重度和「一票否决」条件。
- `docs/review/REPORT.md` —— 详细叙述版。
- `docs/review/report.html` —— 可视化版，把 trajectory 一步步画出来（自带帧图，离线可看）。
- `agent1_debug_report.md` —— 两个 agent 的完整问题清单 + 对账 + 修复状态。

---

## 2. 怎么做的（方法上值得说的几点）

**先写计划再动手。** `TEST_PLAN.md` 是在跑第一个 episode 之前写的，包含严重度定义和什么算「不合格」。
这样后面判断一个现象算不算 bug，不是当场临时定标准。

**盲走对照是主要测量手段。** 很多「这个信息有没有用」的问题，靠读代码是吵不出结论的。
做法是同一配置跑两遍，一遍给图一遍不给图，比完成率。视觉没有增益的配置，说明那一档根本没在测感知。

**自己错了就撤。** 我曾断言「短支路上的路障不可见」，实测下来是错的
（138 个 8 m 以内的接近里只有 6 个标了可见，障碍是近场渲染的），就把这条撤了。
另外，三种检测方法都把真实帧分错类之后，我没有硬上一个凑合的 heuristic caption。

**门牌号那次测量我第一版是错的。** 第一次报 87.5% 非单调，用的是 node-id 排序，不对。
改成按边的几何顺序之后：每侧单调 42/46，显示层面的倒挂 22/23，最大间隔 12 个路口。

**杀进程不要用 `pkill -f`。** 这个模式会匹配到我自己的命令行，把 launcher 一起杀掉，
然后我读到旧日志、误判原因 —— 这个坑我踩了不止一次。现在按端口/PID 杀，且杀和启动分在两条命令里。

---

## 3. code 怎么用

环境：`/data/murray/miniconda3/envs/simworld`（benchmark + 训练），
`/data/murray/miniconda3/envs/vllm`（只用来起 vLLM 服务）。

### 3.1 跑测试

```bash
/data/murray/miniconda3/envs/simworld/bin/python -m pytest -q --ignore=vendor
# 当前：990 passed, 4 skipped
```

### 3.2 起 vLLM

```bash
setsid nohup /data/murray/serve_q3vl.sh > /data/murray/q3vl.log 2>&1 < /dev/null &
# 等 :8200 起来
until ss -ltn | grep -q 8200; do sleep 10; done
```

改模型就编辑脚本里的 `--model` 和 `--served-model-name`。
注意 24 GB 单卡放不下 8B，要换 8B 得上 TP=2 或量化。

### 3.3 拿 VLM 跑 benchmark

```bash
/data/murray/miniconda3/envs/simworld/bin/python docs/review/tools/run_vlm.py \
  --model qwen3-vl-4b --tier solo --stride block --seeds 6 --max-turns 25 \
  --out artifacts/vlm/qwen3vl4b_solo.json
```

`--tier` 可选 `solo/pair/triple/shift/endless`，`--stride` 可选 `waypoint/block`，
`--embodiment` 可选 `human_on_foot/human_on_scooter/human_in_car`。

### 3.4 训练前的自检（不用 GPU 也能跑一部分）

```bash
/data/murray/miniconda3/envs/simworld/bin/python -m embodiedbench.training.courier_gate \
  --tier solo --seeds 3 --report artifacts/training/gate.json
```

5 项检查：mask 里没有 observation/图片 token、每个采样 token 都有 logprob、
reward 拆分守恒、advantage 居中、动作真的进了环境。

### 3.5 训练

```bash
M4=$(ls -d /data/murray/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/ | head -1)
CUDA_VISIBLE_DEVICES=7 HF_HOME=/data/murray/hf \
/data/murray/miniconda3/envs/simworld/bin/python -m embodiedbench.training.train_courier \
  --model "$M4" --lora --no-hazards \
  --iterations 14 --batch 4 --eval-every 2 \
  --learning-rate 1e-5 --format-weight 0.0 --progress-weight 1.0 \
  --max-turns 8 --max-images 1 --max-new-tokens 1024 \
  --log artifacts/training/q3vl4b_lora_progress.json
```

几个必须知道的参数：

- `--max-new-tokens` **别设太小**。48 会把 Qwen3-VL-4B 的长 THOUGHT 截断在 fenced call 之前，
  format_score 假摔到 0.125 —— 这是配置问题，不是模型问题。
  它只是**上限**，生成遇到 EOS 就停，所以设大基本不花额外时间，只在模型真的啰嗦时才有代价。
  用 1024。实测 160 和 1024 的 held-out 基线完全一样（format 1.0 / progress 0.647），
  说明对这个模型 160 已经不截断了；但 1024 更安全，换个更啰嗦的模型也不会翻车。
  显存代价可以忽略：4B + LoRA 在 1024 上占 13.7 GB / 24 GB。
- `--format-weight` 对已经会输出格式的模型请设 0，否则 advantage 全 0、每轮 `step: skipped`。
- `--progress-weight` 是第二级课程，只进训练目标不进 benchmark 分数。
- `--lora` 是 7B 能塞进 24 GB 的原因。
- **报数只看 `eval_*` 字段**，`train_reward` 会受 seed 抽样影响。

### 3.6 声称「学到了」之前必须跑的对照

```bash
# 同样配置，唯一区别是 lr=0
... --learning-rate 0 --log artifacts/training/control.json
```

held-out 曲线如果和 lr=0 分不开，那就不是学习。这条是硬要求，我自己第一版就栽在这。

---

## 4. 还没做 / 已知限制

**benchmark**

- `robot_dog` / `humanoid_robot` 还是空 config（按要求先做人）。需要各自的速度、体力、视角高度，
  以及**对应视角高度的街景 album**（狗的相机高度和人差很多，不能复用人行道帧）。
- 有争议、两份报告没有都判定为 bug 的条目**故意没改**，清单在 `agent1_debug_report.md`。
- 人行道 album 只覆盖了 856 帧主干；不是全图每条边都有人行道视角。

**模型 / serving**

- 8B 单卡放不下，还没试 TP=2 或 AWQ 量化。
- Qwen3.5-9B 权重已经下到 `/data/murray/hf`，但 vllm 0.12.0 不认 `Qwen3_5ForConditionalGeneration`；
  升级 vllm 会重新触发 cu130 / flashinfer 那个死结。想上 3.5 得先解决这个。

**RL：VAGEN / verl（GRPO）**

正式的 RL 走 VAGEN 的 verl 栈（`ymzhang0303/VAGEN`，vendor 在 `vendor/vagen`），
我手写的 REINFORCE 只作为对照保留。

环境：conda env **`vagen`**（python 3.12），严格按 `vendor/vagen/README.md` 装。

```bash
setsid nohup /data/murray/run_grpo_courier.sh > /data/murray/grpo_courier.log 2>&1 &
```

配好这套环境踩到的坑，按撞到的顺序（每一条都是真的会卡死的）：

1. **conda ToS 没接受** → `conda create` 直接 exit 1。
2. **`vendor/vagen/verl/` 是空目录** —— verl 是没初始化的 submodule，
   而空目录会让 `import verl` 假装成功。`git submodule update --init --recursive`。
3. **submodule 落在 `main` 的 commit 上，不是 `.gitmodules` 写的 `vagen-lite`** →
   `ImportError: cannot import name 'compute_reward'`。必须 `git checkout vagen-lite`
   （并先清掉 `main` 上遗留的嵌套 `recipe` submodule）。
4. **flashinfer JIT 编译失败**（`error: math.h: No such file or directory`）→
   `/usr/bin/nvcc` 是 CUDA 12.0 而 torch 是 cu128。换 sglang→vllm **没用**，
   两个后端都要 flashinfer。真正的修法是在 env 里装 `cuda-nvcc=12.8` 并
   `export CUDA_HOME=$CONDA_PREFIX`。
5. **`curand.h` 找不到** —— conda 把头文件放在 `$CUDA_HOME/targets/x86_64-linux/include/`，
   flashinfer 只找 `$CUDA_HOME/include/`。软链过去。
   *（改完这两条要 `rm -rf ~/.cache/flashinfer`，否则复用失败的产物。）*
6. **KV cache 不够** → `ROLLOUT_MEM=0.4` 太小，3B 权重就吃掉大半。用 0.7。
7. **图像预算**（见下）。

**图像预算是这个 benchmark 特有的约束。** courier 每回合最多 9 张图，
一张 640×480 经 vision merge 约 **380 token**。5 张 × 20 回合 ≈ 38,000 图像 token。
prompt 或 response 被截断时，image-pad token 跟着被砍，但 `multi_modal_data`
仍然列着全部图片，`get_rope_index` 就会多走一张图的 grid：

```
RuntimeError: shape mismatch: value tensor of shape [3, 8145]
cannot be broadcast to indexing result of shape [3, 7765]
```

差值恰好是一张图。**注意 response 侧同样要算** —— 多轮拼接里第一回合之后的
observation 全在 response 段，只调 prompt 预算会以更小的差值再报一次同样的错。
当前配置：`max_images=1`、`max_turns=8`、prompt 和 response 各 16384。

代价要说清楚：**每回合只给 1 张图，agent 只能看到一条街的照片**，
这削弱了视觉任务本身。要恢复多图，应该降低图片分辨率（380 token/图可以砍到 ~100），
而不是继续砍回合数。

**模型版本约束**：这套 frozen 依赖是 transformers 4.56.1 + sglang 0.5.2，
**都不认识 `Qwen3VLForConditionalGeneration`**，所以 verl 这条路目前只能跑
**Qwen2.5-VL**（用的 3B）。上 Qwen3-VL 需要升 transformers ≥4.57，会动这套
已知能跑的组合，属于要单独验证的一步。

**手写 REINFORCE（对照，不是主线）**

- 目前只在 `--no-hazards`（无红灯、无路障）的最简设置上验证过学习。
- 只跑到 `solo` tier，`pair/triple/shift/endless` 没训过。
- **还没有任何一次训练把 delivered 从 0 推到 >0。** 已经证明的是 format 和 progress 两级能涨，
  「真的送到」这一级还没验证 —— 这是最重要的下一步。
- REINFORCE + mean baseline，没上 PPO/GRPO，没有 value function。
- 按要求**没做 SFT 冷启动**。

**没验证的**

- 报告里的结论都是在 citycore-paris 这一张图上测的，没有第二张图做泛化对照。
- 没有真人 baseline 分数，所以「人类能做到多少」这一栏是空的。
