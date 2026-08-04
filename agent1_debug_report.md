# Agent 1：Paris Delivery Benchmark 详细调试报告

**报告日期：** 2026-08-03 UTC  
**测试代码版本：** `5130047c0851fbd4c5b4837cf34ba9e4917e6e70`  
**测试目录：** `/home/murray/simworld_nav`  
**测试角色：** 先作为只能看到公开 observation 的 embodied courier 完成任务，再作为 debugger 检查代码、配置、工具、harness 和验证脚本  
**相关报告：** `BENCHMARK_EVALUATION_REPORT.md`、`BENCHMARK_EVALUATION_REPORT.html`、`BENCHMARK_TEST_PLAN.md`

---

## 1. 总结

当前版本已经具备很好的工程基础：工具定义和 dispatch 共用一个 registry、照片和手机地图有明确类型区分、trajectory 能保存 prompt/图片顺序/action/outcome、有 950 个通过的测试、确定性 replay 和 stress compiler 也通过。

但是，当前版本仍存在若干会直接影响 benchmark 有效性、可复现性和 agent 训练质量的问题。最重要的问题是：

1. `collect()` 和 `hand_over()` 的失败反馈泄露精确距离，可被当成免费的 privileged rangefinder。
2. 文档中的 reference policy 调用如果没有显式 `env.reset()`，会得到“no order was generated”。
3. block stride 下，“next junction”“route junction”“实际经过 junction”“一次 action 的停止点”含义不一致。
4. 唯一 validated 的 `full` condition 本身可以依赖文本完成，因此当前 benchmark 不能支撑“通用视觉导航能力”的结论。
5. environment readiness、baseline provenance、训练 CLI 并没有全部通过，当前 release gate 不是绿色的。
6. observation 和 prompt 重复较多，训练和推理成本偏高。
7. 当前 CityCore Paris 是合成城市环境，没有足够证据支持真实巴黎部署。

建议把本文每一个 `DBG-*` 条目直接转成 issue，并把其中的“建议回归测试”加入测试集。

---

## 2. 实测配置与结果

### 2.1 人工 blind-play 配置

```python
env = CourierEnv(
    net,
    seed=0,
    difficulty="solo",
    stride="block",
    album_root=Path("/data/murray/paris_streets_v2/citycore-paris"),
    signal_album_root=Path("/data/murray/paris_signals_kerb/citycore-paris"),
    obstacle_album_root=Path("/data/murray/paris_obstacles/citycore-paris"),
)
env.reset()
session = CourierSession(env, city="Paris")
```

人工运行期间只读取：

- `session.system_prompt()`
- `session.observe().text`
- `session.observe().frames`
- `session.step(reply)` 的公开结果

在人工 episode 结束前，没有读取 `courier_env.py`、`route_nodes`、`light_here()` 等 privileged state。

### 2.2 人工 episode 结果

| 指标 | 结果 |
|---|---:|
| Delivered | 1/1 |
| On time | 1 |
| Turns | 34 |
| Simulated time | 635.2 秒 / 10.59 分钟 |
| Walked | 537.9 m |
| Barrier hits | 0 |
| Red crossings | 0 |
| Waits at red | 2 |
| Rejected actions | 2 |
| Slow passages | 1 |
| Earnings | 5.43 |

### 2.3 自动检查结果

| 检查 | 结果 |
|---|---|
| `pytest tests/ -q` | PASS：950 passed, 4 skipped，379.81 秒 |
| migration check | PASS：366 nodes、428 edges、856/856 walkable directions 有照片、scripted 2/2 delivered |
| deterministic replay | PASS：28/28 assertions，text/visual、两次 replay、clean process 都一致 |
| stress compiler | PASS：15/15 cases |
| compile all maps | PASS：10/10 usable envs |
| environment readiness | FAIL：`expected_images_per_turn` |
| baseline provenance | FAIL：43/49 assertions |
| R1 training gate | BLOCKED：缺少 `torch` |
| Qwen baseline CLI | BLOCKED：缺少 `torch` |

---

## 3. 详细问题列表

## DBG-001：失败的交互工具泄露精确目标距离

**严重级别：High**  
**组件：** `CourierEnv` 的 `collect()` / `hand_over()` outcome 文本  
**影响：** benchmark exploit、privileged state 泄露、导航评价失真

### 问题描述

当 agent 不在 pickup 或 dropoff 时，失败反馈会返回目标距离，而且是精确到米的 metric distance。agent 可以反复调用 `collect()` 或 `hand_over()`，把工具当作距离传感器使用，而不需要从地图、照片、街道名称或门牌号中推理。

这与 `docs/EVAL_BRIEF.md` 中“旧版 `collect()` 会泄露 exact distance，已经移除”的描述矛盾。

### 实际例子 1：pickup 泄露

当时公开 observation：

```text
### your notes
Job: collect from 5 Rue Saint-Antoine

### where you are
You are on Rue Saint-Antoine.

### what just happened
Down Rue Saint-Antoine the doors read 6.
```

执行：

```text
THOUGHT: The doors immediately behind read 6, so number 5 should be adjacent.
```
```python
collect()
```

实际反馈：

```text
You are not at 5 Rue Saint-Antoine. It is 36 m away; you need to be within 8 m.
```

### 实际例子 2：dropoff 泄露

公开 observation：

```text
You are on Rue Oberkampf, outside number 18.
Job: deliver to 17 Rue Oberkampf
```

执行：

```python
hand_over()
```

实际反馈：

```text
You are not at 17 Rue Oberkampf. It is 36 m away; you need to be within 8 m.
```

### 预期行为

失败反馈不应该提供 evaluator-only 的精确距离。建议只返回：

```text
You are not at the pickup address.
```

如果希望提供帮助，可以提供基于 agent 已有 observation 的粗粒度信息，例如：

```text
You are on the correct street but not at the requested door.
```

但不应该返回精确米数，也不应该返回目标 node、方向或最短路径。

### 可能原因

交互 outcome 直接调用了环境内部的目标距离计算，并把结果格式化进公开文本；公开 feedback 和 evaluator diagnostics 没有分层。

### 修复建议

1. 把 outcome 拆为：
   - `public_message`
   - `privileged_diagnostics`
2. trajectory 的 evaluator 私有部分可以保存精确距离，但不能进入 policy observation。
3. 对所有 rejected tool 做信息流审计，不只检查 `collect()`。

### 建议回归测试

```python
def test_rejected_door_interactions_do_not_leak_metric_range():
    outcome = env.collect()  # agent is not at pickup
    assert " m away" not in outcome.message
    assert not re.search(r"\b\d+(?:\.\d+)?\s*m\b", outcome.message)
```

同时为 `hand_over()` 添加相同测试。

---

## DBG-002：文档中的 reference policy 示例缺少必要的 `env.reset()`

**严重级别：High**  
**组件：** `docs/RUNNING.md`、`ObservationOnlyCourier.run()` 生命周期  
**影响：** 用户无法复现实验、reference score 可能为零、文档 smoke test 失败

### 问题描述

文档展示：

```python
ObservationOnlyCourier(env, max_steps=9000).run(seed)
```

如果按照这段 reference-policy 代码独立调用，而没有先显式执行 `env.reset()`，结果不是运行 episode，而是返回零步失败。

### 实际例子

```python
env = CourierEnv(net, seed=1, difficulty="pair", stride="block", ...)
result = ObservationOnlyCourier(env, max_steps=9000).run(1)
print(result)
```

实际结果：

```python
{
    "delivered": False,
    "deliveries": 0,
    "steps": 0,
    "sim_seconds": 0.0,
    "trace": ["no order was generated"],
}
```

显式执行：

```python
env.reset()
result = ObservationOnlyCourier(env, max_steps=9000).run(1)
```

之后 policy 才真正开始运行。

### 预期行为

以下两个方案选择其一，并在所有 API 中保持一致：

1. `run(seed)` 自己负责 reset；或
2. `run()` 明确要求已 reset，并在未 reset 时抛出清晰异常。

不应该静默返回“no order was generated”，因为这看起来像 episode 生成器的合法失败。

### 修复建议

- 推荐让 `ObservationOnlyCourier.run(seed)` 调用 `env.reset(seed=seed)`。
- 如果为了复用当前 env state 不允许自动 reset，则重命名为 `run_current_episode()`，并在未初始化时抛 `RuntimeError("Call env.reset() before run_current_episode()")`。
- 更新 `docs/RUNNING.md` 中每一段可独立复制的示例。

### 建议回归测试

```python
def test_documented_reference_policy_example_runs_verbatim(paris):
    env = CourierEnv(paris, seed=1, difficulty="pair", stride="block")
    result = ObservationOnlyCourier(env, max_steps=9000).run(1)
    assert result.steps > 0
    assert "no order was generated" not in result.trace
```

此外建议在 CI 中真正执行所有文档 code block。

---

## DBG-003：block stride 下 route/junction/distance 的语义不一致

**严重级别：High / Medium**  
**组件：** route rendering、candidate rendering、block-stride transition  
**影响：** agent 无法判断自己是否走错；训练数据存在矛盾 supervision

### 问题描述

当前 observation 同时出现以下概念：

- `next junction 18 m`
- route 中的 `1 junction, 29 m`
- action outcome 中的 `through 2 junctions`
- block stride 的“一次 call 走一个 block”
- graph waypoint
- 真实有分叉或红绿灯的 decision stop

这些概念对实现者可能有明确定义，但 policy-visible 文本没有稳定地区分它们。

### 实际例子 1：18 m、29 m、2 junctions 同时出现

初始 observation：

```text
1. Quai Beaubourg — heading north-east — next junction 18 m
2. Quai Beaubourg — heading south-west — next junction 18 m
```

调用 `navigate()`：

```text
Take Quai Beaubourg — south-west — 1 junction, 29 m.
```

随后执行 `walk_to(2)`：

```text
You walk 29 m along Quai Beaubourg, through 2 junctions.
```

agent 会自然地问：route 说的 1 junction 和 outcome 说的 2 junctions，哪个才是同一单位？candidate 的 18 m 又指哪个停止点？

### 实际例子 2：route 说 36 m/1 junction，但一次 action 只走 18 m

在 Rue Oberkampf：

```text
Route to 17 Rue Oberkampf — 36 m.
1. Take Rue Oberkampf — north-west, behind you — 1 junction, 36 m.
```

执行一次 `walk_to(2)` 后：

```text
You walk 18 m along Rue Oberkampf, through 1 junction.
You are on Rue Oberkampf, outside number 15.
```

还需要再执行一次 `walk_to(4)` 才到 17 号。也就是说 route 的 “1 junction” 实际需要两个 agent action。

### 实际例子 3：route 说 3 junctions，block action 走过 4 junctions

```text
Turn left onto Rue Saint-Antoine — north-west overall — 3 junctions, 151 m.
```

但一次进入 Rue Saint-Antoine 的 block action 返回：

```text
You walk 61 m along Rue Saint-Antoine, through 4 junctions.
```

### 预期行为

公开文本需要明确且一致地使用以下术语，例如：

- `next waypoint`：下一张相机/graph waypoint
- `next decision stop`：下一次 agent 必须作决策的位置
- `street intersections crossed`：途中经过多少真实分叉
- `calls`：在当前 stride 下预计需要多少次 `walk_to`

block stride 的 route 最好直接说：

```text
Continue on Rue Oberkampf for about 36 m. Expect 2 walk actions because an
intermediate decision stop may interrupt the block.
```

或者 route 不预测 call 数，只给距离和街道转向。

### 可能原因

`route_legs()` 在 graph edge、degree-2 waypoint 和 block stopping rule 之间做计数转换，但实际 stop rule 还会受到：

- side street
- fork
- target door
- traffic signal
- obstacle
- street identity change

影响，因此静态 route leg 的 junction 计数不能完全预测 runtime action 数。

### 修复建议

1. 在 schema 中正式定义所有距离和 junction 字段。
2. route 只报告可严格验证的量。
3. 对每个 origin/goal、stride、hazard 配置进行 property test：route 预测和实际 rollout 一致。
4. 如果 hazard 会改变停止点，应明确 route 是 clean-map 估计。

### 建议回归测试

```python
@pytest.mark.parametrize("stride", ["waypoint", "block"])
def test_route_leg_units_match_runtime_semantics(paris, stride):
    # Enumerate many reachable start/goal pairs.
    # Execute the route in a clean environment.
    # Assert reported calls/junctions use their documented definitions.
    ...
```

---

## DBG-004：门牌号提示与目标位置会产生反直觉或矛盾感

**严重级别：Medium**  
**组件：** `look(k)`、address placement、block stopping  
**影响：** agent 在目标街道附近浪费调用；door localization 训练信号不稳定

### 问题描述

agent 在正确街道和相邻门牌附近时，公开门牌号提示不一定足以判断目标方向。奇偶号、node address 和 graph 方向之间表现得不直观。

### 实际例子

目标：

```text
deliver to 17 Rue Oberkampf
```

当前位置：

```text
You are on Rue Oberkampf, outside number 18.
```

尝试 `hand_over()` 被拒，并泄露 36 m。

然后：

```text
look(1) -> Down Rue Oberkampf the doors read 13, 16 falling.
look(2) -> Down Rue Oberkampf the doors read 15 falling.
```

两个方向都显示“falling”，且当前就在 18 号，但目标 17 号实际需要向 north-west 走 36 m。agent 最终必须重新调用 `navigate()` 才能决定方向。

### 预期行为

- 奇数和偶数两侧的门牌序列应分别明确。
- `look()` 应说明该方向可见的号码以及单调趋势，例如：

```text
Odd side: 15, 17 increasing.
Even side: 16, 14 decreasing.
```

- 如果 CityCore 视觉里没有真实可读门牌，应明确这是 survey/tool 信息，而不是视觉 OCR。

### 修复建议

1. 分离 odd/even side 的序列。
2. 检查 address-to-node 绑定是否符合沿街单调顺序。
3. 添加“当前门牌相邻目标门牌”的覆盖测试。
4. 在 block stride 下，目标 door 应强制成为 stop，不能无提示地跨过。

### 建议回归测试

对每一条 street 的两个方向检查：

- 门牌趋势单调；
- odd/even 不混淆；
- 从相邻号码出发，公开 observation 能唯一决定目标方向；
- route 到达 target door 时不会越过。

---

## DBG-005：唯一 validated condition 不是 perception-essential

**严重级别：High（研究有效性）**  
**组件：** `Condition`、benchmark claims、evaluation protocol  
**影响：** benchmark 不能证明 agent 具有通用视觉导航能力

### 问题描述

源码中：

```python
Condition.ALL = ("full", "no_phone", "visual")
Condition.VALIDATED = ("full",)
```

同时源码说明：

- `full` 是 text-only policy 的 solvability floor；
- `no_phone` 当前 reference 表现不足，未 validated；
- `visual` 因门牌号在 CityCore render 中不可读，被认为 information-incomplete / impossible。

因此当前唯一可以正式报告 score 的 condition，本来就可以不依赖视觉完成。hazard 图片确实有价值，但 benchmark 的主有效 rung 不能支持“agent 从视觉中完成一般导航”的结论。

### 实际例子

人工 episode 中图片有两个真实作用：

1. Boulevard du Temple 的图片显示 road block，避免了 collision。
2. Rue de Sévigné 的 signal frame 显示红灯，等待后变绿。

但 pickup/dropoff、route、街道名、方向、距离、门牌工具仍主要来自文本和 phone。这说明当前图片测量的是“hazard recognition 子任务”，不是完整的 vision-grounded delivery。

### 预期行为

至少应有一个 validated condition 满足：

- 删除图片后性能显著下降；
- shuffle 图片后性能显著下降；
- 仅靠文本无法恢复视觉标签；
- 多个人类和多个 VLM 能完成；
- 图片中信息清晰、可读、无 filename 泄露。

### 修复建议

可以先建立较窄、但严格有效的视觉 rung：

- 文本提供 route 和地址；
- 只有图片提供 barrier/slow obstacle/signal 状态；
- 每个 episode 强制至少一个视觉关键决策；
- score 单独报告 hazard avoidance、signal compliance 和 delivery success。

之后再开发 storefront/door OCR condition。

### 建议实验

至少报告：

| Ablation | 预期 |
|---|---|
| 正常图片 | 主结果 |
| 删除图片 | 明显下降 |
| 图片 shuffle | 明显下降 |
| 只给 caption | 判断 caption 是否泄露 |
| oracle hazard labels | perception ceiling |
| text-only policy | planning floor |

---

## DBG-006：图片文件路径包含 ground-truth 语义标签

**严重级别：Medium / High（取决于 adapter）**  
**组件：** `Frame.path`、model adapter、trajectory export  
**影响：** 如果路径字符串进入模型上下文，视觉任务可被直接作弊

### 问题描述

图片路径直接包含 hazard 或 signal 标签，例如：

```text
/data/murray/paris_obstacles/citycore-paris/images/
s004_n011/toward_s004_n008_road_block.png
```

```text
/data/murray/paris_signals_kerb/citycore-paris/images/
s005_n012/toward_s005_n011_red.png
```

```text
.../toward_s007_n011_slow_pedestrian.png
```

人工测试时我直接查看了 pixels，没有用文件名决策。但是任意 adapter、logger、chat template 或 dataset serializer 如果把 path 作为文字传给模型，就会泄露答案。

### 预期行为

policy 只能收到图片 bytes/tensor 和中性 frame ID，例如：

```text
frame_000123.png
```

ground-truth label 可以保留在 evaluator-only manifest 中。

### 修复建议

1. 导出 benchmark artifact 时使用 content hash 或随机中性 ID。
2. adapter 层禁止把本地 path 拼进 prompt。
3. trajectory 对 policy-visible media 和 evaluator metadata 分层。
4. 对 JSONL/Parquet/verl DataProto 导出做信息泄露扫描。

### 建议回归测试

```python
FORBIDDEN = ("red", "green", "road_block", "slow_pedestrian")
serialized = serialize_policy_input(observation)
assert not any(word in serialized.lower() for word in FORBIDDEN)
```

---

## DBG-007：system prompt 和每回合 observation 存在重复与成本浪费

**严重级别：Medium**  
**组件：** courier prompt、memory rendering、media resend  
**影响：** 训练成本、推理延迟、上下文污染、长 episode 稳定性

### 问题描述

environment readiness 测得 system prompt：

```text
5917 chars (~1479 rough tokens)
```

每回合还会重复发送：

- 完整 notes；
- 最近 trail；
- streets seen；
- 全部 candidate captions；
- 同一批 street photographs；
- 手机地图及“这是 drawing，不是 photograph”的解释；
- 上一回合 outcome。

这些设计单独看都有理由，但在 34-turn solo 或数百 turn pair episode 中会产生明显重复。

### 实际例子

在 pickup 后的多个 turn 中，同一张 Rue Saint-Antoine 照片被重复放入 observation；手机 map 开启后每个 turn 都重新生成并附加。`Came from` 和 `Streets you have seen` 也在多个 turn 中大段重复。

### 预期行为

benchmark 应明确计算并报告：

- input tokens；
- output tokens；
- image count；
- image pixels；
- 每个 delivered order 的上下文成本。

可以考虑：

- system prompt cache；
- delta observation；
- 图片 content hash 去重；
- bounded/relevance-based memory；
- map frame 只在变化时发送，或使用 persistent UI contract。

### 修复建议

不要仅通过删掉重要信息来降 token。先做数据统计：

1. 每类字段的 token 占比；
2. 重复 n-gram 和重复 image hash；
3. 移除某字段后的 agent 性能 ablation；
4. 不同模型 context-cache 的公平性规则。

### 建议回归测试

为一个冻结 trajectory 计算：

```text
total_input_tokens
unique_image_hashes / total_image_attachments
memory_tokens_per_turn
map_bytes_per_turn
```

设置公开预算或至少加入 report。

---

## DBG-008：environment readiness gate 当前失败

**严重级别：High（release blocker）**  
**组件：** `embodiedbench.verification.env_readiness`  
**影响：** 当前版本不能宣称所有 release gate 通过

### 实际命令

```bash
python3 -m embodiedbench.verification.env_readiness --out <tmpdir>
```

### 实际结果

```text
FPV manifest not found at deliverybench_fpv/small-city-11-new/manifest.jsonl; skipping.
...
turns: 6, images: 6
[PASS] system_prompt_non_empty
[PASS] every_turn_has_at_least_one_image
[PASS] no_blank_images
[PASS] fpv_album_loaded
[FAIL] expected_images_per_turn
[PASS] observation_text_non_trivial
[PASS] no_unknown_config_keys
ENV readiness: FAIL (1 failed)
```

### 需要确认的问题

`expected_images_per_turn` 的预期值是过时了，还是实际 observation 少图？不能简单删除 assertion。需要检查：

- 配置声明的 camera 数；
- 每回合应有的 FPV 数；
- map image 是否计入；
- fallback album 是否改变了数量；
- skip 后为什么 `fpv_album_loaded` 仍 PASS。

### 修复建议

让 report 输出每一 turn：

```json
{
  "step": 3,
  "expected_images": 2,
  "observed_images": 1,
  "expected_kinds": ["fpv", "map"],
  "observed_kinds": ["fpv"]
}
```

这样失败能直接定位，而不是只有一个汇总布尔值。

### 建议回归测试

在默认 readiness 配置上 assertion 必须通过；如果资源缺失，应是明确 `SKIP` 或 `BLOCKED`，不能一部分检查使用 fallback、一部分仍按原配置判断。

---

## DBG-009：baseline provenance 只有 43/49 assertions 通过

**严重级别：High（reproducibility/release）**  
**组件：** baseline manifest、local roots、tool environment  
**影响：** 当前运行环境不能复现 pinned baseline

### 实际失败

```text
FAIL spear_unrealcv_plugin.tree[structure].digest
expected: 802217e1...
observed: 0836f86d...
```

```text
FAIL spear_unrealcv_plugin.tree[structure].file_count
expected: 16
observed: 142
```

```text
FAIL spear_unrealcv_plugin.tree[content].digest
expected: 2d5c8c3b...
observed: b275a26e...
```

```text
FAIL spear_unrealcv_plugin.tree[content].file_count
expected: 16
observed: 142
```

```text
FAIL citycore_paris_content.exists
expected: True
observed: False
```

```text
FAIL verification_tool_env.env.python
expected: 3.11.15
observed: 3.12.3
```

### 预期行为

发布版本的 baseline provenance 应该全部通过。若某项是本地 optional resource，应明确标为 `SKIP` 或 `UNAVAILABLE`，并从 mandatory pass denominator 中分离。

### 修复建议

1. 确认 plugin tree 是合法升级还是意外 drift。
2. 如果是合法升级，重新生成 manifest，并记录 review/approval。
3. 修复 CityCore content root，避免依赖过时绝对路径。
4. 提供受支持的 Python 环境，例如 lockfile/container。
5. 在 report 中区分：
   - integrity failure
   - resource unavailable
   - version mismatch

### 建议回归测试

release CI 必须在 clean environment 上得到 49/49；否则阻止发布 artifact。

---

## DBG-010：训练和 Qwen baseline CLI 缺少声明的 `torch` 依赖

**严重级别：High（可运行性）**  
**组件：** `pyproject.toml` optional dependencies、training CLI、model adapter  
**影响：** 用户按项目依赖安装后仍无法运行训练 gate 或 baseline

### 实际例子 1：R1 gate

```bash
python3 -m embodiedbench.training.r1_gate --steps 4
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
```

### 实际例子 2：Qwen baseline

```bash
python3 -m embodiedbench.verification.baseline --episodes 3 --no-vision
```

在真正加载模型前就失败：

```text
from embodiedbench.agent.model_adapters.qwen3vl import ...
import torch
ModuleNotFoundError: No module named 'torch'
```

### 根因

当前 `pyproject.toml`：

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "pandas", "pyarrow"]
```

没有训练或 VLM optional group，也没有把 `torch` 声明为 CLI 前置依赖。

### 修复建议

新增：

```toml
[project.optional-dependencies]
train = ["torch>=...", ...]
vlm = ["torch>=...", "transformers>=...", ...]
all = ["embodiedbench[dev,train,vlm]"]
```

并在 CLI import 前做 preflight，输出：

```text
Qwen baseline dependencies are missing.
Install with: pip install -e '.[vlm]'
```

不要直接暴露 Python traceback 作为普通用户体验。

### 建议回归测试

- minimal install：核心包可以 import，不需要 torch；
- `.[train]`：R1 smoke test 可运行；
- `.[vlm]`：adapter 可以 import；
- 缺依赖时 CLI 返回清晰说明和固定 exit code。

---

## DBG-011：文档默认使用 `python`，但提供环境只有 `python3`

**严重级别：Medium**  
**组件：** `docs/RUNNING.md`、`docs/EVAL_BRIEF.md`、环境说明  
**影响：** copy/paste 命令立即失败

### 实际例子

文档命令：

```bash
python -m pytest tests/ -q
python -m embodiedbench.tools.migration_check
```

当前环境结果：

```text
/bin/bash: python: command not found
```

可用解释器：

```text
/usr/bin/python3
Python 3.12.3
```

但 system Python 还缺少项目 dependencies，且 `python3 -m venv` 因 `ensurepip` 缺失而失败。

### 修复建议

推荐文档统一使用受控 runner：

```bash
uv run pytest tests/ -q
```

或：

```bash
python3 -m pytest tests/ -q
```

更重要的是提供 lockfile/container，而不是仅替换命令名字。

### 建议回归测试

在 release container 中逐条执行 README/RUNNING/EVAL_BRIEF 命令。

---

## DBG-012：reference score 表格没有充分绑定配置与 artifact hash

**严重级别：Medium / High（论文结果可信度）**  
**组件：** `docs/RUNNING.md`、source docstrings、结果生成流程  
**影响：** 不同 revision、stride、hazard、reset 方法的数字容易混用

### 问题描述

`docs/RUNNING.md` 中展示六个 seed 的结果：

```text
ceiling: pair 12/12
floor blind: pair 8/12
floor reads frames: pair 8/12
```

源码 `Difficulty` docstring 又有 25/30 seed 的另一组百分比。实测 seed 1、pair、block、all hazards：

| policy | delivered | walked | blocked attempts |
|---|---:|---:|---:|
| blind observation-only | 0/2 | 5,556.5 m | 53 |
| sighted observation-only | 0/2 | 9,240.6 m | 0 |

单个 seed 不能推翻六个 seed 或 30 个 seed 的统计，但足以说明表格必须明确绑定：

- git revision；
- map hash；
- album hashes；
- seed list；
- difficulty；
- stride；
- condition；
- hazard on/off；
- policy revision；
- 是否先 reset；
- max steps 和 termination rule。

### 修复建议

结果表不应手工复制。每次 benchmark run 生成 machine-readable result card：

```json
{
  "git_commit": "...",
  "map_hash": "...",
  "album_hashes": {},
  "policy": "observation_only@...",
  "seeds": [0, 1, 2, 3, 4, 5],
  "difficulty": "pair",
  "stride": "block",
  "condition": "full",
  "hazards": ["signals", "obstacles"],
  "delivered": 8,
  "issued": 12
}
```

Markdown 表应从 JSON 自动生成。

### 建议回归测试

CI 检查文档中的 benchmark table 与冻结 result artifact hash 一致。

---

## DBG-013：`optimal_walk_m` 可能不是严格下界，命名容易误导

**严重级别：Medium**  
**组件：** summary metrics  
**影响：** route efficiency 解释错误

### 实际例子

成功 solo summary：

```text
walked_m = 537.9
optimal_walk_m = 562.7
walk_ratio = 0.96
```

如果 `optimal_walk_m` 真的是同一地图、同一任务 endpoint、同一 distance semantics 下的最短可行路径，那么实际路径不应该比它更短。

可能的合理解释包括：

- optimal 使用不同 pickup/dropoff tolerance；
- optimal 包含 clean-route 或 route-leg 语义；
- 实际 movement distance 与 graph path distance 计算不同；
- hazard detour或 block interpolation 改变 endpoint；
- optimal 是 comparator，不是数学下界。

### 预期行为

如果不是严格下界，应重命名，例如：

```text
reference_route_m
clean_router_m
```

如果应该是严格下界，应修复计算并加入 assertion：

```python
assert optimal_walk_m <= walked_m + tolerance
```

### 建议调试步骤

对 seed 0 solo 输出：

- pickup/dropoff node；
- arrival tolerance；
- oracle node path；
- 每条 edge 距离；
- agent 实际 transition 距离；
- 两者采用的坐标与单位。

---

## DBG-014：environment summary 的 `finished` 与 session termination 语义不够统一

**严重级别：Medium**  
**组件：** `CourierSession.finished`、`CourierEnv.summary()['finished']`、shift termination  
**影响：** evaluator 可能把 timeout、shift-over、all-delivered 混淆

### 实际例子

pair reference run 达到：

```text
shift_over: True
finished: False
sim_seconds: ~8808.9
```

同时 policy `OracleResult` 已经返回。也就是说 runner 停止了，但 environment summary 仍报告 `finished=False`。

### 预期行为

建议拆分：

```text
episode_terminal: bool
episode_truncated: bool
termination_reason: all_delivered | deadline | shift_over | step_budget | format_errors | ...
task_complete: bool
```

不要只用一个 `finished` 同时表示 task completion 和 runner termination。

### 建议回归测试

分别测试：

- 全部交付；
- shift 时间结束；
- step budget；
- tool budget；
- repeated format errors；
- endless hour 结束。

每种状态必须有唯一、稳定的 terminal/truncated/reason 组合。

---

## DBG-015：sighted reference 虽然避免碰撞，但在实测 seed 上走得更远且没有交付

**严重级别：Medium（reference policy/debug signal）**  
**组件：** `ObservationOnlyCourier(..., sighted=True)`  
**影响：** perception ceiling 的解释可能不成立；绕障策略可能存在 livelock

### 实际结果

seed 1、pair、block、all hazards：

```text
blind:
  deliveries = 0
  walked_m = 5556.5
  blocked_attempts = 53

sighted:
  deliveries = 0
  walked_m = 9240.6
  blocked_attempts = 0
```

正面结果是 sighted policy 确实把 barrier collisions 降到了 0。问题是，它没有把视觉识别转化为更好的 task success，反而走得更远。

### 可能原因

- 识别 block 后缺乏有效 detour planning；
- policy 避开 blocked edge，但没有持久化“这个方向通向死路”；
- block stride 和 candidate reindex 造成重复选择；
- obstacle-aware local choice 与全局 route 冲突；
- termination/queue handling 使它持续 wander。

### 修复建议

reference trace 应记录结构化原因：

```json
{
  "node": "...",
  "goal": "...",
  "route_first_edge": "...",
  "visible_blocked_edges": ["..."],
  "chosen_edge": "...",
  "memory_before": {},
  "memory_after": {},
  "decision_reason": "detour"
}
```

用 cycle detection 分析重复 state `(node, goal, blocked-memory)`。

### 建议回归测试

构造一个只有一条简单 detour 的 map：

- blind policy 会撞 barrier；
- sighted policy 必须零碰撞并成功到达；
- sighted path length 有合理上界。

---

## DBG-016：`follow_street` 的 prompt 示例可能与当前允许工具不一致

**严重级别：Low / Medium**  
**组件：** system prompt reply-format 示例、dynamic tool menu  
**影响：** agent 模仿示例后可能产生 unavailable tool

### 问题描述

block stride 的正式 action menu 不提供 `follow_street`，这是合理的，因为 block stride 已经替代该 macro。但通用回复格式说明中仍出现：

```text
Whole numbers go bare: walk_to(2), follow_street(2, 4).
```

同时 skills 部分也可能提到：

```text
take that one, with follow_street(k, n) if the route says to stay on it...
```

实际人工 block prompt 的 action list 中没有 `follow_street`，但其他段落仍提到了它。对倾向模仿 example 的模型来说，这是一种矛盾 supervision。

### 预期行为

system prompt 中所有 action name 都必须从当前 `allowed tools` 动态生成，包括：

- menu；
- procedures；
- formatting examples；
- error recovery examples。

### 建议回归测试

提取 system prompt 中所有形如 `name(...)` 的 token：

```python
mentioned = extract_tool_names(prompt)
assert mentioned <= set(session.allowed)
```

对每个 condition 和 stride 运行。

---

## DBG-017：真实图片存在取景、可读性和边界质量问题

**严重级别：Medium（视觉测量质量）**  
**组件：** cached FPV album、camera pose、signal crops  
**影响：** 视觉错误可能来自 render，而不是 agent 能力

### 实际观察

1. 一些标为“looking down street”的图片主要面对建筑立面，而不是街道纵深。
2. 红绿灯在 1200×843 图中占比很小，虽然人工放大后可读，但不同 VLM 输入缩放后可能不可读。
3. 地图边缘出现平坦灰色区域和天空，视觉上像道路断头，但 graph 可能仍继续。
4. 两条接近同一 bearing 的街道图片很相似，主要依靠文字 caption 区分。
5. 某些照片中的真实交通灯是旧 street render；实际控制状态在单独 signal frame。虽然 prompt 解释了，但仍增加视觉冲突。

### 具体图片例子

```text
/data/murray/paris_streets_v2/citycore-paris/images/
s014_n003/toward_s007_n000.png
```

该图片主要呈现正面的 storefront/facade，作为“looking down Rue Saint-Antoine”的方向证据较弱。

信号例子：

```text
/data/murray/paris_signals_kerb/citycore-paris/images/
s005_n012/toward_s005_n011_red.png
```

红色 pedestrian figure 可见，但目标像素区域较小。

### 修复建议

1. 对每张 FPV 保存 camera pose、target bearing、street centerline deviation。
2. 自动检查道路 vanishing direction 是否位于合理图像区域。
3. 统计 signal lamp 在模型实际 resize 后的像素面积。
4. 做 human/VLM legibility study，而不是只检查文件存在。
5. 对地图边缘添加明确视觉边界或避免把 sky/grey 当作可走方向。

### 建议回归测试

- camera yaw 与 edge bearing 误差阈值；
- signal lamp minimum pixels after resize；
- obstruction bounding box 可见率；
- random sample 人工审核表。

---

## DBG-018：当前“Paris”命名可能造成真实地理能力的过度解读

**严重级别：Critical（部署声明）**  
**组件：** benchmark positioning、map provenance、deployment claims  
**影响：** 研究和安全风险

### 问题描述

当前环境包含巴黎风格街景和巴黎街道名称，但人工路线中观察到的连接关系不应被当成真实巴黎路网。例如人工 detour 是：

```text
Rue de Sévigné
-> Rue de la Roquette
-> Avenue de Crimée
-> Rue Oberkampf
```

这是 CityCore synthetic topology 中的可行路线，不代表现实巴黎中的真实相邻关系、距离或行人通行规则。

当前未验证：

- GPS/georeference；
- 当前 OpenStreetMap/官方道路数据一致性；
- 人行道、路缘、无障碍通行；
- 自行车、机动车、行人动态；
- 夜间、雨雪、施工变化；
- 法国交通法规和配送规则；
- physical robot/avatar dynamics；
- localization drift；
- emergency stop、handoff 和 remote supervision；
- 真实 Paris field trial。

### 预期行为

报告和 README 应明确：

```text
CityCore Paris is a synthetic Paris-themed simulation environment. Benchmark
success does not qualify an agent for autonomous deployment in Paris.
```

### 后续真实部署前置条件

1. 单独建立真实巴黎 georeferenced benchmark。
2. 法律、隐私、数据许可和安全审查。
3. shadow-mode，不控制真实机器人。
4. 人类监督的小范围封闭场地测试。
5. 行人环境的 hazard analysis 和 fail-safe。
6. 明确 OOD 检测与人工接管。
7. 逐阶段安全批准，不能从 simulation score 直接跳到部署。

---

## DBG-019：release gate 没有统一入口和清晰的 required/optional 分类

**严重级别：High**  
**组件：** verification tooling、CI/release process  
**影响：** 某些检查通过会掩盖其他关键 gate 失败

### 实际状态示例

同一 revision 上：

```text
pytest: PASS
migration: PASS
stress: PASS
replay: PASS
compile maps: PASS
env readiness: FAIL
baseline provenance: FAIL
R1 gate: BLOCKED
Qwen baseline: BLOCKED
```

如果用户只运行 `pytest`，会认为版本完全健康；如果只运行 migration check，也会看到：

```text
the move is complete: the world builds, the pictures are there, and an episode runs end to end
```

但 release 仍有 readiness 和 provenance failure。

### 修复建议

提供单一命令，例如：

```bash
python -m embodiedbench.verification.release_gate --profile cached-paris
```

输出：

```text
REQUIRED
  tests              PASS
  migration          PASS
  readiness          FAIL
  provenance         FAIL
  replay             PASS

OPTIONAL: LIVE_UE
  ue_probe            UNAVAILABLE

OPTIONAL: TRAINING
  r1_gate             MISSING_DEPENDENCY

RELEASE: FAIL
```

所有 conditional absence 必须是 `SKIP/UNAVAILABLE/BLOCKED`，不能计为 PASS。

### 建议回归测试

release gate 自身应有测试，确保任一 required child gate 非 PASS 时总状态为 FAIL。

---

## DBG-020：配置空间需要生成式 compatibility matrix，而不是分散在文档和源码中

**严重级别：Medium / High**  
**组件：** difficulty、condition、stride、runtime、profile、map  
**影响：** 用户可能运行语义无效组合，并错误报告结果

### 当前主要轴

```text
difficulty: solo, pair, triple, shift, endless
stride: waypoint, block
condition: full, no_phone, visual
runtime: text, cached, live
hazards: none, signal-only, obstacle-only, both
courier profile: walker_novice, scooter_standard, multi_modal_courier
maps: 10 compiled environments
```

并非所有 Cartesian product 都有效。例如：

- `visual` 已声明但不可报告；
- live runtime 在当前 audit 环境不可用；
- multi-modal courier 需要相应 affordances；
- cached images 依赖具体 album；
- signal mechanics 没有 signal album 时应关闭；
- obstacle mechanics 没有 obstacle album 时应关闭。

### 修复建议

构建机器可读 matrix：

```json
{
  "map": "citycore-paris",
  "task": "delivery",
  "profile": "walker_novice",
  "runtime": "cached",
  "condition": "full",
  "stride": "block",
  "status": "validated",
  "required_assets": ["street_album", "signal_album", "obstacle_album"],
  "evidence": ["result-card-hash"]
}
```

CLI 在运行前校验 matrix；不允许用户无提示运行 unsupported/unvalidated 配置。

### 建议回归测试

- 每个 declared combination 必须有明确状态；
- matrix 中不能有“默认未知”；
- validated cell 必须引用通过的 artifact 和 result card。

---

## 4. 正面发现：建议保留的设计

以下设计对 benchmark 质量非常重要，修 bug 时不要破坏：

### 4.1 Tool prompt 和 dispatch 共用 registry

`CourierSession` 从同一个 allowed-tool 集合生成 prompt 和 dispatch table，可以避免“prompt 宣传了环境不能执行的工具”。这是正确方向。

### 4.2 Photograph 和 map 使用不同类型

```python
Frame(kind="photograph")
Frame(kind="map", svg=...)
```

这个区分非常重要。手机地图是 survey drawing，不能看到 barrier 或 signal；street photograph 才是视觉 observation。

### 4.3 Trajectory 保存的是 agent 真正看到的内容

Turn log 保存：

- 完整 prompt；
- 有序 image paths；
- reply；
- parsed action；
- accepted/rejected；
- simulated cost；
- reward；
- memory。

这为 replay、训练和错误分析提供了良好基础。

### 4.4 确定性 replay 很强

实测：

```text
PASS: 28/28 replay assertions passed in 78.993s
```

text 和 visual episode 都记录后 replay 两次，并额外在 clean process 中比较。这个 gate 应保留为 mandatory release gate。

### 4.5 Stress compiler 覆盖了多种坏地图

实测 15/15，包括：

- no roads；
- disconnected islands；
- degenerate segments；
- self loops；
- star hub；
- very long segments；
- duplicate positions；
- nonfinite coordinates；
- no POIs；
- malformed/unknown asset cases。

这类“坏输入也要得到正确失败”的测试非常有价值。

### 4.6 对未验证 condition 的态度是诚实的

源码明确写出 `visual` 目前不可报告，`no_phone` 也因为 solvability 证据不足从 validated 中移除。这种诚实性应保留，不要为了扩大 benchmark scope 而把未验证配置标为正式结果。

---

## 5. 推荐修复顺序

### P0：立即修复，阻止 benchmark exploit 或错误 release

1. **DBG-001**：移除 exact-distance refusal leak。
2. **DBG-008**：修复 environment readiness failure。
3. **DBG-009**：恢复或更新 provenance，明确 mandatory roots。
4. **DBG-019**：建立统一 release gate。
5. **DBG-002 / DBG-010 / DBG-011**：保证文档和依赖可从 clean environment 运行。

### P1：决定 benchmark 是否真的测到了目标能力

1. **DBG-005**：建立 perception-essential validated condition。
2. **DBG-006**：消除 filename/metadata label leakage。
3. **DBG-012**：冻结、自动生成 reference result card。
4. **DBG-020**：发布 configuration compatibility matrix。

### P2：改善 agent 体验和训练信号

1. **DBG-003**：统一 route/junction/stride 语义。
2. **DBG-004**：修复门牌方向和 target stopping。
3. **DBG-007**：减少重复 observation，并报告 token/image 成本。
4. **DBG-014**：统一 termination/truncation 状态。
5. **DBG-015**：分析 sighted oracle 的 wander/livelock。
6. **DBG-016**：dynamic tool example。
7. **DBG-017**：系统化视觉质量检查。

### P3：真实部署是独立项目

**DBG-018** 不能通过继续增加 unit tests 来解决。真实巴黎部署需要 georeferenced world、动态交通、物理 embodiment、安全 case、法规审查和 field trial。simulation benchmark 的 PASS 不能直接作为部署 gate。

---

## 6. 建议新增的最小 regression suite

建议新建 `tests/test_debug_regressions.py`，至少包含：

```python
def test_collect_refusal_does_not_reveal_exact_distance(): ...
def test_handover_refusal_does_not_reveal_exact_distance(): ...
def test_documented_observation_only_runner_initializes_episode(): ...
def test_prompt_mentions_only_currently_allowed_tools(): ...
def test_block_route_units_match_documented_semantics(): ...
def test_target_door_is_a_block_stride_stop(): ...
def test_door_numbers_are_directionally_monotonic_per_side(): ...
def test_policy_serialization_does_not_contain_media_labels(): ...
def test_all_validated_conditions_have_solvability_evidence(): ...
def test_release_gate_fails_when_readiness_fails(): ...
def test_terminal_and_truncated_reasons_are_unambiguous(): ...
def test_optimal_distance_is_lower_bound_or_named_reference(): ...
```

另外新增 integration tests：

```text
docs smoke test
clean installation smoke test
all CLI --help/import smoke test
cached Paris release gate
image-removal and image-shuffle ablation
simple obstacle-detour sighted-policy test
```

---

## 7. 修复完成的验收标准

下一版本至少达到以下条件后，才建议重新做完整 benchmark evaluation：

- [ ] `collect()` / `hand_over()` 等 rejected tool 不泄露 metric target distance。
- [ ] 文档所有命令在 clean supported environment 中逐条通过。
- [ ] `pytest`、migration、readiness、provenance、replay、map compile 全部通过。
- [ ] training/VLM optional dependencies 有明确安装方式和 preflight。
- [ ] block/waypoint route 单位有正式定义和 property tests。
- [ ] 每个 validated configuration 都有 machine-readable evidence。
- [ ] reference tables 从冻结 result artifacts 自动生成。
- [ ] policy-visible serialization 不包含 `_red`、`_green`、`road_block` 等标签。
- [ ] 至少一个 perception-essential condition 通过人类和多个 VLM 的 solvability 验证。
- [ ] 发布 compatibility matrix，unsupported 配置不能静默执行。
- [ ] README 明确说明 CityCore Paris 不等于真实巴黎部署能力。

---

## 8. 最终判断

当前 benchmark 已经是一个值得继续开发的 embodied-agent research prototype。它的 replay、schema、stress testing、trajectory contract 和对 unvalidated condition 的明确标注都很好。人工 episode 也证明 hazard 图片确实能改变决策：我避开了 road block，并正确等待了红灯。

但后续 debug 的重点不应该只是“让更多 tests 变绿”。必须优先确保：

1. policy 看不到 privileged answer；
2. observation 中的信息足够且不矛盾；
3. benchmark score 对应一个清晰、经过 ablation 验证的能力；
4. 所有结果可以从同一 revision 和 artifact 完整复现；
5. simulation 结论与真实巴黎部署结论严格分离。

处理完 P0 和 P1 后，再扩大 difficulty、模型数量和训练规模，得到的数字才具有稳定的研究意义。


---
---

# 第二部分：Agent 2（Claude Opus）独立复核报告

> 本部分由第二个 agent 独立完成，与上文 DBG-001…020 是**各自独立**跑出来的：
> 先按 `docs/EVAL_BRIEF.md` 的规则完整玩了两个 episode（全程只读
> `session.observe()`，不读源码、不跨回合持有 Python 对象），再把所有声明过的
> 配置跑了一遍。两份报告在若干条上互相印证，也有各自独有的发现。
>
> **本次只做汇总粘贴，不做任何代码修改。**
>
> - 复现脚本：`docs/review/tools/`
> - 原始数据：`docs/review/sweep.json`、`analysis.json`、`analysis2.json`、`episodes/{A,B}/`
> - 可视化轨迹（含真实图片）：`docs/review/report.html`
> - 叙述版：`docs/review/REPORT.md`
>
> 环境：`citycore-paris` — 366 nodes / 428 edges / 86 streets / 477 addresses /
> 105 signalised junctions；albums：streets 856 / signals 694 / obstacles 240，
> view coverage 100.0%；分支 `courier-environment` @ `5130047`；
> 解释器 `/data/murray/miniconda3/bin/python` (3.14.6)。

## A. 两份报告的对照表

| Agent 1 | Agent 2 | 是否一致 | 备注 |
|---|---|---|---|
| DBG-001 泄露精确距离 | A2-01 | ✅ 完全一致 | 两边都实测到 5 s、精确米数、可无限重复 |
| DBG-006 图片路径含标签 | A2-02 | ✅ 完全一致 | 我量化为 **539 帧中 215 帧（39.9%）** |
| DBG-004 门牌号反直觉 | A2-03 | ⚠️ 一致但根因不同 | 我定位到**单双号两侧沿弧长推进速度不同**，最坏 N 与 N+1 相隔 12 个路口 |
| DBG-005 validated condition 非 perception-essential | A2-04 | ⚠️ 一致但更严重 | 我发现支撑 `TIME_BUDGET_MULTIPLE=7.0` 的那张 sweep 表**用当前代码复现不出来** |
| DBG-016 `follow_street` 与允许工具不一致 | A2-05 | ✅ 完全一致 | block stride 下确认 `mentioned_but_not_callable: [follow_street]` |
| DBG-003 block stride 语义不一致 | A2-08 / A2-09 | ✅ 一致 | 我另外发现 heading 标签会**指向相反方向** |
| DBG-017 图片取景/可读性 | A2-06 / A2-18 | ✅ 一致 | 我量化为 **23.4% 的候选边 < 10 m**，镜头对着墙 |
| DBG-007 prompt/observation 冗余 | A2-16 | ✅ 一致 | 我量化为**逐行重复率 65–67%** |
| DBG-013 `optimal_walk_m` 误导 | A2-13 | ✅ 一致 | 我实测 episode 中途该值为 `0.0` |
| DBG-015 sighted 走更远却没交付 | A2-04 | ✅ 一致 | 20 seeds 下 sighted 在 waypoint 反而更差 |
| DBG-002 / 008 / 009 / 010 / 011 / 012 / 014 / 018 / 019 / 020 | — | Agent 2 未覆盖 | 我没有测 release gate / provenance / CLI 依赖 |
| — | A2-07 街道没有路牌 | Agent 1 未覆盖 | 决定了 `visual` 档位理论上无解 |
| — | A2-10/11/12/14/15/17/19 | Agent 1 未覆盖 | 见下文 |

**关于 DBG-010 / DBG-011 我可以独立确认：** 当前机器上 `python` 不在 PATH（只有
`python3`），且**任何 python 都没有安装 torch**（`/usr/bin/python3` 与
`/data/murray/miniconda3/bin/python` 均 `ModuleNotFoundError: No module named 'torch'`，
且 `miniconda3/envs/` 为空）。

## B. Agent 2 的 bug 列表

| id | sev | 组件 | 一句话 |
|---|---|---|---|
| A2-01 | S1 | runtime | `collect()`/`hand_over()` 拒绝时给出精确距离 —— 免费测距仪 |
| A2-02 | S1 | harness | 39.9% 的图片文件名直接写明 hazard；`RUNNING.md` 还教人把 path 传给模型 |
| A2-03 | S2 | compiler | 单双号两侧推进速度不同 —— 23 条街里 22 条显示序列有倒置 |
| A2-04 | S2 | tuning | 支撑 `TIME_BUDGET_MULTIPLE = 7.0` 的 sweep 表复现不出来 |
| A2-05 | S2 | prompt | block stride 的 prompt 宣传了无法调用的 `follow_street` |
| A2-06 | S3 | compiler | 23.4% 的候选是 < 10 m 的短桩，照片拍到的是对面墙 |
| A2-07 | S3 | assets | 全城没有任何路牌，街道身份永远只能靠文本 |
| A2-08 | S3 | runtime | 候选 heading 描述的是短桩，动作走的是整个 block —— 两者可指向相反方向 |
| A2-09 | S3 | runtime | route 的 junction 数与实际走完的结果不一致 |
| A2-10 | S3 | runtime | 撞到路障的提示说"你走回路口"，实际却把你留在两个路口之外 |
| A2-11 | S3 | runtime | `look(k)` 恰好在照片也失效的那些短桩上返回空 |
| A2-12 | S3 | runtime | `check_map(42)` 被接受，尽管 prompt 明确要求文本参数加引号 |
| A2-13 | S3 | runtime | episode 结束前 `optimal_walk_m` 恒为 0.0，`walk_ratio` 中途无意义 |
| A2-14 | S4 | docs | `wait()` 文档写 10 s，实际按相位剩余时间计费（实测 15 s） |
| A2-15 | S4 | runtime | route 在第 5 段截断，强制再花 15 s `navigate()` |
| A2-16 | S4 | harness | 每回合 observation 约 66% 是上一回合的原样重复 |
| A2-17 | S4 | harness | 红绿灯图按候选逐个发 —— 有一回合发了 4 张，全绿，且不影响任何决策 |
| A2-18 | S4 | assets | 渲染破洞：本该是人行道的地方是一块白色空洞多边形 |
| A2-19 | S4 | assets | 城市是空的 —— 856 张街景里没有行人、车流、停放车辆 |
| A2-20 | — | runtime | seed 1 solo：blind policy 撞墙 21 次，106 回合零交付（非死锁，但是好的回归夹具） |

严重度：**S1** 使测量失效 · **S2** 错误信息或错误的调参常数 · **S3** 缺失/矛盾信息，
policy 必须绕开 · **S4** 浪费或文档漂移。

## C. Agent 2 独有发现的详细说明

### A2-04（S2）· 支撑 `TIME_BUDGET_MULTIPLE = 7.0` 的 sweep 表复现不出来

**位置**：`embodiedbench/runtime/city/courier_env.py:370-392`

源码里用一张表（20 seeds，obstacle album 开启）来论证这个常数：

```
#     multiple   solo (blind / sighted)   pair (blind / sighted)
#       4.5           65% / 65%               85% / 90%
#       6.75          85% / 95%               90% / 98%
#       9.0           85% / 95%               90% / 98%
```

论证逻辑是：超过 ~6.75 之后时钟不再是瓶颈，真正的瓶颈变成"能不能找到门"，
而且表里 sighted 明显赢（85→95、90→98）。**7.0 就是这么定下来的。**

**我用当前代码、同样配置、20 seeds 实测：**

| tier | stride | blind | sighted |
|---|---|---|---|
| solo | waypoint | 15/20 (75%) | **14/20 (70%)** |
| solo | block | 13/20 (65%) | 14/20 (70%) |
| pair | waypoint | 28/40 (70%) | **26/40 (65%)** |
| pair | block | 24/40 (60%) | 26/40 (65%) |

sighted 在 waypoint 下**反而更差**，完全不是 85%/95%。而且
`docs/RUNNING.md` 里公布的那张表（blind 5/6、8/12；sighted 5/6、8/12）
我能**精确复现**，它也显示 sighted 一分不赚 —— 也就是说**仓库里两张 sweep 表
互相矛盾，而支撑调参常数的那一张跟当前代码对不上。**

为什么完美视力不赚分：penalty 是真的在收（红灯 75 s、路障 45 s，都实测确认），
但预算太松，penalty 永远咬不动：

| tier | blind 付出的红灯秒数 | 剩余预算 slack | penalty/slack |
|---|---|---|---|
| solo | 1 950 | 4 668 | 0.42 |
| pair | 3 150 | 8 361 | 0.38 |
| triple | 4 650 | 26 217 | 0.18 |
| shift | 8 475 | 125 360 | 0.07 |

**复现**：`docs/review/tools/blind_vs_sighted.py`、`analyze2.py`

**建议**：用当前代码重跑 multiple sweep 再定常数，别信记录里的表。
注意这依赖先修 A2-01 和 A2-02 —— 否则"sighted"参照臂本身就已经能作弊。

### A2-03（S2）· 门牌号的真正根因

**位置**：`embodiedbench/compiler/road_network.py:859-865`

**这不是那个显而易见的 bug。** 每一侧确实是按弧长升序编号的，
**46 侧里 42 侧单调**。问题在于两个 counter 各自独立推进，而两侧建筑
在弧长上的位置不同，于是 `house_numbers_near()` 把两侧混起来报给 courier 之后，
**courier 看到的序列**就不单调了。

| 指标 | 值 |
|---|---|
| 每一侧自身单调 | 42 / 46（91%） |
| **courier 看到的序列**有倒置的街道 | **22 / 23（96%）** |
| N 号与 N+1 号最远相隔 | **12 个路口**（Rue de Vaugirard） |

按真实沿街顺序打印的门牌：

```
Rue Oberkampf      2 | 4,6 | 1,8 | 3,5,10 | 7 | 9,12 | 11,14 | 13,16 | 18 | 15
Rue du Bac         1 | 3,5 | 7 | 9 | 11 | 13 | 2
Rue Mouffetard     1,2,3,5,7,9,11,13 | 4,15,17,19 | 6,21 | 23 | 25 | 8 | 10 | 12,27
Boulevard du Temple 1,2 | 3,4 | 5,6 | 7,8 | 9,10 | 11,12 | 13,14 | 15,16,17,18 | 19,20 | 22
```

`Rue du Bac` 整个双号侧只有一个 2 号，还在 13 号的另一头。
`Boulevard du Temple` 是表现良好的反例。

**实际代价**（episode A，站在 15 号找 17 号）：

```
### what just happened
Down Rue Oberkampf the doors read 18 climbing.
```

我按提示往号码上升的方向走：

```
You are on Rue Oberkampf, outside number 18.
  → You are not at 17 Rue Oberkampf. It is 36 m away
You are on Rue Oberkampf, outside number 7.        ← 再走一个 block
  → You are not at 17 Rue Oberkampf. It is 108 m away
```

一条街上的序列是 15 → 18 → 7，照文档指示走**让我离门远了三倍**。
我只能靠 A2-01 的测距仪救回来 —— 这两条是互相放大的：
本该用的找门通道是错的，不该有的那个是免费的。

**复现**：`docs/review/tools/house_numbers.py`

### A2-07（S3）· 全城没有路牌

**23.7%** 的候选行与同列表中另一行同名，**13.5%** 的候选 compass heading 完全相同。
更深的问题是：**没有任何一张渲染图带路牌。**

episode A turn 10 我必须二选一：

```
2. Quai Ménilmontant — sharp right (west) — next junction 6 m
3. Boulevard du Temple — sharp right (west) — next junction 5 m
```

两张照片是同一个奥斯曼式街角、几乎同一角度，都没有巴黎那种蓝底白字搪瓷路牌
—— 而现实中那块牌子正是快递员确认转弯的依据。

**后果**：街道身份永远只能走文本通道。结合已知的门牌号不可读，
**perception 单独无法定位 courier** —— 这就是 `visual` 档位按现有规格无解的结构性原因，
也是迁移到真实巴黎的硬天花板。

### A2-10（S3）· 撞路障的提示与实际位置矛盾

**位置**：`embodiedbench/runtime/city/courier_env.py:1631`

```
You walk 36 m along Boulevard du Temple, through 2 junctions. Boulevard du
Temple is blocked and you cannot get past. You walk back to the junction.
You will have to go round.
```

我读成"你回到原点了"。**实际没有** —— 我在两个路口之外、贴着路障，
那 36 m 的进展是保留的。它想说的是"退回路障前最后一个路口"，
但字面与紧邻上方的位置栏直接矛盾。（收费 71 s = 26 s 走路 + 45 s 撞击，是对的。）

### A2-14（S4）· `wait()` 文档与实际不符

`docs/RUNNING.md` 工具表写 `wait()` 花 10 s，实测 **15 s**。
runtime 是按信号相位剩余时间收费（`courier_env.py:2232`），这是**更正确**的机制
（固定 10 s 对 60 s 相位是坏的），只是表过时了。

### A2-17（S4）· 红绿灯图按候选逐个发

episode A turn 23，我直行通过：

```
[light 1] pedestrian light for street 1   → …/toward_s005_n005_green.png
[light 2] pedestrian light for street 2   → …/toward_s042_n000_green.png
[light 3] pedestrian light for street 3   → …/toward_s005_n007_green.png
[light 4] pedestrian light for street 4   → …/toward_s009_n009_green.png
```

一回合 4 张多余图片，全绿，不影响任何决策。图片本身是对的（见下方"已验证正确"），
只是经常与决策无关，而图片是 context 里最贵的部分。

### A2-19（S4）· 城市是空的

856 张街景里没有行人、车流、停放车辆。对于目标是"能部署到真实巴黎"的环境，
这是最大的 domain gap；而且它与任务本身冲突：`slow_pedestrian` 这个障碍类别的
图片里**没有行人**（画面是广告柱、垃圾桶、A 字牌和花坛）。

## D. 我实际跑的两个 episode

| | episode A | episode B |
|---|---|---|
| 配置 | solo · block · seed 0 | triple · block · seed 4 |
| 交付 | **1 / 1** | **3 / 3** |
| 准时 | 1 | 1（2 单迟到） |
| 回合数 | 34 | 61 |
| 模拟分钟 | 15.0 | 25.1 |
| 实走/最优 | 767 m / 563 m（1.36×） | 998 m / 621 m（1.61×） |
| 撞路障 | 1 | 0 |
| 闯红灯 | 1 | 3 |
| 被拒动作 | 5 | 10 |
| 收入 | 5.43 | 8.99 |

**episode A** 由一次道路封闭决定：路线要往东北走 Boulevard du Temple，
照片显示护栏加垃圾桶完全封死，地图显示这一整段没有任何岔路。
我还是往前走了，被罚 45 s，然后退回约 120 m 从 Rue de Sévigné 绕行 —— 约 200 m 绕路。
仍然提前两分钟送达，这说明**时钟宽松**，不说明我玩得好。

**episode B** 是排序档位，任务结构设计得确实好：job 0 在 4 Boulevard de Buci 取件，
job 1 在 14 Boulevard de Buci 送达，天然可以合并；第三单还落在我正在走的街上。
三单全部送达，但**我自己**因为一次把两个 `walk_to` 塞进同一轮而没读第一个结果，
连续两次走过了门（一次让 25 Rue Oberkampf 的紧单过期），损失约 8 个回合。
**这是我的 policy 错误，不算环境缺陷** —— 记在这里是因为让我便宜地恢复的正是 A2-01。

## E. 已验证正确 —— 建议不要在这些地方浪费时间

这些是我最用力想弄坏但没弄坏的：

| 属性 | 证据 |
|---|---|
| **Album gate** | 4/4 状态精确：不挂 signal album → 0 次红灯计数；不挂 obstacle album → 0 次碰撞、0 次减速。**不为环境未提供的信息收费** |
| **红绿灯图是实时的** | 发出的灯色与 runtime 收费所依据的相位 **166/166 完全一致**。我最初判断它们是静态装饰，**这个判断是错的** |
| **确定性** | 同 seed 两次 → 完全一致；不同 seed 分叉。另有独立佐证：`play.py` 从 seed 重建 episode B 共 61 次，每次落到同一状态 |
| **时钟一致性** | 每个 tier 都是 7.00× optimal，spread ≤ 0.001。（**数值**是 A2-04 的问题；**一致性**完全符合设计） |
| **配置网格** | 90/90 格（5 difficulty × 2 stride × 3 condition × 3 seeds）全部构造、reset、渲染、可派发，**0 失败**；order 数处处符合 `Difficulty.SPEC` |
| **非法配置** | 未知 difficulty/stride/condition 均在构造期抛错，无静默降级 |
| **畸形回复** | 无代码块 / 未知工具 / 两个调用 / 参数数目错误 → format error，**收费 0 s**；连续三次结束 episode |
| **成本核算** | `check_order` 2 s、`look` 2 s、`check_map` 5 s、`navigate` 15 s、`collect`/`hand_over` 30 s、走路 29.4 m/21 s = 1.40 m/s，全部相符 |
| **公布的参考表** | `RUNNING.md` 的 waypoint 数字精确复现（solo 5/6、pair 8/12） |
| **测试与迁移** | 950 passed / 4 skipped；`migration_check` exit 0，view coverage 100% |

## F. 配置扫描的完整结果

### F.1 Album gate（3 seeds，solo · block）

| signals | obstacles | 闯红灯 | 碰撞 | 减速 | 交付 |
|---|---|---|---|---|---|
| off | off | 0 | 0 | 0 | 3/3 |
| off | on | 0 | 21 | 48 | 2/3 |
| on | off | 15 | 0 | 0 | 3/3 |
| on | on | 13 | 21 | 48 | 2/3 |

完全符合设计。顺带注意它对 A2-04 的旁证：**打开 obstacle album 少交付一单；
打开 signal album 一单都不影响。**

### F.2 非法输入与拒绝

| case | 结果 | 收费 |
|---|---|---|
| 未知 `difficulty`/`stride`/`condition` | 构造期 `ValueError` | — |
| 无代码块 | format_error，返回指引 | 0 s |
| 未知工具 `fly_to(2)` | format_error，列出可用工具 | 0 s |
| 两个代码块 | format_error | 0 s |
| `walk_to()` / `walk_to(1,2,3)` | rejected `bad_arguments` | 0 s |
| `walk_to(999)` | rejected `no_such_street` | 5 s |
| `check_map("nowhere at all")` | rejected `unknown_address` | 5 s |
| `check_map(42)` | **accepted**（见 A2-12） | 5 s |
| block stride 下 `follow_street` | format_error（见 A2-05） | 0 s |
| 连续三次畸形 | episode 结束，`repeated_format_errors` | — |

### F.3 观测经济性（各 40 回合）

| | waypoint | block |
|---|---|---|
| 平均 observation | 161 词 | 156 词 |
| 每回合重发的 system prompt | 1 145 词 | 1 102 词 |
| **与上一回合逐行重复率** | **67%** | **65%** |
| 每回合照片数 | 2.45 | 2.52 |
| 同一回合内重复图片 | 0 | 0 |

episode A 中这一段连续 11 个回合逐字节相同：

```
### streets leaving this junction
  1. Rue Saint-Antoine — behind you (south-east) — next junction 18 m
  2. Rue Saint-Antoine — straight ahead (north-west) — next junction 18 m
```

具体可砍的（都不损失信息）：
1. `### photographs` 的 caption 列表几乎逐项重复 `### streets`（同序号、同街名、同相对方位），二选一即可。
2. `Streets you have seen:` 单调增长，95 个回合里从未起过作用。（`Came from:` 和绕圈警告都有用，保留。）

## G. Agent 2 建议的修复顺序

1. **A2-01（一个字符串）和 A2-02（一次路径映射）必须最先做** —— 在它们关掉之前，
   任何关于 perception 的测量都不作数，**包括用来给其它一切调参的 `sighted` 参照臂本身**。
2. **A2-04** —— 在第 1 步之后重跑 multiple sweep，用结果定常数。这是让 hazard 真正咬人、
   让视觉真正值钱的那个改动。
3. **A2-03** —— 门牌号改用共享弧长游标。这是让最后 50 m 可以靠"看"解决的前提。
4. **A2-05 / A2-06 / A2-08 / A2-09** —— "告诉 courier 的和实际发生的不一致"这一簇。
5. **A2-07 路牌** —— 工作量大些，但这是唯一能打开真正 perception 档位的改动，
   而且是贴图改动，不需要重新烘焙几何。
6. S4 尾巴：A2-10、12、13、14、15、16、17、18。

## H. 复现本部分所有内容

```bash
cd /home/murray/simworld_nav
P=/data/murray/miniconda3/bin/python

$P -m pytest tests/ -q                          # 950 passed, 4 skipped
$P -m embodiedbench.tools.migration_check       # exit 0

$P docs/review/tools/sweep_configs.py           # 网格、gate、非法输入、经济性、测距仪
$P docs/review/tools/analyze.py                 # prompt 承诺、路径泄露、候选边、时钟
$P docs/review/tools/analyze2.py                # perception_pays、灯图、门牌号
$P docs/review/tools/blind_vs_sighted.py        # 20 seeds blind vs sighted
$P docs/review/tools/house_numbers.py           # 门牌号单调性
$P docs/review/tools/make_report.py             # 从 episode 重建 report.html

# 一回合一进程地回放/续玩：
$P docs/review/tools/play.py start --episode C --tier pair --stride block --seed 7
$P docs/review/tools/play.py report --episode C
```

---
---

# 第三部分：修复状态（Agent 2 执行）

只修了**两份报告都确认、且判定为 100% 是 bug** 的条目。涉及设计取舍的没有动，理由见下。
全套测试 **958 passed / 4 skipped**（原 950），`migration_check` **exit 0**。

## 已修复

| 条目 | 改动 | 验证 |
|---|---|---|
| DBG-001 / A2-01 距离泄露 | `courier_env.py` 两处 refusal 改为只说「你不在门口」，不再给米数 | 实测 refusal 文本无数字；旧 5 s 收费保留 |
| DBG-006 / A2-02 文件名泄露 | 新增 `agent/courier/frame_alias.py`，`CourierSession` 发给 policy 的帧改为内容寻址的不透明名 | **39.9% → 0.0%**（539 帧） |
| DBG-004 / A2-03 门牌号 | `road_network.py` 改为**全街共享游标**、按side定奇偶 | N 与 N+1 最远间隔 **12 → 5** 个路口 |
| DBG-016 / A2-05 幻觉工具 | `Step` 支持按工具 gating；格式示例改为从可用工具生成；`CourierSession` 构造期扫描**整段 prompt** | 两个 stride 均 0 泄露 |
| DBG-003 / A2-08/09 stride 语义 | 新增 `_block_preview()`，候选行改为「61 m on, 4 junctions, to the next choice」+ 整段 block 的方位 | **300/300** 报价与实走完全一致 |
| A2-10 路障文案 | 改为「You are at the last junction before it」 | — |
| A2-12 `check_map(42)` | `loop.py` 新增按 tool spec 的参数类型校验 | int/str 互换均报 format error |
| A2-13 `optimal_walk_m` | 未交付时返回 `None` 而非 `0.0` | — |
| A2-14 / DBG-002 / DBG-011 文档 | `wait()` 改为「to the end of the phase」；补 `env.reset()`；`python` → `python3`；`navigate` 截断说明；新增 frame alias 说明 | — |
| DBG-007 / A2-16 冗余 | 删除从未影响决策的 `Streets you have seen`；照片 caption 由每街一行改为一行索引表 | observation **161 → 134 词**（waypoint），逐行重复 **67% → 60%** |

新增回归测试 `tests/test_debug_regressions.py`（8 个），每条锁住的是**性质**而不是措辞。

## 刻意没改（需要你决策，不是 bug）

**A2-04 / DBG-005：`TIME_BUDGET_MULTIPLE = 7.0`。** 源码里论证这个常数的表复现不出来，
这一点我已修（把复现得到的真实数据写进 docstring，并提供
`docs/review/tools/clock_sweep.py` 让它可重跑）。但**常数本身没动**，因为改它是设计取舍。
12 seeds、solo+pair、两个 stride 的实测：

| multiple | blind | sighted | gain | 视觉获胜的格子 |
|---|---|---|---|---|
| 2.0 | 27.3% | 36.9% | +9.6 | 4/4 |
| 2.6 | 39.4% | 50.1% | +10.8 | 4/4 |
| **3.5** | **52.2%** | **63.4%** | **+11.2** | **4/4** |
| 4.5 | 66.0% | 69.1% | +3.2 | 2/4 |
| 6.75 | 69.8% | 72.9% | +3.1 | 2/4 |
| 9.0 | 69.8% | 72.9% | +3.1 | 2/4 |

原 docstring 的「6.75 之后时钟不再是瓶颈」**成立**（6.75 与 9.0 完全相同）；
「此时 sighted 明显赢」**不成立**（只有 +3.1）。视觉收益在 **3.5** 达到峰值。
代价是 `Difficulty` docstring 期望 solo/pair 是"称职者应当通过"的入门档，而 3.5 会把
blind floor 压到 52%。**这个取舍该由你来定。**

**A2-06 短桩照片（10.3% < 5 m）**：需要在 compiler 里 weld 短边或重拍，风险高于收益，未动。
**A2-07 路牌 / A2-19 空城**：需要美术资产，代码改不了。
**A2-17 每候选一张灯图**：复查后认为**不是浪费** —— courier 是看完才决定走哪条，
每盏灯都可能影响决策，删掉是减信息而不是减冗余。

## 参考表已重新生成

`RUNNING.md` 的表原本 `shift` sighted 是 13/60，实测为 12/60，已更正；其余格子精确复现。
现在表下注明用 `docs/review/tools/reference_table.py` 重跑。

---

# 第四部分：人行道 album 烘焙 —— 进展与未解问题

## 结论先行

**相机视角问题已确认且严重**：三个 album（street / signal / obstacle）的相机全部位于
车道中线，实测偏移 **0.00 m**。名字叫 `paris_signals_kerb` 的那个也在中线上，名字是误导的。
步行 courier 因此一直在看一个它永远站不到的视角。

**烘焙管线已搭好但尚未产出可用 album。** 相机位、输出格式、编辑器启动都通了，
但画面本身不可用。下面记录五次尝试各自证明了什么，避免后续重复踩。

## 环境事实

- 素材：`/data/shared/CityCore_Paris`（12 G，只读）→ 已复制到 `/data/murray/CityCore_Paris`
- 引擎：`/data/koe/UnrealEngine-5.8.0-release`，可用 `UnrealEditor-Cmd` 无头启动
- `MIGRATION.md` 警告的 `/tmp/uba_shm_locks`（属 koe，775，我不可写）在这条路径上
  **只产生一条非致命警告**，不阻塞启动
- `paris_min` / `launch_editor.sh` 在这台机器上**不存在**（那是旧机器上的东西）
- 相机位已算好：856 个 approach，按每条街自身宽度往右侧路缘偏 `width/2 + 60cm`
  （6 m 街 → 3.6 m，10 m 街 → 5.6 m）。脚本 `tools/ue/bake_pavement_views.py`

## 五次尝试，各自的结论

| # | 方法 | 结果 | 证明了什么 |
|---|---|---|---|
| 1 | `-ExecutePythonScript` + 异步截图 + tick 回调 | 建了目录，**0 张图** | 该开关含义是"跑完就退出"，编辑器在第一次 tick 前就没了 |
| 2 | `SceneCapture2D`（同步） | 出图了，但是 **EXR 伪装成 .png** | 采样源是 HDR；换 `RTF_RGBA8` + `SCS_FINAL_COLOR_LDR` 后得到正确的 8-bit PNG |
| 3 | SceneCapture + 曝光扫描（手动 8 档、直方图 5 档） | 全部不对 | **不是曝光问题**。参考帧 mean 64.0，最好的候选 55.8 但 MAE 反而更差 |
| 4 | SceneCapture + `always_persist_rendering_state` + Lumen cvar + 96 次收敛 | mean 24.3 → 24.8 | **Lumen 全局光在 scene capture 里不累积**，阴影处全黑。属性设置成功但无效，不是 capture component 能修的 |
| 5 | Python remote execution 驱动活编辑器 | 编辑器**活着并 tick 了 923 帧**，但从不在 multicast 上应答（0.0.0.0 和 loopback 都试过） | 编辑器能长期存活；发现机制不通 |

补充：`-ExecCmds=py <script>` **不含退出语义**，脚本能跑、tick 回调能注册、能写日志、
能自己 `quit_editor()` —— 这条路是对的。但最后一次跑出 `done=0 failed=0` 且没有任何异常
日志，与代码逻辑自相矛盾（三种分支都会改计数器），说明脚本可能被执行了多次导致日志被
`open(...,"w")` 截断。**这是下一步要查的第一件事。**

诊断排除项（都已确认不是原因）：
- 不是 World Partition 串流 —— 探针显示 **3290 个 actor 已加载**（1762 StaticMeshActor）
- 不是着色器没编译完 —— 用暖 DDC 重跑，数值一模一样（24.1 → 24.1）
- 画面顶部烤进去的文字是 SkyAtmosphere 的警告，`DisableAllScreenMessages` 未生效，属表面问题

## 下一步（明确且有界）

1. 把编辑器内脚本的日志改成 `append` 并加运行序号，确认它是否被执行多次 —— 这能解释
   `done=0 failed=0`。
2. 确认无头模式下 `set_level_viewport_camera_info` / `take_high_res_screenshot` 是否
   真的可用（`-RenderOffscreen` 下是否存在 viewport）。若不可用，就必须回到
   SceneCapture 路线并解决 Lumen GI，或改用 MRQ + 生成的 LevelSequence。
3. 曝光结论作废重来 —— 之前是在错误材质上扫的。

## 当前状态：没有把半成品接进环境

步行 embodiment 仍然拿车行道的图，`summary()` 里 `viewpoint_matches_embodiment: false`
会明确标出来。**这个债是可见、可统计的**，不会被误当成行人策略的证据。

## 第四部分补充：找到了原来跑通的方式，但卡在实例 provisioning

看代码 + 看别人正在跑的进程，找到了关键事实。**之前几次尝试全部走错了路**：

### 1. 我一直漏了一个 flag

`-ModelContextProtocolPort=8123` **单独给是没用的**，必须配 `-StartModelContextProtocolServer`。
这就是为什么我之前反复"没有监听"。别人正在跑的三个实例命令行是一模一样的模板：

```
SimWorldEditor <instance>/SimWorld.uproject /Game/Maps/empty \
  -StartModelContextProtocolServer -ModelContextProtocolPort=8091 \
  -AssetRegistry.DisableDirectoryWatcher=1 -SpServicesRole=None -notrace -Unattended \
  -NoUba -NOSPLASH -NOSOUND -Messaging -ResX=1280 -ResY=720 -FPSMAX=15 \
  -graphicsadapter=1 -RenderOffScreen -log \
  -DDC=NoZenLocalFallback -LocalDataCachePath=<ddc>
```

顺带解决了另外两个问题：`-NoUba` 正好绕开 `MIGRATION.md` 警告的 `/tmp/uba_shm_locks`
权限问题；`-graphicsadapter=N` 是选卡的正确方式。参考文档：
`/data/koe/SimWorld_SPEAR/docs/DEPLOY_NEW_MACHINE.md:74`。

端口注意：本机 **8000 已被别人的 uvicorn 占用**（`UeMcp` 的默认端口），
所以必须显式指定；在用的有 8091 / 8110 / 8111。

### 2. 烘焙必须用 SimWorld 项目，不是 CityCore 渲染工程

`UeMcp.TOOL = "SimWorldRuntime.SimWorldStudioToolset.ExecutePythonScript"` —— 这个 toolset
来自 SimWorld 项目模块。CityCore_Paris 那个渲染工程没有它，所以我之前拿 CityCore 工程
起编辑器，永远不可能有这个 MCP 工具。**这是前面四次失败的根因**，
SceneCapture / Lumen / 曝光那一串全是在错误的前提下折腾。

正确结构（照 `/data/vish/simworld/ue_instance` 抄）：

```
Binaries -> /data/koe/SimWorld_SPEAR/Binaries
Plugins  -> /data/koe/SimWorld_SPEAR/Plugins
Source   -> /data/koe/SimWorld_SPEAR/Source
Config/            (实目录)
Content/           (实目录，里面全是指向 content-store release 的 symlink)
SimWorld.uproject  (实文件)
```

我已经在 `/data/murray/ue_instance` 建好了这个结构，并把
`Content/CityCore_Paris -> /data/murray/CityCore_Paris/Content/CityCore_Paris` 挂了进去
（CityCore 不在共享 content store 里，必须这样挂）。

### 3. 当前唯一的卡点

新实例起不来：

```
Build.sh: line 37: Aborted (core dumped) dotnet UnrealBuildTool.dll
LogPluginManager: Error: Plugin 'SpCore' failed to load because module 'SpCore'
                  could not be found.
Exiting abnormally (error code: 1)
```

编辑器认为项目需要重新编译 → UnrealBuildTool 崩了 → 模块找不到。
别人的实例用**完全相同的 symlink** 却能起来，所以差的是某个 provisioning 步骤
（大概率要预置 `Intermediate/Build`，或者按 `DEPLOY_NEW_MACHINE.md` §2 的步骤做一遍）。
`SpCore` 在 `Plugins/spear/Binaries/Linux` 下没有独立 `.so`，是编进 monolithic 编辑器的。

**这一步需要知道实例是怎么建出来的人点一下**，或者直接复制一个已知能用的实例目录结构
（含 `Intermediate/`）再换 Content。

### 4. 一旦编辑器起来，剩下的路是现成的

`fpv_render.render_jobs()` 就是原来烘 street/obstacle album 的循环：
`mcp.python(CAPTURE_SCRIPT)` 发一帧、轮询文件落盘、超时算失败。
`tools/ue/bake_pavement_views.py` 里 856 个相机位（按每条街宽度往右路缘偏
`width/2 + 60cm`）已经算好并写进 `jobs.json`，只要把 `render()` 换成
`UeMcp(port=8123)` + `render_jobs()` 即可，不需要再写新逻辑。

## 第四部分再补充：跑通了 —— 缺的是 `-skipcompile`

`-StartModelContextProtocolServer` 之外，还差一个 **`-skipcompile`**。没有它，编辑器认为
新实例需要重编 → UnrealBuildTool 在这台机器上 core dump → 级联成
`Plugin 'SpCore' failed to load`。加上之后 MCP 立刻起来了。

**能用的启动命令**（`/data/murray/ue_instance`，GPU 由 `CUDA_VISIBLE_DEVICES` 选）：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=6 \
  /data/koe/SimWorld_SPEAR/Binaries/Linux/SimWorldEditor \
  /data/murray/ue_instance/SimWorld.uproject /Game/Maps/empty \
  -StartModelContextProtocolServer -ModelContextProtocolPort=8123 \
  -skipcompile -AssetRegistry.DisableDirectoryWatcher=1 -SpServicesRole=None \
  -notrace -Unattended -NoUba -NOSPLASH -NOSOUND -Messaging \
  -RenderOffScreen -log -DDC=NoZenLocalFallback -LocalDataCachePath=/data/murray/ue_ddc \
  > /data/murray/ue_editor.log 2>&1 < /dev/null &
```

注意：`UeMcp` 默认端口 8000 在本机被别人的 uvicorn 占了，必须显式用 8123。
`ping()` 返回 True、`python()` 返回 None 都是正常的 —— MCP 的响应体是空的，
原始 `render_jobs` 本来就不看返回值，只轮询文件落盘。

**曝光**：相机跳位之后视口的 eye adaptation 需要若干帧才收敛，立刻截图会得到
mean 5.5（参考album是 64.0）。把"移动相机"和"截图"拆成两次 MCP 调用、中间 sleep 3 s，
实测 mean 63.2 / 52.1 / 75.0 —— 与参考一致。这就是 `/data/murray/bake_drive.py` 做的事。

**前四次失败的教训**：SceneCapture2D / Lumen / 手动曝光那一整串全部作废，
根因只是"用错了工程"（CityCore 渲染工程没有 SimWorldStudioToolset）。
先看别人正在跑的进程命令行，比自己推理快得多。

## 第四部分完结：人行道 album 已烘好并接入

**856/856 帧，0 失败。** 质量与车行道 album 对齐：

| album | n | mean | median | min | 全黑帧 | 退化帧 |
|---|---|---|---|---|---|---|
| carriageway (`paris_streets_v2`) | 856 | 92.7 | 90.5 | 27.4 | 0 | 0 |
| pavement (`paris_streets_pavement`) | 856 | 95.5 | 93.9 | 15.7 | 0 | 0 |

初次烘出来有 13 帧是"相机卡在墙里"（集中在 s004 / s005 两条街，实际建筑红线比街道
标称宽度更近）。已按 0.6× 偏移重拍，全部通过，manifest 里逐帧记录了实际用的偏移量和
`repaired` 字段 —— 哪些 approach 没有独立的行人视角是写在数据里的，不是隐瞒的。

接入方式：`CourierEnv(..., pavement_album_root=...)`，由 embodiment 决定发哪个 album。
步行拿人行道、滑板车/开车拿车行道，`viewpoint_matches_embodiment` 在 summary 里如实报告。
新增 3 个集成测试（含"人行道 album 必须覆盖车行道 album 的每一个 approach"）。

**全套 969 tests 通过**，`migration_check` exit 0。参考表已重跑，只有 ceiling 的 shift
从 55/60 变成 54/60（体力机制让完美路由也掉了一单），其余格子不变。
