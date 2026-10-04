"""联合对齐算法。

问题定义
========

给定：

* ``reference``: R 个参考整数电平 ``R[0..R-1]``（8 <= R <= 24）；
* ``observations``: N 个观测整数 ``O[0..N-1]``（8 <= N <= 60）；
* 统一漂移闭区间 ``[drift_min, drift_max]``，漂移 d 必须为其中的整数，
  区间宽度不超过 :data:`DRIFT_WIDTH_MAX`；
* 残差上限 ``residual_limit``，每个采用电平 L = R[i] + d 必须满足
  ``|O[t] - L| <= residual_limit``；
* 每级停留采样数范围 ``[dwell_min, dwell_max]``（1 <= ... <= 3）；
* 最多 ``max_skips`` 个内部跳过（0..2，默认 2；跳过的是参考电平，
  且只能发生在首尾之间）。

求解：联合选择一个整数漂移 d、一个**必含首尾**的参考子序列
``R[i_0]=R[0], R[i_1], ..., R[i_{k-1}]=R[R-1]``（相邻索引严格递增）
以及连续采样边界 ``0 = s_0 < s_1 < ... < s_k = N``，令第 j 级占据
连续采样区间 ``[s_j, s_{j+1})``，每个观测恰好归属一个采用电平，
停留长度 ``s_{j+1} - s_j in [dwell_min, dwell_max]``，区间内每个观测
相对该电平残差不越限。

目标按顺序最小化：

1. 跳过数 ``(R-1) - (k-1)``（等价于最小化级数 k）；
2. 残差绝对值总和；
3. 最大残差；
4. 漂移 d；
5. 边界序列（``s_1, ..., s_{k-1}``）字典序；
6. 以上全部相同时（仅可能出现在重复参考电平上），被跳过参考索引
   元组的字典序——仅用于在响应完全等价的方案间给出确定性规范结果。

分阶段动态规划
==============

第 3 项目标（最大残差）是**单调但不可加**的聚合量：一个前缀最大残差
更大的方案，仍可能被后续更大的块残差追平，此后应由边界字典序裁决。
因此不能像可加代价那样在每个 DP 状态只保留一个最优前缀，否则会把
字典序更小的全局最优前缀提前剪去。本实现严格按目标优先级分阶段求解：

* **阶段 A（跳过数、残差和）**：两者均可加。逐漂移做 DP，
  ``grid[j][i]`` 为 ``{采用级数 q: 最小残差和}``，得到全局最优
  跳过数 S* 与残差和 T*（跨全部漂移取最优）。
* **阶段 B（最大残差、漂移）**：只在跳过数=S*、残差和=T* 的方案中
  比较。逐漂移维护 ``{q: {前缀残差和: 最小前缀最大残差}}``——前缀和
  不同的方案需要不同的后缀和来凑够 T*，彼此不可支配，故按前缀和
  分桶，每桶只留最小前缀最大。漂移按升序枚举并以当前最优最大残差
  严格剪枝（平局归更小漂移）；进入精确 DP 前先跑一遍无残差和约束
  的最大残差 DP 取得合法下界，下界不优于当前最优的漂移整体跳过。
* **阶段 C（边界字典序）**：固定漂移 D*、S*、T*、M*，逐
  ``{q: {前缀残差和: (前缀最大, 边界编码) 的 Pareto 前沿}}`` 扩展，
  仅保留前缀最大不超过 M* 的方案，最终取最小边界编码；节点保留
  前驱指针用于回溯采用电平与各级停留。边界均在 1..60，以 64 为基
  编码为单个整数，同级数下整数大小次序即边界序列字典序。

阶段 B/C 还用一个**放宽的后缀最小残差和**（记忆化，允许任意不超过
上限的跳过、不计级数）作为下界，前缀和 + 后缀下界 > T* 的状态直接
丢弃；放宽保证它是合法下界，不会误删可行方案。

其它实现要点：

* **块可行区间预计算**：对采样块 [j-L, j) 归属 R[i]，令
  c_t = O[t] - R[i]，可行漂移为整数闭区间
  ``[max c_t - lim, min c_t + lim]``，与漂移无关，只算一次。
* **首末级预筛**：任何完整对齐都必须让首级与末级块可行，取两者
  可行区间交集后再枚举漂移。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class AlignmentError(ValueError):
    """输入不合法（字段越界或序列规模非法）。"""


# 漂移闭区间允许的最大宽度（drift_max - drift_min）。
DRIFT_WIDTH_MAX = 2000

_BOUNDARY_BASE = 64  # 必须 > 最大观测数 60


def solve_alignment(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int = 1,
    dwell_max: int = 3,
    max_skips: int = 2,
) -> dict:
    """求最优联合对齐。

    成功返回含 ``feasible=True`` 的结果字典（漂移、逐级采样区间、
    残差证据、残差和/最大残差、跳过信息）；任何合法对齐都不存在时
    返回 ``{"feasible": False, ...}``；非法输入抛 :class:`AlignmentError`。
    """
    _validate_inputs(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )

    R = len(reference)
    N = len(observations)
    lim = residual_limit

    # 级数上下界（首尾必用、至多 max_skips 个内部跳过）。
    min_levels = R - max_skips
    max_levels = R
    if min_levels * dwell_min > N or max_levels * dwell_max < N:
        return _infeasible_result()

    # 预计算每个 (i, j, L) 块的可行漂移区间与中心化残差 c_t = O[t]-R[i]。
    # blocks[(i, j, L)] = (feas_lo, feas_hi, (c_{j-L}, ..., c_{j-1}))
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]] = {}
    for i in range(R):
        ri = reference[i]
        for j in range(N + 1):
            for L in range(dwell_min, dwell_max + 1):
                s = j - L
                if s < 0:
                    continue
                cs = tuple(observations[t] - ri for t in range(s, j))
                blocks[(i, j, L)] = (max(cs) - lim, min(cs) + lim, cs)

    intervals = _drift_candidate_intervals(
        blocks, R, N, dwell_min, dwell_max, max_skips
    )
    drift_ranges = _enumerate_int_intervals(
        intervals, drift_min, drift_max
    )

    # 每个 (i) 的 j 可行窗口与最邻近前驱索引下界（各阶段共用）。
    windows: List[Tuple[int, int, int]] = []
    later_levels_min: List[int] = []
    later_levels_max: List[int] = []
    for i in range(R):
        if i == R - 1:
            later_levels_min.append(0)
        else:
            later_levels_min.append(max(1, R - 1 - i - max_skips))
        later_levels_max.append(R - 1 - i)
    for i in range(1, R):
        used_before_min = i - min(max_skips, max(0, i - 1))
        j_lo = max(
            (used_before_min + 1) * dwell_min,
            N - later_levels_max[i] * dwell_max,
        )
        j_hi = min(
            N,
            (i + 1) * dwell_max,
            N - later_levels_min[i] * dwell_min,
        )
        p_lo = max(0, i - max_skips - 1)
        windows.append((j_lo, j_hi, p_lo))

    # ---------- 阶段 A：最小化 (跳过数, 残差和)，两目标均可加 ----------
    # finals_a[d] = grid[N][R-1] = {q: 最小残差和}
    finals_a: Dict[int, Dict[int, int]] = {}
    best_skips_sum: Optional[Tuple[int, int]] = None
    for drift_range in drift_ranges:
        for d in drift_range:
            grid = _stage_min_skips_sum(
                blocks, R, N, d, dwell_min, dwell_max, max_skips, windows
            )
            final_cell = grid[N][R - 1]
            if final_cell is not None:
                finals_a[d] = final_cell
                for q, s in final_cell.items():
                    candidate = (R - q, s)
                    if (
                        best_skips_sum is None
                        or candidate < best_skips_sum
                    ):
                        best_skips_sum = candidate

    if best_skips_sum is None:
        return _infeasible_result()

    star_skips, star_sum = best_skips_sum
    star_levels = R - star_skips

    # ---------- 阶段 B：固定 S*/T*，最小化最大残差，再最小化漂移 ----------
    best_max_drift: Optional[Tuple[int, int]] = None
    suffix_for_winner: Optional[_SuffixMinSum] = None
    for drift_range in drift_ranges:
        for d in drift_range:
            final_cell = finals_a.get(d)
            if final_cell is None or final_cell.get(star_levels) != star_sum:
                continue
            incumbent = (
                best_max_drift[0] if best_max_drift is not None else None
            )
            # 无残差和约束的最大残差 DP 给出合法下界；下界都不能严格
            # 击败当前最优时（平局归更小漂移），本漂移无需精确计算。
            free_max = _stage_free_max(
                blocks, R, N, d, dwell_min, dwell_max, max_skips,
                windows, star_levels, incumbent,
            )
            if free_max is None:
                continue
            suffix = _SuffixMinSum(
                blocks, R, N, d, dwell_min, dwell_max, max_skips
            )
            star_max = _stage_exact_max(
                blocks, R, N, d, dwell_min, dwell_max, max_skips, windows,
                star_levels, star_sum, incumbent, suffix,
            )
            if star_max is not None and (
                best_max_drift is None or star_max < best_max_drift[0]
            ):
                best_max_drift = (star_max, d)
                suffix_for_winner = suffix

    if best_max_drift is None or suffix_for_winner is None:
        # 理论不可达：阶段 A 确认可行时阶段 B 必能恢复同一方案。
        return _infeasible_result()

    star_max, star_drift = best_max_drift

    # ---------- 阶段 C：固定全部标量目标，最小化边界序列字典序 ----------
    used_indices, dwells, boundary_code = _stage_min_boundaries(
        blocks, R, N, dwell_min, dwell_max, max_skips, windows,
        star_levels, star_sum, star_max, star_drift, suffix_for_winner,
    )

    cost = (star_skips, star_sum, star_max, star_drift, boundary_code)
    return _build_result(
        reference, observations, cost, used_indices, dwells
    )


def _stage_min_skips_sum(
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]],
    R: int,
    N: int,
    d: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    windows: Sequence[Tuple[int, int, int]],
) -> List[List[Optional[Dict[int, int]]]]:
    """阶段 A：grid[j][i] = {采用级数 q: 最小残差绝对和}。"""
    grid: List[List[Optional[Dict[int, int]]]] = [
        [None] * R for _ in range(N + 1)
    ]

    # 首级 i = 0。
    for j in range(dwell_min, min(dwell_max, N) + 1):
        lo, hi, cs = blocks[(0, j, j)]
        if lo <= d <= hi:
            grid[j][0] = {1: _abs_sum(cs, d)}

    for i, (j_lo, j_hi, p_lo) in enumerate(windows, start=1):
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            cell: Optional[Dict[int, int]] = None
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                lo, hi, cs = blocks[(i, j, L)]
                if not (lo <= d <= hi):
                    continue
                block_sum = _abs_sum(cs, d)
                for p in range(i - 1, p_lo - 1, -1):
                    prev_cell = grid[prev_j][p]
                    if prev_cell is None:
                        continue
                    for pq, prev_sum in prev_cell.items():
                        q = pq + 1
                        if i + 1 - q > max_skips:
                            continue
                        new_sum = prev_sum + block_sum
                        if cell is None:
                            cell = {}
                        old = cell.get(q)
                        if old is None or new_sum < old:
                            cell[q] = new_sum
            if cell is not None:
                grid[j][i] = cell
    return grid


def _stage_free_max(
    blocks,
    R: int,
    N: int,
    d: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    windows,
    star_levels: int,
    incumbent: Optional[int],
) -> Optional[int]:
    """无残差和约束下、级数恰为 star_levels 时的最小最大绝对残差。

    结果是“残差和=T* 约束下最大残差”的合法下界。``incumbent`` 非空时
    严格剪去前缀最大残差已 >= incumbent 的状态：最大残差单调不减，
    这些方案不可能严格优于当前最优（平局归更小漂移）。
    """
    grid: List[List[Optional[Dict[int, int]]]] = [
        [None] * R for _ in range(N + 1)
    ]

    for j in range(dwell_min, min(dwell_max, N) + 1):
        lo, hi, cs = blocks[(0, j, j)]
        if lo <= d <= hi:
            bmax = _abs_max(cs, d)
            if incumbent is None or bmax < incumbent:
                grid[j][0] = {1: bmax}

    for i, (j_lo, j_hi, p_lo) in enumerate(windows, start=1):
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            cell: Optional[Dict[int, int]] = None
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                lo, hi, cs = blocks[(i, j, L)]
                if not (lo <= d <= hi):
                    continue
                bmax = _abs_max(cs, d)
                if incumbent is not None and bmax >= incumbent:
                    continue
                for p in range(i - 1, p_lo - 1, -1):
                    prev_cell = grid[prev_j][p]
                    if prev_cell is None:
                        continue
                    for pq, prev_max in prev_cell.items():
                        q = pq + 1
                        if i + 1 - q > max_skips:
                            continue
                        new_max = (
                            prev_max if prev_max >= bmax else bmax
                        )
                        if incumbent is not None and new_max >= incumbent:
                            continue
                        if cell is None:
                            cell = {}
                        old = cell.get(q)
                        if old is None or new_max < old:
                            cell[q] = new_max
            if cell is not None:
                grid[j][i] = cell

    final_cell = grid[N][R - 1]
    if final_cell is None:
        return None
    return final_cell.get(star_levels)


def _stage_exact_max(
    blocks,
    R: int,
    N: int,
    d: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    windows,
    star_levels: int,
    star_sum: int,
    incumbent: Optional[int],
    suffix: "_SuffixMinSum",
) -> Optional[int]:
    """阶段 B：在级数=star_levels、残差和=star_sum 下求最小最大残差。

    grid[j][i] = {q: {前缀残差和: 最小前缀最大残差}}。前缀和不同的
    方案需要不同后缀和，互不支配，故按前缀和分桶。
    """
    grid: List[List[Optional[Dict[int, Dict[int, int]]]]] = [
        [None] * R for _ in range(N + 1)
    ]

    for j in range(dwell_min, min(dwell_max, N) + 1):
        lo, hi, cs = blocks[(0, j, j)]
        if lo <= d <= hi:
            bsum = _abs_sum(cs, d)
            bmax = _abs_max(cs, d)
            if bsum <= star_sum and (
                incumbent is None or bmax < incumbent
            ):
                grid[j][0] = {1: {bsum: bmax}}

    for i, (j_lo, j_hi, p_lo) in enumerate(windows, start=1):
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            merged: Optional[Dict[int, Dict[int, int]]] = None
            suffix_lb = suffix.get(j, i)
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                lo, hi, cs = blocks[(i, j, L)]
                if not (lo <= d <= hi):
                    continue
                block_sum = _abs_sum(cs, d)
                block_max = _abs_max(cs, d)
                if incumbent is not None and block_max >= incumbent:
                    continue
                for p in range(i - 1, p_lo - 1, -1):
                    prev_cell = grid[prev_j][p]
                    if prev_cell is None:
                        continue
                    for pq, sums in prev_cell.items():
                        q = pq + 1
                        if i + 1 - q > max_skips:
                            continue
                        for prefix_sum, prefix_max in sums.items():
                            new_sum = prefix_sum + block_sum
                            if new_sum > star_sum:
                                continue
                            new_max = (
                                prefix_max
                                if prefix_max >= block_max
                                else block_max
                            )
                            if (
                                incumbent is not None
                                and new_max >= incumbent
                            ):
                                continue
                            if merged is None:
                                merged = {}
                            by_sum = merged.setdefault(q, {})
                            old = by_sum.get(new_sum)
                            if old is None or new_max < old:
                                by_sum[new_sum] = new_max
            if merged is not None and suffix_lb is not None:
                pruned: Dict[int, Dict[int, int]] = {}
                for q, by_sum in merged.items():
                    kept = {
                        s: m
                        for s, m in by_sum.items()
                        if s + suffix_lb <= star_sum
                    }
                    if kept:
                        pruned[q] = kept
                if pruned:
                    grid[j][i] = pruned
    final_cell = grid[N][R - 1]
    if final_cell is None or star_levels not in final_cell:
        return None
    return final_cell[star_levels].get(star_sum)


class _PathNode:
    """阶段 C 前缀节点：保留 Pareto 状态与回溯指针。"""

    __slots__ = (
        "prefix_max",
        "code",
        "skipped",
        "prev",
        "level",
        "dwell",
    )

    def __init__(
        self,
        prefix_max: int,
        code: int,
        skipped: Tuple[int, ...],
        prev: Optional["_PathNode"],
        level: int,
        dwell: int,
    ) -> None:
        self.prefix_max = prefix_max
        self.code = code
        self.skipped = skipped
        self.prev = prev
        self.level = level
        self.dwell = dwell


def _stage_min_boundaries(
    blocks,
    R: int,
    N: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    windows,
    star_levels: int,
    star_sum: int,
    star_max: int,
    d: int,
    suffix: "_SuffixMinSum",
) -> Tuple[List[int], List[int], int]:
    """阶段 C：固定 S*/T*/M*/D*，求字典序最小边界并回溯。

    grid[j][i] = {q: {前缀残差和: [_PathNode  Pareto 前沿]}}；
    节点按 (前缀最大, 边界编码, 跳过索引元组) 三元组互相支配。
    """
    grid: List[List[Optional[Dict[int, Dict[int, List[_PathNode]]]]]] = [
        [None] * R for _ in range(N + 1)
    ]

    def insert(by_sum: Dict[int, List[_PathNode]], s: int, node: _PathNode) -> None:
        front = by_sum.get(s)
        if front is None:
            by_sum[s] = [node]
            return
        for other in front:
            if (
                other.prefix_max <= node.prefix_max
                and other.code <= node.code
                and other.skipped <= node.skipped
            ):
                return
        front[:] = [
            other
            for other in front
            if not (
                node.prefix_max <= other.prefix_max
                and node.code <= other.code
                and node.skipped <= other.skipped
            )
        ]
        front.append(node)

    for j in range(dwell_min, min(dwell_max, N) + 1):
        lo, hi, cs = blocks[(0, j, j)]
        if lo <= d <= hi:
            bsum = _abs_sum(cs, d)
            bmax = _abs_max(cs, d)
            if bsum <= star_sum and bmax <= star_max:
                node = _PathNode(bmax, j, (), None, 0, j)
                grid[j][0] = {1: {bsum: [node]}}

    for i, (j_lo, j_hi, p_lo) in enumerate(windows, start=1):
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            merged: Optional[Dict[int, Dict[int, List[_PathNode]]]] = None
            suffix_lb = suffix.get(j, i)
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                lo, hi, cs = blocks[(i, j, L)]
                if not (lo <= d <= hi):
                    continue
                block_sum = _abs_sum(cs, d)
                block_max = _abs_max(cs, d)
                if block_max > star_max:
                    continue
                for p in range(i - 1, p_lo - 1, -1):
                    prev_cell = grid[prev_j][p]
                    if prev_cell is None:
                        continue
                    gap_skips = tuple(range(p + 1, i))
                    for pq, sums in prev_cell.items():
                        q = pq + 1
                        if i + 1 - q > max_skips:
                            continue
                        for prefix_sum, front in sums.items():
                            new_sum = prefix_sum + block_sum
                            if new_sum > star_sum:
                                continue
                            for prev_node in front:
                                new_max = (
                                    prev_node.prefix_max
                                    if prev_node.prefix_max >= block_max
                                    else block_max
                                )
                                if new_max > star_max:
                                    continue
                                if merged is None:
                                    merged = {}
                                node = _PathNode(
                                    new_max,
                                    prev_node.code * _BOUNDARY_BASE + j,
                                    prev_node.skipped + gap_skips,
                                    prev_node,
                                    i,
                                    L,
                                )
                                insert(merged.setdefault(q, {}), new_sum, node)
            if merged is not None and suffix_lb is not None:
                pruned: Dict[int, Dict[int, List[_PathNode]]] = {}
                for q, by_sum in merged.items():
                    kept = {
                        s: front
                        for s, front in by_sum.items()
                        if s + suffix_lb <= star_sum
                    }
                    if kept:
                        pruned[q] = kept
                if pruned:
                    grid[j][i] = pruned

    final_cell = grid[N][R - 1]
    assert final_cell is not None and star_levels in final_cell
    front = final_cell[star_levels][star_sum]
    winner = min(
        front,
        key=lambda node: (node.prefix_max, node.code, node.skipped),
    )

    used_indices: List[int] = []
    dwells: List[int] = []
    node: Optional[_PathNode] = winner
    while node is not None:
        used_indices.append(node.level)
        dwells.append(node.dwell)
        node = node.prev
    used_indices.reverse()
    dwells.reverse()
    return used_indices, dwells, winner.code


class _SuffixMinSum:
    """放宽的后缀最小残差和（记忆化），仅作前缀剪枝下界。

    值 ``get(j, i)`` 为：观测前缀已结束于 (j, i) 时，铺完 O[j..N) 并以
    R[R-1] 收尾所需的最小残差和；允许任意不超过 max_skips 的相邻跳过、
    不计采用级数。该放宽使结果始终是真实最优后缀和的下界，故
    ``前缀和 + 下界 > T*`` 的剪枝不会误删任何可行方案。
    """

    def __init__(
        self,
        blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]],
        R: int,
        N: int,
        d: int,
        dwell_min: int,
        dwell_max: int,
        max_skips: int,
    ) -> None:
        self._blocks = blocks
        self._R = R
        self._N = N
        self._d = d
        self._dwell_min = dwell_min
        self._dwell_max = dwell_max
        self._max_skips = max_skips
        self._memo: Dict[Tuple[int, int], Optional[int]] = {}

    def get(self, j: int, i: int) -> Optional[int]:
        key = (j, i)
        cached = self._memo.get(key)
        if key in self._memo:
            return cached
        R = self._R
        N = self._N
        if i == R - 1:
            value = 0 if j == N else None
            self._memo[key] = value
            return value

        best: Optional[int] = None
        k_hi = min(R - 1, i + 1 + self._max_skips)
        for k in range(i + 1, k_hi + 1):
            for L in range(self._dwell_min, min(self._dwell_max, N - j) + 1):
                end = j + L
                later = self.get(end, k)
                if later is None:
                    continue
                lo, hi, cs = self._blocks[(k, end, L)]
                if not (lo <= self._d <= hi):
                    continue
                value = _abs_sum(cs, self._d) + later
                if best is None or value < best:
                    best = value
        self._memo[key] = best
        return best


def _abs_sum(cs: Sequence[int], drift: int) -> int:
    """块内残差绝对和（块长 <= 3）。"""
    total = 0
    for c in cs:
        r = c - drift
        total += r if r >= 0 else -r
    return total


def _abs_max(cs: Sequence[int], drift: int) -> int:
    """块内最大绝对残差（块长 <= 3）。"""
    worst = 0
    for c in cs:
        r = c - drift
        if r < 0:
            r = -r
        if r > worst:
            worst = r
    return worst


def _drift_candidate_intervals(
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]],
    R: int,
    N: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> List[Tuple[int, int]]:
    """首、末级块可行漂移区间的交集（与请求漂移闭区间再取交）。"""

    def first_block_intervals() -> List[Tuple[int, int]]:
        later_min = max(1, R - 1 - max_skips)
        out = []
        for j in range(dwell_min, dwell_max + 1):
            if N - j < later_min * dwell_min:
                continue
            lo, hi, _ = blocks[(0, j, j)]
            out.append((lo, hi))
        return _merge_intervals(out)

    def last_block_intervals() -> List[Tuple[int, int]]:
        earlier_min = max(1, R - 1 - max_skips)
        out = []
        for L in range(dwell_min, dwell_max + 1):
            start = N - L
            if start < earlier_min * dwell_min:
                continue
            lo, hi, _ = blocks[(R - 1, N, L)]
            out.append((lo, hi))
        return _merge_intervals(out)

    return _intersect_interval_lists(
        first_block_intervals(), last_block_intervals()
    )


def _merge_intervals(
    intervals: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """合并相交的整数闭区间。"""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi + 1:
            if hi > mhi:
                merged[-1] = (mlo, hi)
        else:
            merged.append((lo, hi))
    return merged


def _intersect_interval_lists(
    a: List[Tuple[int, int]], b: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """两个已合并区间列表的交集。"""
    out: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _enumerate_int_intervals(
    intervals: List[Tuple[int, int]], lo: int, hi: int
) -> List[range]:
    """区间列表与 [lo, hi] 取交后返回各段整数 range。"""
    result: List[range] = []
    for ilo, ihi in intervals:
        clo = max(ilo, lo)
        chi = min(ihi, hi)
        if clo <= chi:
            result.append(range(clo, chi + 1))
    return result


def _build_result(
    reference: Sequence[int],
    observations: Sequence[int],
    cost: Tuple[int, int, int, int, int],
    used_indices: List[int],
    dwells: List[int],
) -> dict:
    skipped, residual_sum, max_abs, drift, _code = cost
    boundaries: List[int] = []
    acc = 0
    for L in dwells:
        acc += L
        boundaries.append(acc)
    assert acc == len(observations)

    levels_out = []
    residual_sum_check = 0
    max_abs_check = 0
    sample_t = 0
    for slot, (ri, L) in enumerate(zip(used_indices, dwells)):
        adopted = reference[ri] + drift
        s = sample_t
        e = sample_t + L
        samples = []
        for t in range(s, e):
            r = observations[t] - adopted
            ar = abs(r)
            residual_sum_check += ar
            if ar > max_abs_check:
                max_abs_check = ar
            samples.append(
                {"index": t, "observed": observations[t], "residual": r}
            )
        levels_out.append(
            {
                "level_order": slot,
                "reference_index": ri,
                "reference_level": reference[ri],
                "adopted_level": adopted,
                "sample_start": s,
                "sample_end": e,
                "dwell": L,
                "samples": samples,
            }
        )
        sample_t = e

    used_set = set(used_indices)
    skipped_indices = [i for i in range(len(reference)) if i not in used_set]

    # 防御性自检：DP 代价必须与重建结果完全一致。
    assert residual_sum_check == residual_sum
    assert max_abs_check == max_abs
    assert skipped == len(skipped_indices)
    assert used_indices[0] == 0
    assert used_indices[-1] == len(reference) - 1

    return {
        "feasible": True,
        "drift": drift,
        "num_skips": skipped,
        "skipped_reference_indices": skipped_indices,
        "residual_sum": residual_sum,
        "max_abs_residual": max_abs,
        "boundaries": boundaries[:-1],
        "num_levels_used": len(used_indices),
        "levels": levels_out,
    }


def _infeasible_result() -> dict:
    return {
        "feasible": False,
        "reason": "no_alignment_exists",
        "message": (
            "在给定漂移区间、残差上限、停留范围与跳过上限下，"
            "不存在任何合法对齐。"
        ),
    }


def _validate_inputs(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> None:
    def _is_int_list(v: object, name: str) -> None:
        if not isinstance(v, list) or not v:
            raise AlignmentError(f"{name} 必须是非空整数数组")
        for x in v:
            if isinstance(x, bool) or not isinstance(x, int):
                raise AlignmentError(f"{name} 的元素必须为整数")

    def _is_int(v: object, name: str) -> None:
        if isinstance(v, bool) or not isinstance(v, int):
            raise AlignmentError(f"{name} 必须为整数")

    _is_int_list(reference, "reference_levels")
    _is_int_list(observations, "observations")
    _is_int(drift_min, "drift_min")
    _is_int(drift_max, "drift_max")
    _is_int(residual_limit, "residual_limit")
    _is_int(dwell_min, "dwell_min")
    _is_int(dwell_max, "dwell_max")
    _is_int(max_skips, "max_skips")

    R = len(reference)
    N = len(observations)

    if not 8 <= R <= 24:
        raise AlignmentError("参考电平数量必须在 8 至 24 之间")
    if not 8 <= N <= 60:
        raise AlignmentError("观测值数量必须在 8 至 60 之间")
    if drift_min > drift_max:
        raise AlignmentError("漂移闭区间要求 drift_min <= drift_max")
    if abs(drift_min) > 1_000_000 or abs(drift_max) > 1_000_000:
        raise AlignmentError("漂移区间超出允许范围 (+/-1000000)")
    if drift_max - drift_min > DRIFT_WIDTH_MAX:
        raise AlignmentError(
            f"漂移闭区间宽度不得超过 {DRIFT_WIDTH_MAX}"
        )
    if residual_limit < 0 or residual_limit > 1_000_000:
        raise AlignmentError("残差上限必须为非负整数且不超过 1000000")
    if not 1 <= dwell_min <= dwell_max <= 3:
        raise AlignmentError("停留范围要求 1 <= dwell_min <= dwell_max <= 3")
    if not 0 <= max_skips <= 2:
        raise AlignmentError("内部跳过数上限必须在 0 至 2 之间")
    for x in list(reference) + list(observations):
        if abs(x) > 1_000_000_000:
            raise AlignmentError("电平/观测值超出允许范围 (+/-1e9)")
