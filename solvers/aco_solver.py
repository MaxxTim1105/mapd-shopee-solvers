from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
GAMMA = 1.0


class EfficientBFS:
    """Fast lazy BFS with precomputed neighbor indices and list-based distances."""

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        H, W = len(grid), len(grid[0])
        free = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 0]
        self.cell_to_idx: Dict[Position, int] = {p: i for i, p in enumerate(free)}
        self.idx_to_cell: Dict[int, Position] = {i: p for i, p in enumerate(free)}
        self._n = len(free)
        self._nbrs: List[list] = [[] for _ in range(self._n)]
        for i, (r, c) in enumerate(free):
            for dr, dc, m in ((-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                    ni = self.cell_to_idx[(nr, nc)]
                    self._nbrs[i].append((ni, m))
        self._dist_cache: Dict[int, List[int]] = {}
        self._move_cache: Dict[int, List[str]] = {}

    def _bfs_from(self, start: Position) -> None:
        si = self.cell_to_idx.get(start)
        if si is None or si in self._dist_cache:
            return
        n = self._n
        dists = [-1] * n
        fmoves = [None] * n
        dists[si] = 0
        q = [si]
        head = 0
        while head < len(q):
            pi = q[head]
            head += 1
            for ni, move in self._nbrs[pi]:
                if dists[ni] != -1:
                    continue
                dists[ni] = dists[pi] + 1
                fmoves[ni] = move if pi == si else fmoves[pi]
                q.append(ni)
        self._dist_cache[si] = dists
        self._move_cache[si] = fmoves

    def dist(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        si = self.cell_to_idx.get(start)
        gi = self.cell_to_idx.get(goal)
        if si is None or gi is None:
            return INF
        self._bfs_from(start)
        dists = self._dist_cache.get(si)
        if dists is not None:
            d = dists[gi]
            return d if d != -1 else INF
        return INF

    def next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        si = self.cell_to_idx.get(start)
        gi = self.cell_to_idx.get(goal)
        if si is None or gi is None:
            return "S"
        self._bfs_from(start)
        moves = self._move_cache.get(si)
        if moves is not None and moves[gi] is not None:
            return moves[gi]
        return "S"

    def after(self, pos: Position, move: Move) -> Position:
        return valid_next_pos(pos, move, self.grid)


class ACOSolver(Solver):
    """Ant Colony Optimizer for MAPD (N<=100, C<=25, G<=1500, T<=2400)."""

    method_name = "ACOSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs: Optional[EfficientBFS] = None
        self.rng = random.Random(0)
        self.pheromone: Dict[Tuple[Position, Position], float] = defaultdict(lambda: 1.0)
        self.pickup_heat: Dict[Position, float] = defaultdict(float)
        self.targets: Dict[int, Tuple[str, int, Position]] = {}
        self.run_deadline = 0.0
        self.grid: List[List[int]] = [[0]]
        self.N = 1
        self.C = 1
        self.T = 1
        self._step_pool_cache: Dict = {}
        self._step_heuristic_cache: Dict = {}
        self._configure_by_observation(1, 1, 1, [[0]])

    def _configure_by_observation(self, n: int, c: int, t: int, grid: List[List[int]]) -> None:
        self.N = n
        self.C = c
        self.T = t
        self.grid = grid
        self.bfs = EfficientBFS(grid)
        cells = max(1, n * n)
        blocked = sum(1 for row in grid for cell in row if cell != 0)
        obs_density = blocked / cells
        map_scale = min(1.0, max(0.0, (n - 6) / max(1.0, n + 6)))
        agent_pressure = min(1.0, c / max(1.0, n))

        self.d1_penalty = 0.15 + 0.50 * map_scale + 0.20 * obs_density + 0.10 * agent_pressure
        self.d2_penalty = 0.05 + 0.12 * map_scale + 0.08 * obs_density
        self.late_penalty = 0.40 + 0.75 * map_scale + 0.30 * obs_density + 0.15 * agent_pressure

        if t <= 600:
            self.ant_count = min(24, max(12, 3 * c))
            self.iterations = max(2, min(4, 2 + c // 5))
            self.pool_limit = min(48, max(20, 2 * c, n * n // 30))
        elif t <= 1200:
            self.ant_count = min(18, max(10, 2 * c))
            self.iterations = max(2, min(4, 2 + c // 6))
            self.pool_limit = min(40, max(18, c + 8, n * n // 40))
        else:
            self.ant_count = min(10, max(6, c + 2))
            self.iterations = max(1, min(2, 1 + c // 10))
            self.pool_limit = min(25, max(14, c + 4, n * n // 100))

        self.rush_threshold = t // 3

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _carried(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _can_take(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered or len(shipper.bag) >= shipper.K_max:
            return False
        if self._carried_weight(shipper, orders) + order.w > shipper.W_max:
            return False
        d1m = abs(shipper.position[0] - order.sx) + abs(shipper.position[1] - order.sy)
        d2m = abs(order.sx - order.ex) + abs(order.sy - order.ey)
        if d1m + d2m > self.T:
            return False
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        return self.bfs.dist(shipper.position, pickup) < INF and self.bfs.dist(pickup, drop) < INF

    def _can_finish_before_end(self, shipper: Shipper, order: Order, now: int) -> bool:
        d1m = abs(shipper.position[0] - order.sx) + abs(shipper.position[1] - order.sy)
        d2m = abs(order.sx - order.ex) + abs(order.sy - order.ey)
        remaining = self.T - now
        if d1m + d2m > remaining:
            return False
        d1 = self.bfs.dist(shipper.position, (order.sx, order.sy))
        d2 = self.bfs.dist((order.sx, order.sy), (order.ex, order.ey))
        if d1 >= INF or d2 >= INF:
            return False
        margin = max(1, min(10, remaining // 20))
        return d1 + d2 + margin <= remaining

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((o.ex, o.ey) == pos for o in self._carried(shipper, orders))

    def _safe_pickup_while_carrying(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int) -> bool:
        carried = self._carried(shipper, orders)
        if not carried:
            return True
        pickup = (order.sx, order.sy)
        to_pickup = self.bfs.dist(shipper.position, pickup)
        if to_pickup >= INF:
            return False
        remaining = self.T - now
        max_extra = max(6, min(self.N // 2, remaining // 20))
        for o in carried:
            drop_pos = (o.ex, o.ey)
            direct = self.bfs.dist(shipper.position, drop_pos)
            via = to_pickup + self.bfs.dist(pickup, drop_pos)
            if via >= INF:
                return False
            extra = via - direct if direct < INF else via
            if extra > max_extra:
                return False
            if o.et - (now + via) < -5:
                return False
        return True

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        candidates = [o for o in orders.values() if (o.sx, o.sy) == pos and self._can_take(shipper, o, orders)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: (-o.p, o.et, o.id))

    def _update_heat(self, obs: dict) -> None:
        for oid in obs.get("new_order_ids", []):
            order = obs["orders"].get(oid)
            if order is not None:
                self.pickup_heat[(order.sx, order.sy)] += 1.0 + 0.5 * order.p

    def _move_cost_estimate(self, shipper: Shipper, orders: Dict[int, Order], steps: int) -> float:
        w = self._carried_weight(shipper, orders)
        return 0.01 * (1.0 + GAMMA * w / max(shipper.W_max, 1.0)) * steps

    def _heuristic(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int) -> float:
        key = (shipper.id, order.id)
        cached = self._step_heuristic_cache.get(key)
        if cached is not None:
            return cached

        start = shipper.position
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(start, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            self._step_heuristic_cache[key] = 0.0
            return 0.0
        finish = now + d1 + d2
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        slack = order.et - finish

        move_cost = self._move_cost_estimate(shipper, orders, d1 + d2)
        heat = self.pickup_heat.get(pickup, 0.0)
        cluster = sum(1 for o in orders.values() if not o.picked and abs(o.sx - order.sx) + abs(o.sy - order.sy) <= 3)

        remaining = self.T - now
        time_factor = 1.0 + max(0.0, (1.0 - remaining / max(self.T, 1)) * 0.5)

        base = (
            reward
            - move_cost
            + 2.5 * order.p
            + heat
            + 0.5 * cluster
            - self.d1_penalty * d1 * time_factor
            - self.d2_penalty * d2 * time_factor
            - self.late_penalty * late * time_factor
        )

        if slack >= 0:
            base += (0.8 + 0.25 * order.p) / (1.0 + slack / max(remaining // 30, 4))
        else:
            base /= 1.0 + min(10.0, -slack)

        val = max(0.001, base)
        self._step_heuristic_cache[key] = val
        return val

    def _build_candidate_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Order]:
        limit = self.pool_limit
        candidates = []
        remaining = self.T - now
        sx, sy = shipper.position

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            d1m = abs(sx - order.sx) + abs(sy - order.sy)
            d2m = abs(order.sx - order.ex) + abs(order.sy - order.ey)
            if d1m + d2m > remaining:
                continue
            if not self._can_take(shipper, order, orders):
                continue
            if not self._can_finish_before_end(shipper, order, now):
                continue
            candidates.append(order)

        if not candidates:
            return []

        if len(candidates) <= limit:
            scored = [(self._heuristic(shipper, o, orders, now), o.et, o.id, o) for o in candidates]
            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            return [x[3] for x in scored[:limit] if x[0] > 0.001]

        scored = []
        for order in candidates:
            d_man = abs(sx - order.sx) + abs(sy - order.sy) + abs(order.sx - order.ex) + abs(order.sy - order.ey)
            rough = (order.p + 1) / max(1, d_man)
            scored.append((rough, order))
        scored.sort(key=lambda x: -x[0])

        top = scored[:limit * 2]
        refined = [(self._heuristic(shipper, o, orders, now), o.et, o.id, o) for _, o in top]
        refined.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [x[3] for x in refined[:limit] if x[0] > 0.001]

    def _get_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Order]:
        cached = self._step_pool_cache.get(shipper.id)
        if cached is not None:
            return cached
        pool = self._build_candidate_pool(shipper, orders, now)
        self._step_pool_cache[shipper.id] = pool
        return pool

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        carried = [
            order for order in self._carried(shipper, orders)
            if self.bfs.dist(shipper.position, (order.ex, order.ey)) < INF
        ]
        if not carried:
            return None

        remaining = self.T - now
        rush = remaining < self.rush_threshold

        def score(order: Order) -> float:
            drop = (order.ex, order.ey)
            dist = self.bfs.dist(shipper.position, drop)
            if dist >= INF:
                return -INF
            arrival = now + dist
            rew = delivery_reward(order, arrival, self.T)
            if rush:
                if arrival > order.et:
                    return rew - max(0, arrival - order.et) * 5
                return rew + (order.et - arrival) * 2
            return rew - 0.20 * dist + max(0.0, 30.0 - (order.et - arrival)) * (0.15 + 0.06 * order.p) \
                   - max(0, arrival - order.et) * 3

        return max(carried, key=score)

    def _construct_ant(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Tuple[float, Dict[int, Tuple[str, int, Position]], List[Tuple[Position, Position]]]:
        used: set[int] = set()
        targets: Dict[int, Tuple[str, int, Position]] = {}
        edges: List[Tuple[Position, Position]] = []
        total = 0.0
        shuffled = list(shippers)
        self.rng.shuffle(shuffled)

        remaining = self.T - now
        rush = remaining < self.rush_threshold

        for shipper in shuffled:
            if time.time() > self.run_deadline:
                break
            if shipper.bag:
                delivery = self._delivery_target(shipper, orders, now)
                deliver_value = -INF
                if delivery is not None:
                    dist_to_delivery = self.bfs.dist(shipper.position, (delivery.ex, delivery.ey))
                    if dist_to_delivery < INF:
                        deliver_value = delivery_reward(delivery, now + dist_to_delivery, self.T)

                if rush and delivery is not None:
                    carry_deadline = min((o.et for o in self._carried(shipper, orders)), default=self.T)
                    if now + dist_to_delivery > carry_deadline - 10:
                        targets[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))
                        total += deliver_value
                        continue

                best_pickup = None
                best_pickup_value = -INF
                if len(shipper.bag) < shipper.K_max:
                    for order in self._get_pool(shipper, orders, now):
                        if order.id in used:
                            continue
                        if rush and order.et - (now + self.bfs.dist(shipper.position, (order.sx, order.sy))) < -20:
                            continue

                        pickup = (order.sx, order.sy)
                        drop = (order.ex, order.ey)

                        extra_dist = 0
                        can_evaluate = True
                        for oid in shipper.bag:
                            o = orders.get(oid)
                            if o and not o.delivered:
                                d_via = self.bfs.dist(shipper.position, pickup) + self.bfs.dist(pickup, (o.ex, o.ey))
                                if d_via >= INF:
                                    can_evaluate = False
                                    break
                                d_direct = self.bfs.dist(shipper.position, (o.ex, o.ey))
                                if d_direct < INF:
                                    extra_dist = max(extra_dist, d_via - d_direct)
                        if not can_evaluate:
                            continue
                        max_extra = max(8, self.N // 3, remaining // 30)
                        if extra_dist > max_extra:
                            continue

                        pher = self.pheromone.get((pickup, drop), 1.0)
                        h = self._heuristic(shipper, order, orders, now)
                        lateness = max(0, now + self.bfs.dist(shipper.position, pickup) + self.bfs.dist(pickup, drop) - order.et)
                        feasibility = 1.0 / (1.0 + 0.5 * lateness)

                        alpha_p = 1.2 if not rush else 1.0
                        beta_h = 2.0 if not rush else 2.5
                        value = (pher ** alpha_p) * (h ** beta_h) * feasibility - 0.5 * extra_dist

                        if value > best_pickup_value:
                            best_pickup_value = value
                            best_pickup = order

                threshold = 1.15 if rush else 1.2
                if best_pickup is not None and self._safe_pickup_while_carrying(shipper, best_pickup, orders, now) and (delivery is None or best_pickup_value > deliver_value * threshold):
                    order = best_pickup
                    used.add(order.id)
                    pickup = (order.sx, order.sy)
                    drop = (order.ex, order.ey)
                    targets[shipper.id] = ("pickup", order.id, pickup)
                    edges.append((pickup, drop))
                    total += self._heuristic(shipper, order, orders, now)
                elif delivery is not None:
                    targets[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))
                    total += deliver_value
                continue

            empty_candidates = []
            weights = []
            for order in self._get_pool(shipper, orders, now):
                if order.id in used:
                    continue
                pickup = (order.sx, order.sy)
                drop = (order.ex, order.ey)
                pher = self.pheromone.get((pickup, drop), 1.0)
                h = self._heuristic(shipper, order, orders, now)
                lateness = max(0, now + self.bfs.dist(shipper.position, pickup) + self.bfs.dist(pickup, drop) - order.et)
                feasibility = 1.0 / (1.0 + 0.5 * lateness)
                desirability = (pher ** 1.2) * (h ** 2.0) * feasibility
                empty_candidates.append(order)
                weights.append(desirability)

            if not empty_candidates:
                continue
            order = self.rng.choices(empty_candidates, weights=weights, k=1)[0]
            used.add(order.id)
            pickup = (order.sx, order.sy)
            drop = (order.ex, order.ey)
            targets[shipper.id] = ("pickup", order.id, pickup)
            edges.append((pickup, drop))
            total += self._heuristic(shipper, order, orders, now)

        return total, targets, edges

    def _aco_targets(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, Tuple[str, int, Position]]:
        self._step_pool_cache = {}
        self._step_heuristic_cache = {}

        remaining = self.T - now
        rush = remaining < self.rush_threshold

        decay = 0.92 + 0.06 * min(1.0, remaining / max(self.T, 1))
        for edge in list(self.pheromone):
            self.pheromone[edge] = max(0.1, self.pheromone[edge] * decay)

        best_score = -INF
        best_targets: Dict[int, Tuple[str, int, Position]] = {}
        best_edges: List[Tuple[Position, Position]] = []

        ants = self.ant_count if not rush else max(6, self.ant_count // 2)
        iterations = self.iterations if not rush else max(1, self.iterations // 2)

        for _ in range(iterations):
            for _ in range(ants):
                if self.run_deadline and time.time() > self.run_deadline:
                    break
                score, targets, edges = self._construct_ant(shippers, orders, now)
                if score > best_score:
                    best_score = score
                    best_targets = targets
                    best_edges = edges
            if self.run_deadline and time.time() > self.run_deadline:
                break

        reward_scale = max(0.05, best_score / 100.0)
        mul = 1.5 if rush else 1.0
        for edge in best_edges:
            self.pheromone[edge] = min(25.0, self.pheromone[edge] + reward_scale * mul)

        return best_targets

    def _action_to(self, shipper: Shipper, target: Tuple[str, int, Position], orders: Dict[int, Order]) -> Action:
        kind, _, pos = target
        move = self.bfs.next_move(shipper.position, pos)
        nxt = self.bfs.after(shipper.position, move)
        op = 0
        if nxt == pos:
            op = 1 if kind == "pickup" else 2
        if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
            op = 0
        if op == 2 and not self._deliverable(shipper, orders, nxt):
            op = 0
        return move, op

    def _avoid_collisions(
        self,
        obs: dict,
        actions: Dict[int, Action],
        target_positions: Optional[Dict[int, Position]] = None,
    ) -> Dict[int, Action]:
        return resolve_collisions_and_blocks(
            list(obs["shippers"]),
            actions,
            self.grid,
            obs["orders"],
            self._pickup_at,
            self._deliverable,
            target_positions,
            self.bfs.dist,
        )

    def _decide(self, obs: dict) -> Dict[int, Action]:
        self._update_heat(obs)
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])
        now = int(obs["t"])

        targets = self._aco_targets(shippers, orders, now)
        actions: Dict[int, Action] = {}

        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                continue
            here = self._pickup_at(shipper, orders, shipper.position)
            if here is not None and (not shipper.bag or self._safe_pickup_while_carrying(shipper, here, orders, now)):
                actions[shipper.id] = ("S", 1)
                continue
            target = targets.get(shipper.id)
            if target is None:
                actions[shipper.id] = ("S", 0)
            else:
                actions[shipper.id] = self._action_to(shipper, target, orders)

        self.targets = targets
        return self._avoid_collisions(obs, actions, {sid: target[2] for sid, target in targets.items()})

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])
        self.run_deadline = start + max(20.0, min(110.0, 0.12 * self.T + 9.0 * self.C))

        while not obs.get("done", False):
            if time.time() > self.run_deadline:
                actions = {s.id: ("S", 2) for s in obs["shippers"]}
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, time.time() - start)
