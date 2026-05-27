from __future__ import annotations

import time
from collections import OrderedDict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class GridPaths:
    """Shortest-path distance fields cached by current task destination."""

    def __init__(self, grid: List[List[int]], cache_limit: int):
        self.grid = grid
        self.cache_limit = cache_limit
        self.neighbor_map: Dict[Position, List[Tuple[Move, Position]]] = {}
        self.fields: "OrderedDict[Position, Dict[Position, int]]" = OrderedDict()
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value != 0:
                    continue
                pos = (r, c)
                neighbors: List[Tuple[Move, Position]] = []
                for move in MOVES:
                    nxt = valid_next_pos(pos, move, grid)
                    if nxt != pos:
                        neighbors.append((move, nxt))
                self.neighbor_map[pos] = neighbors

    def neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        if include_wait:
            yield "S", pos
        yield from self.neighbor_map.get(pos, [])

    def _field(self, goal: Position) -> Dict[Position, int]:
        cached = self.fields.get(goal)
        if cached is not None:
            self.fields.move_to_end(goal)
            return cached
        dist: Dict[Position, int] = {}
        if goal in self.neighbor_map:
            dist[goal] = 0
            q = deque([goal])
            while q:
                pos = q.popleft()
                nd = dist[pos] + 1
                for _, nxt in self.neighbor_map[pos]:
                    if nxt not in dist:
                        dist[nxt] = nd
                        q.append(nxt)
        self.fields[goal] = dist
        self.fields.move_to_end(goal)
        while len(self.fields) > self.cache_limit:
            self.fields.popitem(last=False)
        return dist

    def distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        return self._field(goal).get(start, INF)

    def move_between(self, start: Position, nxt: Position) -> Move:
        if start == nxt:
            return "S"
        for move, pos in self.neighbor_map.get(start, []):
            if pos == nxt:
                return move
        return "S"


class MAPDCBSSolver(Solver):
    """Online MAPD task assignment with cached BFS routes and CBS-lite reservations."""

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.grid: List[List[int]] = [[0]]
        self.router: Optional[GridPaths] = None
        self.N = 1
        self.C = 1
        self.T = 1
        self.horizon = 5
        self.pickup_tasks: Dict[int, int] = {}
        self.replanned_tasks: Dict[int, int] = {}
        self.delivery_commitments: Dict[int, int] = {}
        self.failed_assignment_retry: Dict[int, int] = {}
        self.seen_orders: Set[int] = set()
        self.pickup_heat: Dict[Position, int] = {}
        self.pickup_last_seen: Dict[Position, int] = {}
        self.run_deadline = 0.0
        self.repairs = 0
        self.repair_search_budget = 0

    def _configure(self, obs: dict) -> None:
        self.N = int(obs["N"])
        self.C = int(obs["C"])
        self.T = int(obs["T"])
        self.grid = obs["grid"]
        cache_limit = max(32, min(88, 24 + 2 * self.C))
        if self.C >= 15:
            cache_limit = max(48, min(180, 24 + 5 * self.C))
        self.router = GridPaths(self.grid, cache_limit)
        self.horizon = max(4, min(10, 4 + self.C // 4))
        self.pickup_tasks.clear()
        self.replanned_tasks.clear()
        self.delivery_commitments.clear()
        self.failed_assignment_retry.clear()
        self.seen_orders.clear()
        self.pickup_heat.clear()
        self.pickup_last_seen.clear()

    def _mdist(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _endgame_window(self) -> int:
        extra = 2 * max(0, self.N - 30) if self.C >= 15 else 0
        return max(20, min(320, self.T // 10 + extra))

    def _carried(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(order.w for order in self._carried(shipper, orders))

    def _can_take_light(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        return (
            not order.picked
            and not order.delivered
            and len(shipper.bag) < shipper.K_max
            and self._carried_weight(shipper, orders) + order.w <= shipper.W_max
        )

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        candidates = [
            order
            for order in orders.values()
            if (order.sx, order.sy) == pos and self._can_take_light(shipper, order, orders)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda order: (-order.p, order.et, order.id))

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((order.ex, order.ey) == pos for order in self._carried(shipper, orders))

    def _visible_density(self, orders: Dict[int, Order]) -> Tuple[Dict[Position, int], Dict[Position, int]]:
        pickups: Dict[Position, int] = {}
        drops: Dict[Position, int] = {}
        for order in orders.values():
            if order.delivered:
                continue
            if not order.picked:
                pickup = (order.sx, order.sy)
                pickups[pickup] = pickups.get(pickup, 0) + 1
            drop = (order.ex, order.ey)
            drops[drop] = drops.get(drop, 0) + 1
        return pickups, drops

    def _cluster_value(self, cell: Position, counts: Dict[Position, int]) -> float:
        value = float(counts.get(cell, 0))
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            value += (0.45 if abs(dr) + abs(dc) == 1 else 0.20) * counts.get((r + dr, c + dc), 0)
        return value

    def _observe_pickups(self, orders: Dict[int, Order], now: int) -> None:
        for order in orders.values():
            if order.id in self.seen_orders:
                continue
            self.seen_orders.add(order.id)
            pickup = (order.sx, order.sy)
            self.pickup_heat[pickup] = self.pickup_heat.get(pickup, 0) + 1
            self.pickup_last_seen[pickup] = now

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        carried = self._carried(shipper, orders)
        if not carried:
            self.delivery_commitments.pop(shipper.id, None)
            return None
        committed_oid = self.delivery_commitments.get(shipper.id)
        if committed_oid is not None:
            committed = next((order for order in carried if order.id == committed_oid), None)
            if committed is not None:
                return committed
            self.delivery_commitments.pop(shipper.id, None)
        best: Optional[Order] = None
        best_key: Optional[Tuple[float, int, int]] = None
        for order in carried:
            goal = (order.ex, order.ey)
            dist = self.router.distance(shipper.position, goal)
            if dist >= INF:
                continue
            finish = now + dist
            slack = order.et - finish
            same_drop = sum(1 for other in carried if (other.ex, other.ey) == goal)
            score = (
                delivery_reward(order, finish, self.T)
                + 12.0 * order.p
                + 5.0 * same_drop
                + max(0.0, 30.0 - slack) * (0.28 + 0.09 * order.p)
                - 0.10 * dist
            )
            key = (score, -order.et, -order.id)
            if best_key is None or key > best_key:
                best_key = key
                best = order
        return best

    def _passing_pickup_is_worthwhile(
        self,
        shipper: Shipper,
        current_delivery: Order,
        passing: Order,
        orders: Dict[int, Order],
        now: int,
    ) -> bool:
        primary_drop = (current_delivery.ex, current_delivery.ey)
        new_drop = (passing.ex, passing.ey)
        primary_dist = self.router.distance(shipper.position, primary_drop)
        onward = self.router.distance(primary_drop, new_drop)
        if primary_dist >= INF or onward >= INF:
            return False
        finish = now + primary_dist + onward
        if finish >= self.T:
            return False
        reward = delivery_reward(passing, finish, self.T)
        late = max(0, finish - passing.et)
        score = reward + 28.0 * reward / (onward + 8.0) - 0.12 * onward - 0.28 * late
        return new_drop == primary_drop or score >= 10.0

    def _rough_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
        drop_counts: Dict[Position, int],
    ) -> float:
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        travel = self._mdist(shipper.position, pickup) + self._mdist(pickup, drop) + 1
        if now + travel >= self.T:
            return -INF
        finish = now + travel
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        density = self._cluster_value(pickup, pickup_counts) + 0.35 * self._cluster_value(drop, drop_counts)
        return (
            reward
            + 80.0 * reward / (travel + 8.0)
            + 4.0 * order.p
            + 1.5 * density
            - 0.15 * travel
            - 0.80 * late
        )

    def _exact_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
        drop_counts: Dict[Position, int],
    ) -> float:
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.router.distance(shipper.position, pickup)
        d2 = self.router.distance(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        travel = d1 + d2 + 1
        finish = now + travel
        if finish >= self.T:
            return -INF
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        slack = order.et - finish
        density = self._cluster_value(pickup, pickup_counts) + 0.35 * self._cluster_value(drop, drop_counts)
        persistence = 6.0 if (
            self.pickup_tasks.get(shipper.id) == order.id
            or self.replanned_tasks.get(shipper.id) == order.id
        ) else 0.0
        return (
            reward
            + 95.0 * reward / (travel + 8.0)
            + 5.0 * order.p
            + 1.7 * density
            + min(8.0, max(0.0, slack) * 0.08)
            + persistence
            - 0.17 * travel
            - 0.90 * late
        )

    def _backlog_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
    ) -> float:
        """Value for otherwise-idle agents; late delivery is still better than idling."""
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.router.distance(shipper.position, pickup)
        d2 = self.router.distance(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        travel = d1 + d2 + 1
        finish = now + travel
        if finish >= self.T:
            return -INF
        reward = delivery_reward(order, finish, self.T)
        density = self._cluster_value(pickup, pickup_counts)
        return reward + 16.0 * reward / (travel + 8.0) + 0.4 * density - 0.14 * travel

    def _valid_task(self, shipper: Shipper, orders: Dict[int, Order], used_orders: Set[int]) -> Optional[Order]:
        oid = self.pickup_tasks.get(shipper.id)
        order = orders.get(oid) if oid is not None else None
        if (
            order is None
            or oid in used_orders
            or order.picked
            or order.delivered
            or not self._can_take_light(shipper, order, orders)
        ):
            self.pickup_tasks.pop(shipper.id, None)
            return None
        return order

    def _assign_pickups(self, idle: List[Shipper], orders: Dict[int, Order], now: int, step_deadline: float) -> None:
        if not idle:
            return
        pickup_counts, drop_counts = self._visible_density(orders)
        self.replanned_tasks.clear()
        used_orders: Set[int] = set()
        used_cells: Dict[Position, int] = {}
        pending: List[Shipper] = []
        reconsider_dist = max(8, min(14, self.N // 7 + 4))
        reconsider_period = max(4, min(10, self.N // 12 + 4))
        if self.N >= 30 and reconsider_period <= 8:
            reconsider_period = max(3, reconsider_period - 1)
        for shipper in sorted(idle, key=lambda s: s.id):
            order = self._valid_task(shipper, orders, used_orders)
            if order is None:
                pending.append(shipper)
                continue
            if (
                now % reconsider_period == 0
                and self.router.distance(shipper.position, (order.sx, order.sy)) > reconsider_dist
            ):
                self.replanned_tasks[shipper.id] = order.id
                self.pickup_tasks.pop(shipper.id, None)
                pending.append(shipper)
                continue
            pickup = (order.sx, order.sy)
            used_orders.add(order.id)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1

        candidates: List[Tuple[float, int, int, Order]] = []
        exact_cap = max(6, min(14, 6 + len(pending) // 3))
        for shipper in pending:
            if time.perf_counter() > step_deadline:
                break
            rough: List[Tuple[float, int, int, Order]] = []
            for order in orders.values():
                if order.id in used_orders or order.picked or order.delivered:
                    continue
                score = self._rough_pickup_score(shipper, order, orders, now, pickup_counts, drop_counts)
                if score > 0.0:
                    rough.append((score, order.et, order.id, order))
            rough.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, _, order in rough[:exact_cap]:
                if time.perf_counter() > step_deadline:
                    break
                score = self._exact_pickup_score(shipper, order, orders, now, pickup_counts, drop_counts)
                if score > 0.0:
                    candidates.append((score, shipper.id, order.id, order))

        scores_by_order: Dict[int, List[float]] = {}
        for score, _, oid, _ in candidates:
            scores_by_order.setdefault(oid, []).append(score)
        regrets: Dict[int, float] = {}
        for oid, values in scores_by_order.items():
            values.sort(reverse=True)
            regrets[oid] = values[0] - (values[1] if len(values) > 1 else 0.0)

        for score, sid, oid, order in sorted(
            candidates,
            key=lambda item: (item[0] + 0.20 * regrets.get(item[2], 0.0), item[0], item[3].p, -item[3].et, -item[2]),
            reverse=True,
        ):
            if sid in self.pickup_tasks or oid in used_orders:
                continue
            pickup = (order.sx, order.sy)
            cell_limit = max(1, min(3, pickup_counts.get(pickup, 1)))
            if used_cells.get(pickup, 0) >= cell_limit:
                continue
            self.pickup_tasks[sid] = oid
            used_orders.add(oid)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1

        backlog: List[Tuple[float, int, int, Order]] = []
        for shipper in pending:
            if now < self.T - self._endgame_window():
                continue
            if shipper.id in self.pickup_tasks or time.perf_counter() > step_deadline:
                continue
            rough: List[Tuple[float, int, int, Order]] = []
            for order in orders.values():
                if order.id in used_orders or order.picked or order.delivered or not self._can_take_light(shipper, order, orders):
                    continue
                travel = self._mdist(shipper.position, (order.sx, order.sy)) + self._mdist(
                    (order.sx, order.sy), (order.ex, order.ey)
                ) + 1
                finish = now + travel
                if finish >= self.T:
                    continue
                value = delivery_reward(order, finish, self.T) - 0.10 * travel
                rough.append((value, order.et, order.id, order))
            rough.sort(key=lambda item: (-item[0], item[1], item[2]))
            backlog_cap = exact_cap if self.C >= 15 else max(3, exact_cap // 2)
            for _, _, _, order in rough[:backlog_cap]:
                if time.perf_counter() > step_deadline:
                    break
                score = self._backlog_pickup_score(shipper, order, orders, now, pickup_counts)
                if score > 0.0:
                    backlog.append((score, shipper.id, order.id, order))

        for score, sid, oid, order in sorted(backlog, key=lambda item: (item[0], item[3].p, -item[3].et), reverse=True):
            if sid in self.pickup_tasks or oid in used_orders:
                continue
            pickup = (order.sx, order.sy)
            cell_limit = max(1, min(3, pickup_counts.get(pickup, 1)))
            if used_cells.get(pickup, 0) >= cell_limit:
                continue
            self.pickup_tasks[sid] = oid
            used_orders.add(oid)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1

        for shipper in pending:
            if shipper.id not in self.pickup_tasks:
                self.failed_assignment_retry[shipper.id] = now + 2

    def _target_priority(self, shipper: Shipper, kind: str, order: Order, now: int) -> float:
        goal = (order.ex, order.ey) if kind == "deliver" else (order.sx, order.sy)
        dist = self.router.distance(shipper.position, goal)
        if dist >= INF:
            return -INF
        if kind == "deliver":
            slack = order.et - (now + dist)
            return (
                1000.0
                + delivery_reward(order, now + dist, self.T)
                + 18.0 * order.p
                + max(0.0, 45.0 - slack) * (0.4 + 0.1 * order.p)
                - 0.08 * dist
            )
        slack = order.et - (now + dist)
        return 50.0 + 5.0 * order.p + max(0.0, 20.0 - slack) * 0.1 - 0.06 * dist

    def _stage_idle(self, idle: List[Shipper], occupied_goals: Set[Position], now: int) -> Dict[int, Position]:
        if self.C < 15 or not idle or now >= self.T - self._endgame_window():
            return {}
        age_scale = max(20.0, self.T / 12.0)
        ranked_cells: List[Tuple[float, Position]] = []
        for cell, heat in self.pickup_heat.items():
            if heat < 3 or cell in occupied_goals:
                continue
            age = now - self.pickup_last_seen.get(cell, now)
            observed_value = heat / (1.0 + age / age_scale)
            if observed_value >= 1.0:
                ranked_cells.append((observed_value, cell))
        ranked_cells.sort(key=lambda item: (-item[0], item[1]))
        ranked_cells = ranked_cells[: max(4, min(12, self.C // 2))]
        if not ranked_cells:
            return {}

        pair_values: List[Tuple[float, int, Position]] = []
        for shipper in idle:
            for observed_value, cell in ranked_cells:
                dist = self.router.distance(shipper.position, cell)
                if dist <= 0 or dist >= INF:
                    continue
                value = observed_value / (1.0 + 0.10 * dist)
                if value >= 0.45:
                    pair_values.append((value, shipper.id, cell))
        stage_limit = 1
        assignments: Dict[int, Position] = {}
        used_cells: Set[Position] = set()
        for _, sid, cell in sorted(pair_values, key=lambda item: (-item[0], item[1], item[2])):
            if len(assignments) >= stage_limit:
                break
            if sid in assignments or cell in used_cells:
                continue
            assignments[sid] = cell
            used_cells.add(cell)
        return assignments

    def _reserved_path(
        self,
        start: Position,
        goal: Position,
        vertex_res: Dict[int, Set[Position]],
        edge_res: Set[Tuple[int, Position, Position]],
    ) -> List[Position]:
        path = [start]
        pos = start
        needs_repair = False
        for t in range(1, self.horizon + 1):
            current_dist = self.router.distance(pos, goal)
            options: List[Tuple[int, int, Move, Position]] = []
            for move, nxt in self.router.neighbors(pos, include_wait=True):
                if nxt in vertex_res.get(t, set()) or (t, nxt, pos) in edge_res:
                    continue
                dist = self.router.distance(nxt, goal)
                wait_cost = 1 if move == "S" and pos != goal else 0
                options.append((dist, wait_cost, move, nxt))
            if options:
                options.sort(key=lambda item: (item[0], item[1], item[2]))
                _, _, _, nxt = options[0]
                if current_dist < INF and self.router.distance(nxt, goal) >= current_dist and pos != goal:
                    self.repairs += 1
                    needs_repair = True
            else:
                nxt = pos
                self.repairs += 1
                needs_repair = True
            path.append(nxt)
            pos = nxt

        if needs_repair and self.repair_search_budget > 0:
            self.repair_search_budget -= 1
            repaired = self._space_time_repair(start, goal, vertex_res, edge_res)
            if repaired is not None and self._path_key(repaired, goal) < self._path_key(path, goal):
                path = repaired

        previous = path[0]
        for t, nxt in enumerate(path[1:], 1):
            vertex_res.setdefault(t, set()).add(nxt)
            edge_res.add((t, previous, nxt))
            previous = nxt
        return path

    def _path_key(self, path: List[Position], goal: Position) -> Tuple[int, int]:
        final_dist = self.router.distance(path[-1], goal)
        stalls = sum(
            1 for previous, current in zip(path, path[1:]) if previous == current and current != goal
        )
        return final_dist, stalls

    def _space_time_repair(
        self,
        start: Position,
        goal: Position,
        vertex_res: Dict[int, Set[Position]],
        edge_res: Set[Tuple[int, Position, Position]],
    ) -> Optional[List[Position]]:
        """Find a short reserved detour when greedy movement stalls behind another agent."""
        layer: Dict[Position, Tuple[int, Tuple[Position, ...]]] = {start: (0, (start,))}
        for t in range(1, self.horizon + 1):
            next_layer: Dict[Position, Tuple[int, Tuple[Position, ...]]] = {}
            for pos, (stalls, path) in layer.items():
                for move, nxt in self.router.neighbors(pos, include_wait=True):
                    if nxt in vertex_res.get(t, set()) or (t, nxt, pos) in edge_res:
                        continue
                    candidate = (
                        stalls + (1 if move == "S" and pos != goal else 0),
                        path + (nxt,),
                    )
                    previous = next_layer.get(nxt)
                    if previous is None or candidate < previous:
                        next_layer[nxt] = candidate
            if not next_layer:
                return None
            layer = next_layer
        _, best = min(
            layer.items(),
            key=lambda item: (self.router.distance(item[0], goal), item[1][0], item[0]),
        )
        return list(best[1])

    def _plan_paths(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Tuple[str, Optional[Order], Position]],
        priorities: Dict[int, float],
        staying: Set[int],
    ) -> Dict[int, List[Position]]:
        vertex_res: Dict[int, Set[Position]] = {}
        edge_res: Set[Tuple[int, Position, Position]] = set()
        by_id = {shipper.id: shipper for shipper in shippers}
        for sid in staying:
            pos = by_id[sid].position
            for t in range(1, self.horizon + 1):
                vertex_res.setdefault(t, set()).add(pos)
                edge_res.add((t, pos, pos))
        paths: Dict[int, List[Position]] = {}
        self.repair_search_budget = max(1, min(6, len(targets) // 4 + 1))
        for sid in sorted(targets, key=lambda item: (-priorities.get(item, 0.0), item)):
            shipper = by_id[sid]
            path = self._reserved_path(shipper.position, targets[sid][2], vertex_res, edge_res)
            paths[sid] = path
        return paths

    def _decide(self, obs: dict, cheap_only: bool = False) -> Dict[int, Action]:
        now = int(obs["t"])
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])
        self._observe_pickups(orders, now)
        actions: Dict[int, Action] = {}
        targets: Dict[int, Tuple[str, Optional[Order], Position]] = {}
        priorities: Dict[int, float] = {}
        staying: Set[int] = set()
        idle: List[Shipper] = []

        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                staying.add(shipper.id)
                self.pickup_tasks.pop(shipper.id, None)
                self.delivery_commitments.pop(shipper.id, None)
                continue
            delivery = self._delivery_target(shipper, orders, now)
            if delivery is not None:
                goal = (delivery.ex, delivery.ey)
                targets[shipper.id] = ("deliver", delivery, goal)
                priorities[shipper.id] = self._target_priority(shipper, "deliver", delivery, now)
                self.pickup_tasks.pop(shipper.id, None)
                continue
            idle.append(shipper)

        fresh_orders = bool(obs.get("new_order_ids", []))
        assignable = [
            shipper
            for shipper in idle
            if (
                now >= self.failed_assignment_retry.get(shipper.id, -1)
                or shipper.id in self.pickup_tasks
                or (fresh_orders and self.C >= 15)
            )
        ]
        if cheap_only:
            step_budget = 0.02
        elif now >= self.T - self._endgame_window():
            step_budget = min(0.90, 0.12 + 0.032 * max(1, len(assignable)))
        else:
            step_budget = min(0.42, 0.04 + 0.014 * max(1, len(assignable)))
        self._assign_pickups(assignable, orders, now, time.perf_counter() + step_budget)

        unassigned_idle: List[Shipper] = []
        for shipper in idle:
            oid = self.pickup_tasks.get(shipper.id)
            order = orders.get(oid) if oid is not None else None
            if order is None or order.picked or not self._can_take_light(shipper, order, orders):
                self.pickup_tasks.pop(shipper.id, None)
                unassigned_idle.append(shipper)
                continue
            pickup = (order.sx, order.sy)
            if shipper.position == pickup:
                actions[shipper.id] = ("S", 1)
                staying.add(shipper.id)
                continue
            targets[shipper.id] = ("pickup", order, pickup)
            priorities[shipper.id] = self._target_priority(shipper, "pickup", order, now)

        staging = self._stage_idle(unassigned_idle, {target[2] for target in targets.values()}, now)
        for shipper in unassigned_idle:
            goal = staging.get(shipper.id)
            if goal is None:
                actions[shipper.id] = ("S", 0)
                staying.add(shipper.id)
                continue
            targets[shipper.id] = ("stage", None, goal)
            priorities[shipper.id] = 1.0

        paths = self._plan_paths(shippers, targets, priorities, staying)
        target_positions: Dict[int, Position] = {}
        claimed_pickups = {
            target_order.id
            for target_kind, target_order, _ in targets.values()
            if target_kind == "pickup" and target_order is not None
        }
        for sid, (kind, order, goal) in targets.items():
            shipper = next(s for s in shippers if s.id == sid)
            path = paths.get(sid, [shipper.position, shipper.position])
            nxt = path[1] if len(path) > 1 else shipper.position
            move = self.router.move_between(shipper.position, nxt)
            op = 0
            if nxt == goal:
                if kind == "pickup":
                    op = 1
                elif kind == "deliver":
                    op = 2
            elif kind == "deliver" and order is not None and len(shipper.bag) < shipper.K_max:
                passing = self._pickup_at(shipper, orders, nxt)
                if (
                    passing is not None
                    and passing.id not in claimed_pickups
                    and self._passing_pickup_is_worthwhile(shipper, order, passing, orders, now)
                ):
                    op = 1
                    claimed_pickups.add(passing.id)
                    self.delivery_commitments[sid] = order.id
            actions[sid] = (move, op)
            target_positions[sid] = goal

        return resolve_collisions_and_blocks(
            shippers,
            actions,
            self.grid,
            orders,
            self._pickup_at,
            self._deliverable,
            target_positions,
            self.router.distance,
            allow_unblock=False,
        )

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure(obs)
        self.run_deadline = start + max(180.0, min(900.0, 0.25 * self.T + 8.0 * self.C))
        while not obs.get("done", False):
            actions = self._decide(obs, cheap_only=time.time() > self.run_deadline)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
