from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
Stop = Tuple[str, int, Position]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
ALL_MOVES: Tuple[Move, ...] = ("S", "U", "D", "L", "R")


class BFS:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.dist_cache: Dict[Tuple[Position, Position], int] = {}

    def neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        moves = ALL_MOVES if include_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if move == "S" or nxt != pos:
                yield move, nxt

    def dist(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        key = (start, goal)
        if key in self.dist_cache:
            return self.dist_cache[key]
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            self.dist_cache[key] = INF
            return INF
        q = deque([start])
        dist = {start: 0}
        while q:
            pos = q.popleft()
            for _, nxt in self.neighbors(pos):
                if nxt in dist:
                    continue
                nd = dist[pos] + 1
                if nxt == goal:
                    self.dist_cache[key] = nd
                    self.dist_cache[(goal, start)] = nd
                    return nd
                dist[nxt] = nd
                q.append(nxt)
        self.dist_cache[key] = INF
        return INF

    def move_between(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        for move in MOVES:
            if valid_next_pos(a, move, self.grid) == b:
                return move
        return "S"


class MAPDCBSSolver(Solver):
    """General online MAPD solver with route-aware tasks and CBS-lite paths."""

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs: Optional[BFS] = None
        self.grid: List[List[int]] = [[0]]
        self.N = 1
        self.C = 1
        self.T = 1
        self.horizon = 8
        self.run_deadline = 0.0
        self.repairs = 0
        self.paths: Dict[int, List[Position]] = {}
        self.route_plan: Dict[int, List[Stop]] = {}
        self.target_scores: Dict[int, float] = {}

    def _configure_by_observation(self, n: int, c: int, t: int, grid: List[List[int]]) -> None:
        self.N = n
        self.C = c
        self.T = t
        self.grid = grid
        self.bfs = BFS(grid)
        self.horizon = max(8, min(18, 5 + c + n // 8))
        self.route_travel_penalty = 0.05 + min(0.08, 0.02 * max(0, c - 2))
        self.route_late_penalty = 0.42 + min(0.35, 0.015 * n)

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _carried(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _can_take(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered or len(shipper.bag) >= shipper.K_max:
            return False
        if self._carried_weight(shipper, orders) + order.w > shipper.W_max:
            return False
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        return self.bfs.dist(shipper.position, pickup) < INF and self.bfs.dist(pickup, drop) < INF

    def _can_finish_before_end(self, shipper: Shipper, order: Order, now: int) -> bool:
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(shipper.position, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return False
        remaining = self.T - now
        margin = max(1, min(7, remaining // 14))
        return d1 + d2 + margin <= remaining

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((order.ex, order.ey) == pos for order in self._carried(shipper, orders))

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        candidates = [o for o in orders.values() if (o.sx, o.sy) == pos and self._can_take(shipper, o, orders)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: (-o.p, o.et, o.id))

    def _visible_pickup_density(self, order: Order, orders: Dict[int, Order], radius: int = 4) -> float:
        density = 0.0
        for other in orders.values():
            if other.picked or other.delivered:
                continue
            md = abs(other.sx - order.sx) + abs(other.sy - order.sy)
            if md == 0:
                density += 1.0 + 0.25 * other.p
            elif md <= radius:
                density += (0.45 + 0.08 * other.p) / (1.0 + 0.25 * md)
        return density

    def _visible_drop_density(self, drop: Position, orders: Dict[int, Order], radius: int = 4) -> float:
        density = 0.0
        for other in orders.values():
            if other.delivered:
                continue
            md = abs(other.ex - drop[0]) + abs(other.ey - drop[1])
            if md <= radius:
                density += (0.4 + 0.08 * other.p) / (1.0 + md)
        return density

    def _pickup_value(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int, pos: Optional[Position] = None) -> float:
        start = shipper.position if pos is None else pos
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(start, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        finish = now + d1 + d2
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        slack = order.et - finish
        urgency = max(0.0, min(24.0, 24.0 - slack)) * (0.08 + 0.05 * order.p)
        density = self._visible_pickup_density(order, orders)
        drop_density = self._visible_drop_density(drop, orders)
        return reward + 3.0 * order.p + 1.1 * density + 0.6 * drop_density + urgency - 0.28 * d1 - 0.10 * d2 - 0.8 * late

    def _candidate_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Order]:
        rough = []
        for order in orders.values():
            if not self._can_take(shipper, order, orders):
                continue
            if not self._can_finish_before_end(shipper, order, now):
                continue
            d1 = abs(shipper.r - order.sx) + abs(shipper.c - order.sy)
            d2 = abs(order.sx - order.ex) + abs(order.sy - order.ey)
            finish = now + d1 + d2
            late = max(0, finish - order.et)
            score = delivery_reward(order, finish, self.T) + 4.0 * order.p + self._visible_pickup_density(order, orders) - 0.25 * d1 - 0.08 * d2 - late
            rough.append((score, order.et, order.id, order))
        rough.sort(key=lambda item: (-item[0], item[1], item[2]))
        limit = max(16, min(44, 10 + 4 * self.C + len(orders) // 8))
        scored = []
        for _, _, _, order in rough[:limit]:
            exact = self._pickup_value(shipper, order, orders, now)
            if exact > -INF:
                scored.append((exact, order.et, order.id, order))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored[: min(limit, 34)]]

    def _route_score(self, shipper: Shipper, route: Sequence[Stop], orders: Dict[int, Order], now: int) -> float:
        pos = shipper.position
        t = now
        value = 0.0
        carried = set(shipper.bag)
        load = self._carried_weight(shipper, orders)
        picked: Set[int] = set()
        for kind, oid, target in route:
            d = self.bfs.dist(pos, target)
            if d >= INF:
                return -INF
            value -= self.route_travel_penalty * d * (1.0 + 0.35 * load / max(shipper.W_max, 1.0))
            t += d
            pos = target
            order = orders.get(oid)
            if order is None:
                continue
            if kind == "pickup":
                if oid not in carried:
                    if len(carried) >= shipper.K_max or load + order.w > shipper.W_max:
                        return -INF
                    carried.add(oid)
                    picked.add(oid)
                    load += order.w
            elif kind == "deliver" and oid in carried:
                reward = delivery_reward(order, t, self.T)
                late = max(0, t - order.et)
                slack = order.et - t
                same_dest = sum(
                    1
                    for other_id in carried
                    if other_id != oid and other_id in orders and (orders[other_id].ex, orders[other_id].ey) == target
                )
                low_slack_penalty = max(0.0, 4.0 - slack) * (0.05 + 0.03 * order.p)
                value += reward - self.route_late_penalty * late - low_slack_penalty + 0.8 * same_dest
                if oid in picked:
                    value += 0.7 + 0.2 * self._visible_drop_density(target, orders)
                carried.remove(oid)
                load -= order.w
        return value

    def _carried_detour_risk(
        self,
        shipper: Shipper,
        route: Sequence[Stop],
        candidate: Sequence[Stop],
        orders: Dict[int, Order],
        now: int,
    ) -> float:
        carried_ids = set(shipper.bag)
        if not carried_ids:
            return 0.0

        def finish_times(path: Sequence[Stop]) -> Dict[int, int]:
            pos = shipper.position
            t = now
            result: Dict[int, int] = {}
            for kind, oid, target in path:
                d = self.bfs.dist(pos, target)
                if d >= INF:
                    return {}
                t += d
                pos = target
                if kind == "deliver" and oid in carried_ids:
                    result[oid] = t
            return result

        base_finish = finish_times(route)
        candidate_finish = finish_times(candidate)
        if len(candidate_finish) < len(carried_ids):
            return INF
        risk = 0.0
        for oid in carried_ids:
            order = orders.get(oid)
            if order is None:
                continue
            base_t = base_finish.get(oid, now)
            cand_t = candidate_finish.get(oid, INF)
            delay = max(0, cand_t - base_t)
            slack = order.et - cand_t
            risk = max(risk, 1.3 * delay + 5.0 * max(0, -slack) + 0.7 * max(0, 3 - slack))
        return risk

    def _initial_route(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Stop]:
        route: List[Stop] = []
        pos = shipper.position
        remaining = {order.id for order in self._carried(shipper, orders)}
        while remaining:
            best_oid = min(
                remaining,
                key=lambda oid: (
                    self.bfs.dist(pos, (orders[oid].ex, orders[oid].ey)),
                    orders[oid].et,
                    -orders[oid].p,
                    oid,
                ),
            )
            order = orders[best_oid]
            route.append(("deliver", order.id, (order.ex, order.ey)))
            pos = (order.ex, order.ey)
            remaining.remove(best_oid)
        return route

    def _best_insertion(self, shipper: Shipper, route: List[Stop], order: Order, orders: Dict[int, Order], now: int) -> Tuple[float, Optional[List[Stop]]]:
        base = self._route_score(shipper, route, orders, now)
        best_gain = -INF
        best_route = None
        pickup = ("pickup", order.id, (order.sx, order.sy))
        deliver = ("deliver", order.id, (order.ex, order.ey))
        for i in range(len(route) + 1):
            for j in range(i + 1, len(route) + 2):
                candidate = list(route)
                candidate.insert(i, pickup)
                candidate.insert(j, deliver)
                score = self._route_score(shipper, candidate, orders, now)
                risk = self._carried_detour_risk(shipper, route, candidate, orders, now)
                gain = score - base - risk
                if gain > best_gain:
                    best_gain = gain
                    best_route = candidate
        return best_gain, best_route

    def _build_routes(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, List[Stop]]:
        routes = {shipper.id: self._initial_route(shipper, orders, now) for shipper in shippers}
        assigned: Set[int] = set()
        slots = {shipper.id: max(0, shipper.K_max - len(shipper.bag)) for shipper in shippers}
        for _ in range(sum(slots.values())):
            if self.run_deadline and time.time() > self.run_deadline:
                break
            best = None
            for shipper in sorted(shippers, key=lambda s: s.id):
                if slots[shipper.id] <= 0:
                    continue
                for order in self._candidate_pool(shipper, orders, now):
                    if order.id in assigned:
                        continue
                    gain, route = self._best_insertion(shipper, routes[shipper.id], order, orders, now)
                    if route is None or gain <= 0:
                        continue
                    key = (gain, order.p, -order.et, -order.id)
                    if best is None or key > best[0]:
                        best = (key, shipper.id, order.id, route)
            if best is None:
                break
            _, sid, oid, route = best
            routes[sid] = route
            assigned.add(oid)
            slots[sid] -= 1
        return routes

    def _idle_pickup_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        if shipper.bag:
            return None
        candidates = []
        for order in self._candidate_pool(shipper, orders, now):
            score = self._pickup_value(shipper, order, orders, now)
            if score <= 0:
                continue
            pickup_dist = self.bfs.dist(shipper.position, (order.sx, order.sy))
            finish_dist = pickup_dist + self.bfs.dist((order.sx, order.sy), (order.ex, order.ey))
            slack = min(order.et - (now + finish_dist), self.T - now - finish_dist)
            candidates.append((score + 0.18 * max(0.0, slack) - 0.12 * pickup_dist, order.et, order.id, order))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return candidates[0][3]

    def _target_priority(self, shipper: Shipper, target: Optional[Tuple[str, int, Position]], orders: Dict[int, Order], now: int) -> float:
        if target is None:
            return 0.0
        kind, oid, pos = target
        order = orders.get(oid)
        dist = self.bfs.dist(shipper.position, pos)
        if order is None or dist >= INF:
            return 0.0
        if kind == "deliver":
            finish = now + dist
            slack = order.et - finish
            return 180.0 + delivery_reward(order, finish, self.T) + 5.0 * order.p + max(0.0, 35.0 - slack) - 0.15 * dist
        finish = now + dist + self.bfs.dist((order.sx, order.sy), (order.ex, order.ey))
        return self._pickup_value(shipper, order, orders, now) + max(0.0, 20.0 - (order.et - finish)) * 0.2

    def _reserved_path(
        self,
        start: Position,
        goal: Position,
        vertex_res: Dict[int, Set[Position]],
        edge_res: Set[Tuple[int, Position, Position]],
    ) -> List[Position]:
        if start == goal:
            return [start] * (self.horizon + 1)
        base_dist = self.bfs.dist(start, goal)
        if base_dist >= INF:
            return [start] * (self.horizon + 1)
        max_depth = min(80, max(self.horizon, base_dist + self.horizon))
        q = deque([(start, 0)])
        parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {(start, 0): None}
        best_state = (start, 0)
        best_key = (base_dist, 0)
        goal_state = None
        while q:
            pos, t = q.popleft()
            d_goal = self.bfs.dist(pos, goal)
            key = (d_goal, -t)
            if key < best_key:
                best_key = key
                best_state = (pos, t)
            if pos == goal:
                goal_state = (pos, t)
                break
            if t >= max_depth:
                continue
            for _, nxt in self.bfs.neighbors(pos, include_wait=True):
                nt = t + 1
                if nxt in vertex_res.get(nt, set()):
                    continue
                if (nt, nxt, pos) in edge_res:
                    continue
                state = (nxt, nt)
                if state in parent:
                    continue
                parent[state] = (pos, t)
                q.append(state)
        end_state = goal_state if goal_state is not None else best_state
        path = [end_state[0]]
        cur = end_state
        while parent[cur] is not None:
            cur = parent[cur]
            path.append(cur[0])
        path.reverse()
        while len(path) <= self.horizon:
            path.append(path[-1])
        return path

    def _plan_paths(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Tuple[str, int, Position]],
        priorities: Dict[int, float],
    ) -> Dict[int, List[Position]]:
        paths: Dict[int, List[Position]] = {}
        vertex_res: Dict[int, Set[Position]] = {}
        edge_res: Set[Tuple[int, Position, Position]] = set()
        for shipper in sorted(shippers, key=lambda s: (-priorities.get(s.id, 0.0), s.id)):
            target = targets.get(shipper.id)
            goal = shipper.position if target is None else target[2]
            path = self._reserved_path(shipper.position, goal, vertex_res, edge_res)
            paths[shipper.id] = path
            for t in range(1, self.horizon + 1):
                prev = path[min(t - 1, len(path) - 1)]
                cur = path[min(t, len(path) - 1)]
                vertex_res.setdefault(t, set()).add(cur)
                edge_res.add((t, prev, cur))
        return paths

    def _action_from_path(self, shipper: Shipper, path: List[Position], target: Optional[Tuple[str, int, Position]], orders: Dict[int, Order]) -> Action:
        nxt = path[1] if len(path) > 1 else shipper.position
        move = self.bfs.move_between(shipper.position, nxt)
        op = 0
        if target is not None and nxt == target[2]:
            op = 1 if target[0] == "pickup" else 2
        if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
            op = 0
        if op == 2 and not self._deliverable(shipper, orders, nxt):
            op = 0
        return move, op

    def _decide(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])
        now = int(obs["t"])
        actions: Dict[int, Action] = {}
        active: List[Shipper] = []
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
            else:
                active.append(shipper)

        routes = self._build_routes(active, orders, now)
        targets: Dict[int, Tuple[str, int, Position]] = {}
        priorities: Dict[int, float] = {}
        for shipper in active:
            route = routes.get(shipper.id, [])
            if route:
                targets[shipper.id] = route[0]
            else:
                here = self._pickup_at(shipper, orders, shipper.position)
                if here is not None and not shipper.bag:
                    targets[shipper.id] = ("pickup", here.id, (here.sx, here.sy))
                else:
                    idle_order = self._idle_pickup_target(shipper, orders, now)
                    if idle_order is not None:
                        targets[shipper.id] = ("pickup", idle_order.id, (idle_order.sx, idle_order.sy))
            priorities[shipper.id] = self._target_priority(shipper, targets.get(shipper.id), orders, now)

        paths = self._plan_paths(active, targets, priorities)
        self.paths = paths
        self.route_plan = routes
        self.target_scores = priorities
        for shipper in active:
            actions[shipper.id] = self._action_from_path(shipper, paths.get(shipper.id, [shipper.position]), targets.get(shipper.id), orders)

        return resolve_collisions_and_blocks(
            shippers,
            actions,
            self.grid,
            orders,
            self._pickup_at,
            self._deliverable,
            {sid: target[2] for sid, target in targets.items()},
            self.bfs.dist,
            allow_unblock=True,
        )

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])
        self.run_deadline = start + max(20.0, min(140.0, 0.14 * self.T + 10.0 * self.C))
        while not obs.get("done", False):
            if time.time() > self.run_deadline:
                actions = {s.id: ("S", 2) for s in obs["shippers"]}
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
