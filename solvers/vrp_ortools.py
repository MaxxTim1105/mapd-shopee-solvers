from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
Stop = Tuple[str, int, Position]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class BFS:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.dist_cache: Dict[Tuple[Position, Position], int] = {}
        self.move_cache: Dict[Tuple[Position, Position], Move] = {}

    def neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
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

    def next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        if key in self.move_cache:
            return self.move_cache[key]
        q = deque([start])
        first = {start: "S"}
        while q:
            pos = q.popleft()
            for move, nxt in self.neighbors(pos):
                if nxt in first:
                    continue
                first[nxt] = move if pos == start else first[pos]
                if nxt == goal:
                    self.move_cache[key] = first[nxt]
                    return first[nxt]
                q.append(nxt)
        self.move_cache[key] = "S"
        return "S"

    def after(self, pos: Position, move: Move) -> Position:
        return valid_next_pos(pos, move, self.grid)


class VRPOrToolsSolver(Solver):
    """Rolling-horizon VRP solver with route insertion fallback."""

    method_name = "VRPOrToolsSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs: Optional[BFS] = None
        self.route_plan: Dict[int, List[Stop]] = {}
        self.run_deadline = 0.0
        self.grid: List[List[int]] = [[0]]
        self.N = 1
        self.C = 1
        self.T = 1
        self._configure_by_observation(1, 1, 1, [[0]])
        self.ortools_calls = 0
        self.ortools_successes = 0
        self.ortools_failures = 0
        self.ortools_available: Optional[bool] = None

    def _configure_by_observation(self, n: int, c: int, t: int, grid: List[List[int]]) -> None:
        self.N = n
        self.C = c
        self.T = t
        self.grid = grid
        self.bfs = BFS(grid)
        if c <= 2:
            self.travel_penalty = 0.012
            self.late_penalty = 0.20
        elif n >= 20:
            self.travel_penalty = 0.135
            self.late_penalty = 0.48
        elif n >= 17:
            self.travel_penalty = 0.075
            self.late_penalty = 0.38
        elif n >= 13:
            self.travel_penalty = 0.06
            self.late_penalty = 0.35
        else:
            self.travel_penalty = 0.04
            self.late_penalty = 0.30

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

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((o.ex, o.ey) == pos for o in self._carried(shipper, orders))

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        here = [o for o in orders.values() if (o.sx, o.sy) == pos and self._can_take(shipper, o, orders)]
        if not here:
            return None
        return min(here, key=lambda o: (-o.p, o.et, o.id))

    def _visible_pickup_density(self, order: Order, orders: Dict[int, Order]) -> float:
        density = 0.0
        for other in orders.values():
            if other.picked or other.delivered:
                continue
            md = abs(other.sx - order.sx) + abs(other.sy - order.sy)
            if md == 0:
                density += 1.1 + 0.30 * other.p
            elif md <= 4:
                density += 0.35 + 0.10 * other.p
        return density

    def _completion_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int) -> float:
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(shipper.position, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        finish = now + d1 + d2
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        slack = order.et - finish
        urgency = max(0.0, min(18.0, 18.0 - slack)) * (0.12 + 0.08 * order.p)
        return reward + 3.5 * order.p + self._visible_pickup_density(order, orders) + urgency - 0.28 * d1 - 0.10 * d2 - 0.85 * late

    def _candidate_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int, limit: int = 34) -> List[Order]:
        if self.N < 20:
            scored = []
            for order in orders.values():
                if not self._can_take(shipper, order, orders):
                    continue
                d1 = abs(shipper.r - order.sx) + abs(shipper.c - order.sy)
                d2 = abs(order.sx - order.ex) + abs(order.sy - order.ey)
                finish = now + d1 + d2
                late = max(0, finish - order.et)
                density = sum(
                    1
                    for other in orders.values()
                    if not other.picked and abs(other.sx - order.sx) + abs(other.sy - order.sy) <= 4
                )
                rough_score = delivery_reward(order, finish, self.T) + 1.8 * density + 4.0 * order.p - 0.35 * d1 - late
                scored.append((rough_score, order.et, order.id, order))
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            return [item[3] for item in scored[:limit]]

        rough = []
        for order in orders.values():
            if not self._can_take(shipper, order, orders):
                continue
            d1 = abs(shipper.r - order.sx) + abs(shipper.c - order.sy)
            d2 = abs(order.sx - order.ex) + abs(order.sy - order.ey)
            finish = now + d1 + d2
            late = max(0, finish - order.et)
            rough_score = delivery_reward(order, finish, self.T) + 4.0 * order.p + self._visible_pickup_density(order, orders) - 0.28 * d1 - 0.08 * d2 - late
            rough.append((rough_score, order.et, order.id, order))
        rough.sort(key=lambda item: (-item[0], item[1], item[2]))
        scored = []
        for _, _, _, order in rough[: max(limit, min(60, 2 * limit))]:
            score = self._completion_score(shipper, order, orders, now)
            if score > -INF:
                scored.append((score, order.et, order.id, order))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored[:limit]]

    def _route_score(self, shipper: Shipper, route: Sequence[Stop], orders: Dict[int, Order], now: int) -> float:
        pos = shipper.position
        t = now
        value = 0.0
        carried = set(shipper.bag)
        load = self._carried_weight(shipper, orders)
        picked: set[int] = set()
        for kind, oid, target in route:
            d = self.bfs.dist(pos, target)
            if d >= INF:
                return -INF
            value -= self.travel_penalty * d * (1.0 + 0.5 * load / max(shipper.W_max, 1.0))
            t += d
            pos = target
            order = orders.get(oid)
            if order is None:
                continue
            if kind == "pickup":
                if oid not in carried:
                    if self.N >= 20 and (len(carried) >= shipper.K_max or load + order.w > shipper.W_max):
                        return -INF
                    load += order.w
                    carried.add(oid)
                    picked.add(oid)
            elif kind == "deliver" and oid in carried:
                reward = delivery_reward(order, t, self.T)
                late = max(0, t - order.et)
                value += reward - self.late_penalty * late
                if oid in picked:
                    value += 1.0
                    if self.N >= 20:
                        value += 0.15 * self._visible_pickup_density(order, orders)
                if self.N >= 20:
                    same_dest = sum(
                        1
                        for other_id in carried
                        if other_id != oid and other_id in orders and (orders[other_id].ex, orders[other_id].ey) == target
                    )
                    value += 0.8 * same_dest
                load -= order.w
                carried.remove(oid)
        return value

    def _initial_route(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Stop]:
        carried = self._carried(shipper, orders)
        route: List[Stop] = []
        pos = shipper.position
        remaining = set(o.id for o in carried)
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

    def _best_insertion(
        self,
        shipper: Shipper,
        route: List[Stop],
        order: Order,
        orders: Dict[int, Order],
        now: int,
    ) -> Tuple[float, Optional[List[Stop]]]:
        base = self._route_score(shipper, route, orders, now)
        best_gain = -INF
        best_route = None
        n = len(route)
        pickup = ("pickup", order.id, (order.sx, order.sy))
        deliver = ("deliver", order.id, (order.ex, order.ey))
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                candidate = list(route)
                candidate.insert(i, pickup)
                candidate.insert(j, deliver)
                score = self._route_score(shipper, candidate, orders, now)
                gain = score - base
                if gain > best_gain:
                    best_gain = gain
                    best_route = candidate
        return best_gain, best_route

    def _build_vrp_routes(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, List[Stop]]:
        routes = {s.id: self._initial_route(s, orders, now) for s in shippers}
        assigned: set[int] = set()
        capacity_slots = {s.id: max(0, s.K_max - len(s.bag)) for s in shippers}
        max_rounds = sum(capacity_slots.values())
        for _ in range(max_rounds):
            if self.run_deadline and time.time() > self.run_deadline:
                break
            best = None
            for shipper in sorted(shippers, key=lambda s: s.id):
                if capacity_slots[shipper.id] <= 0:
                    continue
                for order in self._candidate_pool(shipper, orders, now, limit=30):
                    if order.id in assigned:
                        continue
                    gain, new_route = self._best_insertion(shipper, routes[shipper.id], order, orders, now)
                    if new_route is None or gain <= 0:
                        continue
                    key = (gain, order.p, -order.et, -order.id)
                    if best is None or key > best[0]:
                        best = (key, shipper.id, order.id, new_route)
            if best is None:
                break
            _, sid, oid, route = best
            routes[sid] = route
            assigned.add(oid)
            capacity_slots[sid] -= 1
        return routes

    def _try_ortools_assignment(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Optional[Dict[int, List[Stop]]]:
        if self.ortools_available is False:
            return None
        candidates = []
        for shipper in shippers:
            for order in self._candidate_pool(shipper, orders, now, limit=18):
                d1 = self.bfs.dist(shipper.position, (order.sx, order.sy))
                d2 = self.bfs.dist((order.sx, order.sy), (order.ex, order.ey))
                if d1 >= INF or d2 >= INF:
                    continue
                score = self._completion_score(shipper, order, orders, now)
                if score > 0:
                    candidates.append((shipper.id, order.id, int(score * 1000)))
        if not candidates:
            return None
        self.ortools_calls += 1
        try:
            from ortools.sat.python import cp_model  # type: ignore
        except Exception:
            self.ortools_available = False
            self.ortools_failures += 1
            return None
        self.ortools_available = True
        try:
            model = cp_model.CpModel()
            x = {(sid, oid): model.NewBoolVar(f"x_{sid}_{oid}") for sid, oid, _ in candidates}
            for shipper in shippers:
                vars_s = [var for (sid, _), var in x.items() if sid == shipper.id]
                if vars_s:
                    model.Add(sum(vars_s) <= max(0, shipper.K_max - len(shipper.bag)))
            for oid in {oid for _, oid, _ in candidates}:
                vars_o = [var for (_, order_id), var in x.items() if order_id == oid]
                model.Add(sum(vars_o) <= 1)
            weights = {(sid, oid): score for sid, oid, score in candidates}
            model.Maximize(sum(weights[pair] * var for pair, var in x.items()))
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 0.08
            solver.parameters.num_search_workers = 1
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                self.ortools_failures += 1
                return None
            routes = {s.id: self._initial_route(s, orders, now) for s in shippers}
            for (sid, oid), var in x.items():
                if solver.Value(var):
                    order = orders[oid]
                    routes[sid].append(("pickup", oid, (order.sx, order.sy)))
                    routes[sid].append(("deliver", oid, (order.ex, order.ey)))
            self.ortools_successes += 1
            return routes
        except Exception:
            self.ortools_failures += 1
            return None

    def _first_action(self, shipper: Shipper, route: List[Stop], orders: Dict[int, Order]) -> Action:
        while route:
            kind, oid, target = route[0]
            if kind == "pickup" and (oid not in orders or not self._can_take(shipper, orders[oid], orders)):
                route.pop(0)
                continue
            if kind == "deliver" and oid not in shipper.bag:
                route.pop(0)
                continue
            if self.bfs.dist(shipper.position, target) >= INF:
                route.pop(0)
                continue
            move = self.bfs.next_move(shipper.position, target)
            nxt = self.bfs.after(shipper.position, move)
            op = 0
            if nxt == target:
                op = 1 if kind == "pickup" else 2
            if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
                op = 0
            if op == 2 and not self._deliverable(shipper, orders, nxt):
                op = 0
            return move, op
        return "S", 0

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
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])
        now = int(obs["t"])
        actions: Dict[int, Action] = {}
        target_positions: Dict[int, Position] = {}
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
            elif self._pickup_at(shipper, orders, shipper.position) is not None and not shipper.bag:
                actions[shipper.id] = ("S", 1)

        active = [s for s in shippers if s.id not in actions]
        routes = self._try_ortools_assignment(active, orders, now)
        if routes is None:
            routes = self._build_vrp_routes(active, orders, now)
        self.route_plan = routes
        for shipper in active:
            actions[shipper.id] = self._first_action(shipper, routes.get(shipper.id, []), orders)
            if routes.get(shipper.id):
                target_positions[shipper.id] = routes[shipper.id][0][2]
        return self._avoid_collisions(obs, actions, target_positions)

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])
        self.run_deadline = start + max(20.0, min(90.0, 0.10 * self.T + 8.0 * self.C))
        while not obs.get("done", False):
            if time.time() > self.run_deadline:
                actions = {s.id: ("S", 2) for s in obs["shippers"]}
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
