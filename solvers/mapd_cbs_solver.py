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
        self.path_cache: Dict[Tuple[Position, Position], List[Position]] = {}

    def neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        moves = ALL_MOVES if include_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if move == "S" or nxt != pos:
                yield move, nxt

    def dist(self, start: Position, goal: Position) -> int:
        path = self.path(start, goal)
        return INF if not path else len(path) - 1

    def path(self, start: Position, goal: Position) -> List[Position]:
        if start == goal:
            return [start]
        key = (start, goal)
        if key in self.path_cache:
            return list(self.path_cache[key])
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return []
        q = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}
        while q:
            pos = q.popleft()
            for _, nxt in self.neighbors(pos):
                if nxt in parent:
                    continue
                parent[nxt] = pos
                if nxt == goal:
                    out = [goal]
                    cur = goal
                    while parent[cur] is not None:
                        cur = parent[cur]
                        out.append(cur)
                    out.reverse()
                    self.path_cache[key] = out
                    self.dist_cache[key] = len(out) - 1
                    return list(out)
                q.append(nxt)
        return []

    def move_between(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        for move in MOVES:
            if valid_next_pos(a, move, self.grid) == b:
                return move
        return "S"


class MAPDCBSSolver(Solver):
    """MAPD solver with short-horizon CBS-lite conflict repair."""

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs = BFS(env.grid)
        self.paths: Dict[int, List[Position]] = {}
        self.constraints: Dict[int, Set[Tuple[int, Position]]] = {}
        self.edge_constraints: Dict[int, Set[Tuple[int, Position, Position]]] = {}
        self.repairs = 0
        if 11 <= env.N <= 12:
            self.horizon = 10
        else:
            self.horizon = 6 if env.N >= 20 else 8
        self.run_deadline = 0.0
        self.use_small_global_assignment = env.N <= 10
        self.use_midlarge_route_assignment = 17 <= env.N < 20
        self.route_travel_penalty = 0.09 if self.use_midlarge_route_assignment else 0.012
        self.route_late_penalty = 0.50 if self.use_midlarge_route_assignment else 0.20
        self.use_medium_delivery_rescue = 13 <= env.N <= 16
        self.large_followup_weight = 0.25
        self.large_density_weight = 0.30
        self.mid_followup_weight = 0.0
        self.mid_density_weight = 0.0

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
        candidates = [o for o in orders.values() if (o.sx, o.sy) == pos and self._can_take(shipper, o, orders)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: (-o.p, o.et, o.id))

    def _next_move_to(self, start: Position, goal: Position) -> Move:
        path = self.bfs.path(start, goal)
        if len(path) <= 1:
            return "S"
        return self.bfs.move_between(start, path[1])

    def _small_candidate_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int, limit: int = 28) -> List[Order]:
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
            score = delivery_reward(order, finish, self.env.T) + 1.8 * density + 4.0 * order.p - 0.35 * d1 - late
            scored.append((score, order.et, order.id, order))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored[:limit]]

    def _small_route_score(self, shipper: Shipper, route: Sequence[Stop], orders: Dict[int, Order], now: int) -> float:
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
            value -= self.route_travel_penalty * d * (1.0 + 0.5 * load / max(shipper.W_max, 1.0))
            t += d
            pos = target
            order = orders.get(oid)
            if order is None:
                continue
            if kind == "pickup":
                if oid not in carried:
                    if self.use_midlarge_route_assignment and (len(carried) >= shipper.K_max or load + order.w > shipper.W_max):
                        return -INF
                    load += order.w
                    carried.add(oid)
                    picked.add(oid)
            elif kind == "deliver" and oid in carried:
                reward = delivery_reward(order, t, self.env.T)
                late = max(0, t - order.et)
                value += reward - self.route_late_penalty * late
                if oid in picked:
                    value += 1.0
                load -= order.w
                carried.remove(oid)
        return value

    def _small_initial_route(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> List[Stop]:
        route: List[Stop] = []
        pos = shipper.position
        remaining = set(o.id for o in self._carried(shipper, orders))
        while remaining:
            best_oid = None
            best_key = None
            for oid in remaining:
                order = orders[oid]
                target = (order.ex, order.ey)
                d = self.bfs.dist(pos, target)
                key = (order.et - (now + d), -delivery_reward(order, now + d, self.env.T), d, oid)
                if best_key is None or key < best_key:
                    best_key = key
                    best_oid = oid
            order = orders[best_oid]
            route.append(("deliver", order.id, (order.ex, order.ey)))
            pos = (order.ex, order.ey)
            remaining.remove(best_oid)
        return route

    def _small_best_insertion(
        self,
        shipper: Shipper,
        route: List[Stop],
        order: Order,
        orders: Dict[int, Order],
        now: int,
    ) -> Tuple[float, Optional[List[Stop]]]:
        base = self._small_route_score(shipper, route, orders, now)
        best_gain = -INF
        best_route = None
        pickup = ("pickup", order.id, (order.sx, order.sy))
        deliver = ("deliver", order.id, (order.ex, order.ey))
        for i in range(len(route) + 1):
            for j in range(i + 1, len(route) + 2):
                candidate = list(route)
                candidate.insert(i, pickup)
                candidate.insert(j, deliver)
                score = self._small_route_score(shipper, candidate, orders, now)
                gain = score - base
                if gain > best_gain:
                    best_gain = gain
                    best_route = candidate
        return best_gain, best_route

    def _small_build_routes(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, List[Stop]]:
        routes = {s.id: self._small_initial_route(s, orders, now) for s in shippers}
        assigned: set[int] = set()
        capacity_slots = {s.id: max(0, s.K_max - len(s.bag)) for s in shippers}
        for _ in range(sum(capacity_slots.values())):
            best = None
            for shipper in sorted(shippers, key=lambda s: s.id):
                if capacity_slots[shipper.id] <= 0:
                    continue
                for order in self._small_candidate_pool(shipper, orders, now):
                    if order.id in assigned:
                        continue
                    gain, route = self._small_best_insertion(shipper, routes[shipper.id], order, orders, now)
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
            capacity_slots[sid] -= 1
        return routes

    def _small_first_action(self, shipper: Shipper, route: List[Stop], orders: Dict[int, Order]) -> Action:
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
            move = self._next_move_to(shipper.position, target)
            nxt = valid_next_pos(shipper.position, move, self.env.grid)
            op = 0
            if nxt == target:
                op = 1 if kind == "pickup" else 2
            if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
                op = 0
            if op == 2 and not self._deliverable(shipper, orders, nxt):
                op = 0
            return move, op
        return "S", 0

    def _small_avoid_collisions(
        self,
        obs: dict,
        actions: Dict[int, Action],
        target_positions: Optional[Dict[int, Position]] = None,
    ) -> Dict[int, Action]:
        return resolve_collisions_and_blocks(
            list(obs["shippers"]),
            actions,
            self.env.grid,
            obs["orders"],
            self._pickup_at,
            self._deliverable,
            target_positions,
            self.bfs.dist,
        )

    def _small_route_decide(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])
        now = int(obs["t"])
        actions: Dict[int, Action] = {}
        target_positions: Dict[int, Position] = {}
        active = []
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
            elif self._pickup_at(shipper, orders, shipper.position) is not None and not shipper.bag:
                actions[shipper.id] = ("S", 1)
            else:
                active.append(shipper)
        routes = self._small_build_routes(active, orders, now)
        for shipper in active:
            actions[shipper.id] = self._small_first_action(shipper, routes.get(shipper.id, []), orders)
            if routes.get(shipper.id):
                target_positions[shipper.id] = routes[shipper.id][0][2]
        return self._small_avoid_collisions(obs, actions, target_positions)

    def _task_priority(self, shipper: Shipper, target_kind: str, oid: int, orders: Dict[int, Order], now: int) -> float:
        order = orders.get(oid)
        if order is None:
            return 0.0
        target = (order.ex, order.ey) if target_kind == "deliver" else (order.sx, order.sy)
        d = self.bfs.dist(shipper.position, target)
        if d >= INF:
            return -INF
        finish = now + (0 if d >= INF else d)
        reward = delivery_reward(order, finish, self.env.T)
        slack = order.et - finish
        if target_kind == "deliver" and self.use_medium_delivery_rescue:
            rescue_urgency = max(0.0, 35.0 - max(0.0, slack)) * order.p if slack >= 0 else 0.0
            late_penalty = 1.2 * max(0, -slack)
            return 120.0 + reward + rescue_urgency - late_penalty - 0.35 * min(d, 1000)
        urgency = max(0, 40 - slack) * order.p
        carry_bonus = 100.0 if target_kind == "deliver" else 0.0
        return carry_bonus + reward + urgency - 0.2 * min(d, 1000)

    def _visible_density(self, pos: Position, orders: Dict[int, Order], radius: int = 4) -> float:
        density = 0.0
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            md = abs(order.sx - pos[0]) + abs(order.sy - pos[1])
            if md == 0:
                density += 1.0 + 0.25 * order.p
            elif md <= radius:
                density += 0.35 + 0.08 * order.p
        return density

    def _followup_bonus(self, after_pos: Position, first_oid: int, orders: Dict[int, Order], now: int) -> float:
        best = 0.0
        for order in orders.values():
            if order.id == first_oid or order.picked or order.delivered:
                continue
            pickup = (order.sx, order.sy)
            drop = (order.ex, order.ey)
            d1 = self.bfs.dist(after_pos, pickup)
            d2 = self.bfs.dist(pickup, drop)
            if d1 >= INF or d2 >= INF:
                continue
            finish = now + d1 + d2
            late = max(0, finish - order.et)
            value = delivery_reward(order, finish, self.env.T) + 2.5 * order.p - 0.22 * d1 - 0.08 * d2 - 0.6 * late
            if value > best:
                best = value
        return best

    def _assign_targets(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, Tuple[str, int, Position]]:
        targets: Dict[int, Tuple[str, int, Position]] = {}
        reserved: set[int] = set()
        idle: List[Shipper] = []
        for shipper in sorted(shippers, key=lambda s: s.id):
            carried = [
                order
                for order in self._carried(shipper, orders)
                if self.bfs.dist(shipper.position, (order.ex, order.ey)) < INF
            ]
            if carried:
                order = max(
                    carried,
                    key=lambda o: self._task_priority(shipper, "deliver", o.id, orders, now),
                )
                targets[shipper.id] = ("deliver", order.id, (order.ex, order.ey))
                continue
            if self.use_small_global_assignment:
                idle.append(shipper)
                continue
            best = None
            for order in orders.values():
                if order.id in reserved or not self._can_take(shipper, order, orders):
                    continue
                pickup = (order.sx, order.sy)
                drop = (order.ex, order.ey)
                d1 = self.bfs.dist(shipper.position, pickup)
                d2 = self.bfs.dist(pickup, drop)
                if d1 >= INF or d2 >= INF:
                    continue
                finish = now + d1 + d2
                late = max(0, finish - order.et)
                score = delivery_reward(order, finish, self.env.T) + 3.0 * order.p - 0.3 * d1 - 0.1 * d2 - late
                if self.env.N >= 20:
                    score += self.large_followup_weight * self._followup_bonus(drop, order.id, orders, finish)
                    score += self.large_density_weight * self._visible_density(drop, orders)
                elif self.mid_followup_weight or self.mid_density_weight:
                    score += self.mid_followup_weight * self._followup_bonus(drop, order.id, orders, finish)
                    score += self.mid_density_weight * self._visible_density(drop, orders)
                if best is None or score > best[0]:
                    best = (score, order)
            if best is not None and best[0] > 0:
                order = best[1]
                reserved.add(order.id)
                targets[shipper.id] = ("pickup", order.id, (order.sx, order.sy))
        if idle:
            pair_scores = []
            for shipper in idle:
                for order in orders.values():
                    if not self._can_take(shipper, order, orders):
                        continue
                    pickup = (order.sx, order.sy)
                    drop = (order.ex, order.ey)
                    d1 = self.bfs.dist(shipper.position, pickup)
                    d2 = self.bfs.dist(pickup, drop)
                    if d1 >= INF or d2 >= INF:
                        continue
                    finish = now + d1 + d2
                    late = max(0, finish - order.et)
                    density = sum(
                        1
                        for other in orders.values()
                        if not other.picked and abs(other.sx - order.sx) + abs(other.sy - order.sy) <= 3
                    )
                    score = delivery_reward(order, finish, self.env.T) + 3.0 * order.p + 0.8 * density - 0.3 * d1 - 0.1 * d2 - late
                    if score > 0:
                        pair_scores.append((score, -d1, order.p, -order.et, -shipper.id, shipper, order))
            pair_scores.sort(reverse=True)
            used_shippers: set[int] = set()
            for _, _, _, _, _, shipper, order in pair_scores:
                if shipper.id in used_shippers or order.id in reserved:
                    continue
                used_shippers.add(shipper.id)
                reserved.add(order.id)
                targets[shipper.id] = ("pickup", order.id, (order.sx, order.sy))
        return targets

    def _constrained_path(
        self,
        sid: int,
        start: Position,
        goal: Position,
        vertex_constraints: Set[Tuple[int, Position]],
        edge_constraints: Set[Tuple[int, Position, Position]],
    ) -> List[Position]:
        if start == goal:
            return [start] + [start] * self.horizon
        depth_cap = 32 if self.horizon <= 6 else 48
        max_depth = min(depth_cap, max(self.horizon, self.bfs.dist(start, goal) + self.horizon))
        q = deque([(start, 0)])
        parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {(start, 0): None}
        best_state = None
        while q:
            pos, t = q.popleft()
            if pos == goal:
                best_state = (pos, t)
                break
            if t >= max_depth:
                continue
            for _, nxt in self.bfs.neighbors(pos, include_wait=True):
                nt = t + 1
                if (nt, nxt) in vertex_constraints or (nt, pos, nxt) in edge_constraints:
                    continue
                state = (nxt, nt)
                if state in parent:
                    continue
                parent[state] = (pos, t)
                q.append(state)
        if best_state is None:
            base = self.bfs.path(start, goal)
            return base if base else [start] + [start] * self.horizon
        out = [best_state[0]]
        cur = best_state
        while parent[cur] is not None:
            cur = parent[cur]
            out.append(cur[0])
        out.reverse()
        while len(out) <= self.horizon:
            out.append(out[-1])
        return out

    def _priority(self, sid: int, shippers: Dict[int, Shipper], targets: Dict[int, Tuple[str, int, Position]], orders: Dict[int, Order], now: int) -> float:
        shipper = shippers[sid]
        target = targets.get(sid)
        if target is None:
            return 0.0
        return self._task_priority(shipper, target[0], target[1], orders, now)

    def _find_conflict(self, paths: Dict[int, List[Position]]) -> Optional[Tuple[str, int, int, int, Position]]:
        ids = sorted(paths)
        for t in range(1, self.horizon + 1):
            seen: Dict[Position, int] = {}
            for sid in ids:
                path = paths[sid]
                pos = path[min(t, len(path) - 1)]
                if pos in seen:
                    return "vertex", t, seen[pos], sid, pos
                seen[pos] = sid
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    pa0 = paths[a][min(t - 1, len(paths[a]) - 1)]
                    pa1 = paths[a][min(t, len(paths[a]) - 1)]
                    pb0 = paths[b][min(t - 1, len(paths[b]) - 1)]
                    pb1 = paths[b][min(t, len(paths[b]) - 1)]
                    if pa0 == pb1 and pb0 == pa1 and pa0 != pa1:
                        return "edge", t, a, b, pa1
        return None

    def _plan_paths(self, obs: dict, targets: Dict[int, Tuple[str, int, Position]]) -> Dict[int, List[Position]]:
        orders: Dict[int, Order] = obs["orders"]
        now = int(obs["t"])
        shippers = {s.id: s for s in obs["shippers"]}
        constraints: Dict[int, Set[Tuple[int, Position]]] = {sid: set() for sid in shippers}
        edge_constraints: Dict[int, Set[Tuple[int, Position, Position]]] = {sid: set() for sid in shippers}
        paths: Dict[int, List[Position]] = {}
        for sid, shipper in shippers.items():
            target = targets.get(sid)
            goal = shipper.position if target is None else target[2]
            paths[sid] = self._constrained_path(sid, shipper.position, goal, constraints[sid], edge_constraints[sid])

        repair_rounds = 10 if self.horizon <= 6 else 24
        for _ in range(repair_rounds):
            if self.run_deadline and time.time() > self.run_deadline:
                break
            conflict = self._find_conflict(paths)
            if conflict is None:
                break
            kind, t, a, b, pos = conflict
            pa = self._priority(a, shippers, targets, orders, now)
            pb = self._priority(b, shippers, targets, orders, now)
            loser = b if pa >= pb else a
            prev = paths[loser][min(t - 1, len(paths[loser]) - 1)]
            if kind == "vertex":
                constraints[loser].add((t, pos))
            else:
                constraints[loser].add((t, pos))
                edge_constraints[loser].add((t, prev, pos))
            target = targets.get(loser)
            goal = shippers[loser].position if target is None else target[2]
            new_path = self._constrained_path(loser, shippers[loser].position, goal, constraints[loser], edge_constraints[loser])
            if new_path == paths[loser]:
                new_path = [shippers[loser].position] + paths[loser][:-1]
            paths[loser] = new_path
            self.repairs += 1
        self.constraints = constraints
        self.edge_constraints = edge_constraints
        return paths

    def _action_from_path(self, shipper: Shipper, path: List[Position], target: Optional[Tuple[str, int, Position]], orders: Dict[int, Order]) -> Action:
        nxt = path[1] if len(path) > 1 else shipper.position
        move = self.bfs.move_between(shipper.position, nxt)
        op = 0
        if target is not None and nxt == target[2]:
            if target[0] == "pickup":
                op = 1
            elif target[0] == "deliver":
                op = 2
        if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
            op = 0
        if op == 2 and not self._deliverable(shipper, orders, nxt):
            op = 0
        return move, op

    def _decide(self, obs: dict) -> Dict[int, Action]:
        if self.env.N <= 10 or self.use_midlarge_route_assignment:
            return self._small_route_decide(obs)
        orders: Dict[int, Order] = obs["orders"]
        now = int(obs["t"])
        shippers: List[Shipper] = list(obs["shippers"])
        actions: Dict[int, Action] = {}
        targets = self._assign_targets(shippers, orders, now)
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                targets.pop(shipper.id, None)
            elif self._pickup_at(shipper, orders, shipper.position) is not None and not shipper.bag:
                actions[shipper.id] = ("S", 1)
                targets.pop(shipper.id, None)
        active_targets = {sid: target for sid, target in targets.items() if sid not in actions}
        active_obs = dict(obs)
        active_obs["shippers"] = [s for s in shippers if s.id not in actions]
        paths = self._plan_paths(active_obs, active_targets)
        self.paths = paths
        for shipper in active_obs["shippers"]:
            actions[shipper.id] = self._action_from_path(shipper, paths.get(shipper.id, [shipper.position]), active_targets.get(shipper.id), orders)
        return resolve_collisions_and_blocks(
            shippers,
            actions,
            self.env.grid,
            orders,
            self._pickup_at,
            self._deliverable,
            {sid: target[2] for sid, target in active_targets.items()},
            self.bfs.dist,
            allow_unblock=self.env.N <= 12,
        )

    def run(self) -> dict:
        start = time.time()
        self.run_deadline = start + max(20.0, min(110.0, 0.12 * self.env.T + 9.0 * self.env.C))
        obs = self.env.reset()
        while not obs.get("done", False):
            if time.time() > self.run_deadline:
                actions = {s.id: ("S", 2) for s in obs["shippers"]}
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
