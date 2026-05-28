from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.detour_utils import (
    evaluate_detour_net_reward,
    find_best_detour_target,
    is_pickup_safe,
)
from solvers.solver import Solver


from solvers.heuristic_utils import get_aco_params, estimate_average_shortest_path


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")



class BFS:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.sssp_cache: Dict[Position, Dict[Position, int]] = {}
        self.dist_cache: Dict[Tuple[Position, Position], int] = {}
        self.move_cache: Dict[Tuple[Position, Position], Move] = {}

    def neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def get_all_dists(self, start: Position) -> Dict[Position, int]:
        if start in self.sssp_cache:
            return self.sssp_cache[start]
        if not is_valid_cell(start, self.grid):
            return {}
        q = deque([start])
        dist = {start: 0}
        while q:
            pos = q.popleft()
            d = dist[pos]
            for _, nxt in self.neighbors(pos):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        self.sssp_cache[start] = dist
        return dist

    def dist(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        if start in self.sssp_cache:
            return self.sssp_cache[start].get(goal, INF)
        if goal in self.sssp_cache:
            return self.sssp_cache[goal].get(start, INF)
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF
        dist_map = self.get_all_dists(start)
        return dist_map.get(goal, INF)

    def next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        if key in self.move_cache:
            return self.move_cache[key]
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            self.move_cache[key] = "S"
            return "S"
        
        dist_map = self.get_all_dists(goal)
        start_d = dist_map.get(start, INF)
        if start_d >= INF:
            self.move_cache[key] = "S"
            return "S"
            
        best_move = "S"
        best_d = start_d
        for move in MOVES:
            nxt = valid_next_pos(start, move, self.grid)
            if nxt != start:
                nd = dist_map.get(nxt, INF)
                if nd < best_d:
                    best_d = nd
                    best_move = move
                    
        self.move_cache[key] = best_move
        return best_move

    def after(self, pos: Position, move: Move) -> Position:
        return valid_next_pos(pos, move, self.grid)


class ACOSolver(Solver):
    """Online ant colony optimizer over short pickup-delivery routes."""

    method_name = "ACOSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs: Optional[BFS] = None
        self.rng = random.Random(0)
        self.pheromone: Dict[Tuple[Position, Position], float] = defaultdict(lambda: 1.0)
        self.pickup_heat: Dict[Position, float] = defaultdict(float)
        self.targets: Dict[int, Tuple[str, int, Position]] = {}
        self.run_deadline = 0.0
        self.grid: List[List[int]] = [[0]]
        self.N = 1
        self.C = 1
        self.T = 1
        self._configure_by_observation(1, 1, 1, [[0]])

    def _configure_by_observation(self, n: int, c: int, t: int, grid: List[List[int]]) -> None:
        self.N = n
        self.C = c
        self.T = t
        self.grid = grid
        self.bfs = BFS(grid)
        self.shipper_commitments: Dict[int, int] = {}
        self.asd = estimate_average_shortest_path(grid, self.bfs)
        self.d1_penalty, self.d2_penalty, self.late_penalty = get_aco_params(n, c, t, grid, self.bfs)

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

    def _update_heat(self, obs: dict) -> None:
        for oid in obs.get("new_order_ids", []):
            order = obs["orders"].get(oid)
            if order is not None:
                self.pickup_heat[(order.sx, order.sy)] += 1.0 + 0.5 * order.p

    def _heuristic(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int, pos: Optional[Position] = None) -> float:
        start = shipper.position if pos is None else pos
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(start, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return 0.0
        finish = now + d1 + d2
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        heat = self.pickup_heat.get(pickup, 0.0)
        cluster = sum(1 for o in orders.values() if not o.picked and abs(o.sx - order.sx) + abs(o.sy - order.sy) <= 3)
        return max(
            0.001,
            reward
            + 2.5 * order.p
            + heat
            + cluster
            - self.d1_penalty * d1
            - self.d2_penalty * d2
            - self.late_penalty * late,
        )

    def _candidate_pool(self, shipper: Shipper, orders: Dict[int, Order], now: int, limit: int = 24) -> List[Order]:
        scored = []
        max_dist = 22 if self.N > 30 else 15
        for order in orders.values():
            if abs(shipper.r - order.sx) + abs(shipper.c - order.sy) > max_dist:
                continue
            if not self._can_take(shipper, order, orders):
                continue
            h = self._heuristic(shipper, order, orders, now)
            if h > 0.001:
                scored.append((h, order.et, order.id, order))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored[:limit]]

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        carried = [
            order
            for order in self._carried(shipper, orders)
            if self.bfs.dist(shipper.position, (order.ex, order.ey)) < INF
        ]
        if not carried:
            return None
        return max(
            carried,
            key=lambda o: delivery_reward(o, now + self.bfs.dist(shipper.position, (o.ex, o.ey)), self.T)
            - 0.2 * self.bfs.dist(shipper.position, (o.ex, o.ey)),
        )

    def _construct_ant(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Tuple[float, Dict[int, Tuple[str, int, Position]], List[Tuple[Position, Position]]]:
        used: set[int] = set()
        targets: Dict[int, Tuple[str, int, Position]] = {}
        edges: List[Tuple[Position, Position]] = []
        total = 0.0
        shuffled = list(shippers)
        self.rng.shuffle(shuffled)
        for shipper in shuffled:
            if shipper.bag:
                # 1. Kiểm tra cam kết đi vòng hiện tại trước
                committed_order = None
                if shipper.id in self.shipper_commitments:
                    committed_oid = self.shipper_commitments[shipper.id]
                    if committed_oid in orders:
                        committed_order = orders[committed_oid]
                
                if committed_order is not None:
                    pickup_pos = (committed_order.sx, committed_order.sy)
                    if is_pickup_safe(shipper, committed_order, orders, now, pickup_pos, self.bfs.dist):
                        targets[shipper.id] = ("pickup", committed_order.id, pickup_pos)
                        used.add(committed_order.id)
                        pickup = pickup_pos
                        drop = (committed_order.ex, committed_order.ey)
                        edges.append((pickup, drop))
                        total += self._heuristic(shipper, committed_order, orders, now)
                        continue
                
                # 2. Nếu không có cam kết hoặc cam kết không an toàn, tìm đường vòng mới hoặc đi giao hàng
                detour_allowed = self.asd < 7.2 or self.T >= 700
                can_pickup_more = (
                    detour_allowed
                    and len(shipper.bag) < shipper.K_max
                    and self._carried_weight(shipper, orders) < shipper.W_max
                )
                detour = None
                if can_pickup_more:
                    detour = find_best_detour_target(
                        shipper, orders, now, used, shuffled, self.bfs.dist, self.T, self._can_take
                    )
                
                if detour is not None:
                    targets[shipper.id] = ("pickup", detour.id, (detour.sx, detour.sy))
                    used.add(detour.id)
                    pickup = (detour.sx, detour.sy)
                    drop = (detour.ex, detour.ey)
                    edges.append((pickup, drop))
                    total += self._heuristic(shipper, detour, orders, now)
                else:
                    delivery = self._delivery_target(shipper, orders, now)
                    if delivery is not None:
                        targets[shipper.id] = ("deliver", delivery.id, (delivery.ex, delivery.ey))
                        total += delivery_reward(delivery, now + self.bfs.dist(shipper.position, (delivery.ex, delivery.ey)), self.T)
                continue
            candidates = []
            weights = []
            for order in self._candidate_pool(shipper, orders, now):
                if order.id in used:
                    continue
                pickup = (order.sx, order.sy)
                drop = (order.ex, order.ey)
                pher = self.pheromone[(pickup, drop)]
                h = self._heuristic(shipper, order, orders, now)
                desirability = (pher ** 1.2) * (h ** 2.0)
                candidates.append(order)
                weights.append(desirability)
            if not candidates:
                continue
            order = self.rng.choices(candidates, weights=weights, k=1)[0]
            used.add(order.id)
            pickup = (order.sx, order.sy)
            drop = (order.ex, order.ey)
            targets[shipper.id] = ("pickup", order.id, pickup)
            edges.append((pickup, drop))
            total += self._heuristic(shipper, order, orders, now)
        return total, targets, edges

    def _aco_targets(self, shippers: List[Shipper], orders: Dict[int, Order], now: int) -> Dict[int, Tuple[str, int, Position]]:
        for edge in list(self.pheromone):
            self.pheromone[edge] = max(0.1, self.pheromone[edge] * 0.96)
        best_score = -INF
        best_targets: Dict[int, Tuple[str, int, Position]] = {}
        best_edges: List[Tuple[Position, Position]] = []
        ants = max(8, min(18, 3 * max(1, len(shippers))))
        iterations = 3 if self.asd <= 6.0 else 2
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
        for edge in best_edges:
            self.pheromone[edge] = min(20.0, self.pheromone[edge] + reward_scale)
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
        
        # Clean up invalid/completed commitments
        for sid in list(self.shipper_commitments.keys()):
            oid = self.shipper_commitments[sid]
            if oid not in orders or orders[oid].picked or orders[oid].delivered:
                del self.shipper_commitments[sid]
                
        targets = self._aco_targets(shippers, orders, now)
        
        # Save commitments for carrying shippers
        for shipper in shippers:
            if shipper.bag:
                target = targets.get(shipper.id)
                if target is not None and target[0] == "pickup":
                    self.shipper_commitments[shipper.id] = target[1]
                else:
                    self.shipper_commitments.pop(shipper.id, None)
                    
        actions: Dict[int, Action] = {}
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                self.shipper_commitments.pop(shipper.id, None)
                continue
            here = self._pickup_at(shipper, orders, shipper.position)
            detour_allowed = self.asd < 7.2 or self.T >= 700
            if not detour_allowed:
                pickup_allowed = here is not None and not shipper.bag
            else:
                pickup_allowed = here is not None and (
                    not shipper.bag
                    or (
                        is_pickup_safe(shipper, here, orders, now, shipper.position, self.bfs.dist)
                        and evaluate_detour_net_reward(shipper, here, orders, now, self.bfs.dist, self.T) > 0.0
                    )
                )
            if pickup_allowed:
                actions[shipper.id] = ("S", 1)
                self.shipper_commitments.pop(shipper.id, None)
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
                if not hasattr(self, "fallback_solver") or self.fallback_solver is None:
                    from solvers.greedy_bfs import GreedyBFS
                    self.fallback_solver = GreedyBFS(self.env)
                    self.fallback_solver._configure_by_observation(self.N, self.C, self.T, self.grid)
                actions = self.fallback_solver._decide(obs)
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
