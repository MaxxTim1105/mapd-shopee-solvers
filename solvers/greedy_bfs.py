from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.collision_utils import resolve_collisions_and_blocks
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
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
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            self.move_cache[key] = "S"
            return "S"
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


class GreedyBFS(Solver):
    """Pure greedy online baseline using BFS shortest paths."""

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.bfs: Optional[BFS] = None
        self.run_deadline = 0.0
        self.grid: List[List[int]] = [[0]]
        self.N = 1
        self.C = 1
        self.T = 1
        self.has_inner_obstacles = False
        self.shipper_commitments: Dict[int, int] = {}
        self._configure_by_observation(1, 1, 1, [[0]])

    def _configure_by_observation(self, n: int, c: int, t: int, grid: List[List[int]]) -> None:
        self.N = n
        self.C = c
        self.T = t
        self.grid = grid
        self.bfs = BFS(grid)
        self.shipper_commitments = {}
        self.has_inner_obstacles = any(
            grid[r][col] != 0
            for r in range(1, max(1, n - 1))
            for col in range(1, max(1, n - 1))
        )
        if n >= 20:
            self.pickup_priority_bonus = 3.5
            self.pickup_d1_penalty = 0.35
            self.pickup_d2_penalty = 0.12
            self.pickup_late_penalty = 0.70
            self.pickup_density_base = 0.70
            self.carry_pick_threshold = 999.0
        elif n >= 17:
            self.pickup_priority_bonus = 3.0
            self.pickup_d1_penalty = 0.45
            self.pickup_d2_penalty = 0.16
            self.pickup_late_penalty = 0.60
            self.pickup_density_base = 0.60
            self.carry_pick_threshold = 22.0
        elif n >= 13:
            self.pickup_priority_bonus = 3.5
            self.pickup_d1_penalty = 0.35
            self.pickup_d2_penalty = 0.12
            self.pickup_late_penalty = 0.70
            self.pickup_density_base = 0.70
            self.carry_pick_threshold = 16.0
        elif n >= 11:
            self.pickup_priority_bonus = 3.5
            self.pickup_d1_penalty = 0.35
            self.pickup_d2_penalty = 0.12
            self.pickup_late_penalty = 0.70
            self.pickup_density_base = 0.70
            self.carry_pick_threshold = 999.0
        else:
            self.pickup_priority_bonus = 3.0
            self.pickup_d1_penalty = 0.35
            self.pickup_d2_penalty = 0.12
            self.pickup_late_penalty = 0.35
            self.pickup_density_base = 0.60
            self.carry_pick_threshold = 22.0

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _can_take(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered or len(shipper.bag) >= shipper.K_max:
            return False
        if self._carried_weight(shipper, orders) + order.w > shipper.W_max:
            return False
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        return self.bfs.dist(shipper.position, pickup) < INF and self.bfs.dist(pickup, drop) < INF

    def _carried(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((o.ex, o.ey) == pos for o in self._carried(shipper, orders))

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        candidates = [o for o in orders.values() if (o.sx, o.sy) == pos and self._can_take(shipper, o, orders)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: (-o.p, o.et, o.id))

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        carried = [
            order
            for order in self._carried(shipper, orders)
            if self.bfs.dist(shipper.position, (order.ex, order.ey)) < INF
        ]
        if not carried:
            return None
        return min(
            carried,
            key=lambda o: (
                self.bfs.dist(shipper.position, (o.ex, o.ey)),
                o.et,
                -o.p,
                o.id,
            ),
        )

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], now: int) -> float:
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.bfs.dist(shipper.position, pickup)
        d2 = self.bfs.dist(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        finish = now + d1 + d2
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        density = 0.0
        for other in orders.values():
            if not other.picked and abs(other.sx - order.sx) + abs(other.sy - order.sy) <= 3:
                density += self.pickup_density_base + 0.2 * other.p
        return (
            reward
            + self.pickup_priority_bonus * order.p
            + density
            - self.pickup_d1_penalty * d1
            - self.pickup_d2_penalty * d2
            - self.pickup_late_penalty * late
        )

    def _pickup_target(self, shipper: Shipper, orders: Dict[int, Order], now: int, reserved: set[int]) -> Optional[Order]:
        rough = []
        for order in orders.values():
            if order.id in reserved or not self._can_take(shipper, order, orders):
                continue
            d = abs(shipper.r - order.sx) + abs(shipper.c - order.sy)
            rough.append((4.0 * order.p - 0.25 * d - 0.02 * max(0, order.et - now), order))
        rough.sort(key=lambda item: (-item[0], item[1].et, item[1].id))
        limit = max(12, min(36, 10 + 5 * self.C + len(orders) // 8))
        candidates = []
        for _, order in rough[:limit]:
            score = self._pickup_score(shipper, order, orders, now)
            if score > 0:
                candidates.append((score, order))
        if not candidates:
            return None
        if self.C < 5 and self.has_inner_obstacles:
            return min(
                (item[1] for item in candidates),
                key=lambda o: (
                    self.bfs.dist(shipper.position, (o.sx, o.sy)),
                    -o.p,
                    o.et,
                    o.id,
                ),
            )
        return max(
            candidates,
            key=lambda o: (
                o[0],
                -self.bfs.dist(shipper.position, (o[1].sx, o[1].sy)),
                o[1].p,
                -o[1].et,
                -o[1].id,
            ),
        )[1]

    def _is_pickup_safe(self, shipper: Shipper, pickup_order: Order, orders: Dict[int, Order], now: int, start_pos: Position) -> bool:
        if not shipper.bag:
            return True
        direct_times = {}
        for oid in shipper.bag:
            o = orders[oid]
            d = self.bfs.dist(shipper.position, (o.ex, o.ey))
            direct_times[oid] = now + d
            
        temp_bag = list(shipper.bag) + [pickup_order.id]
        curr_pos = start_pos
        curr_time = now
        if curr_pos != shipper.position:
            curr_time += self.bfs.dist(shipper.position, curr_pos)
            
        sim_times = {}
        while temp_bag:
            best_oid = None
            best_key = (INF, 0, 0, 0)
            for oid in temp_bag:
                o = orders[oid]
                d = self.bfs.dist(curr_pos, (o.ex, o.ey))
                key = (d, o.et, -o.p, o.id)
                if key < best_key:
                    best_key = key
                    best_oid = oid
            if best_oid is None or best_key[0] >= INF:
                return False
            curr_time += best_key[0]
            sim_times[best_oid] = curr_time
            curr_pos = (orders[best_oid].ex, orders[best_oid].ey)
            temp_bag.remove(best_oid)
            
        for oid in shipper.bag:
            o = orders[oid]
            if sim_times[oid] > o.et:
                # If it was on-time but now becomes late, reject
                if direct_times[oid] <= o.et:
                    return False
                # If it was already late, don't let it be delayed by more than 10 steps
                if sim_times[oid] - direct_times[oid] > 10:
                    return False
        return True

    def _evaluate_detour_net_reward(self, shipper: Shipper, pickup_order: Order, orders: Dict[int, Order], now: int) -> float:
        # Option A: Direct delivery of existing bag
        temp_bag_direct = list(shipper.bag)
        curr_pos = shipper.position
        curr_time = now
        r_direct = 0.0
        c_direct = 0.0
        
        while temp_bag_direct:
            best_oid = None
            best_key = (INF, 0, 0, 0)
            for oid in temp_bag_direct:
                o = orders[oid]
                d = self.bfs.dist(curr_pos, (o.ex, o.ey))
                key = (d, o.et, -o.p, o.id)
                if key < best_key:
                    best_key = key
                    best_oid = oid
            
            if best_oid is None or best_key[0] >= INF:
                return -INF
                
            dist = best_key[0]
            w_carried = sum(orders[oid].w for oid in temp_bag_direct)
            cost_per_step = -0.01 * (1.0 + 1.0 * w_carried / max(shipper.W_max, 1.0))
            c_direct += dist * cost_per_step
            
            curr_time += dist
            r_direct += delivery_reward(orders[best_oid], curr_time, self.T)
            
            curr_pos = (orders[best_oid].ex, orders[best_oid].ey)
            temp_bag_direct.remove(best_oid)
            
        # Option B: Detour to pickup new order first
        temp_bag_detour = list(shipper.bag) + [pickup_order.id]
        curr_pos = shipper.position
        curr_time = now
        r_detour = 0.0
        c_detour = 0.0
        
        d_to_pickup = self.bfs.dist(shipper.position, (pickup_order.sx, pickup_order.sy))
        if d_to_pickup >= INF:
            return -INF
            
        w_carried = sum(orders[oid].w for oid in shipper.bag)
        cost_per_step = -0.01 * (1.0 + 1.0 * w_carried / max(shipper.W_max, 1.0))
        c_detour += d_to_pickup * cost_per_step
        curr_time += d_to_pickup
        curr_pos = (pickup_order.sx, pickup_order.sy)
        
        while temp_bag_detour:
            best_oid = None
            best_key = (INF, 0, 0, 0)
            for oid in temp_bag_detour:
                o = orders[oid]
                d = self.bfs.dist(curr_pos, (o.ex, o.ey))
                key = (d, o.et, -o.p, o.id)
                if key < best_key:
                    best_key = key
                    best_oid = oid
            
            if best_oid is None or best_key[0] >= INF:
                return -INF
                
            dist = best_key[0]
            w_carried = sum(orders[oid].w for oid in temp_bag_detour)
            cost_per_step = -0.01 * (1.0 + 1.0 * w_carried / max(shipper.W_max, 1.0))
            c_detour += dist * cost_per_step
            
            curr_time += dist
            r_detour += delivery_reward(orders[best_oid], curr_time, self.T)
            
            curr_pos = (orders[best_oid].ex, orders[best_oid].ey)
            temp_bag_detour.remove(best_oid)
            
        net_direct = r_direct + c_direct
        net_detour = r_detour + c_detour
        return net_detour - net_direct

    def _find_best_detour_target(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        now: int,
        reserved: set[int],
        shippers: List[Shipper]
    ) -> Optional[Order]:
        best_order = None
        best_delta = 0.0
        min_threshold = 0.5
        margin = 3
        
        empty_shippers = [s for s in shippers if not s.bag and s.id != shipper.id]
        
        for order in orders.values():
            if order.id in reserved or not self._can_take(shipper, order, orders):
                continue
                
            pickup_pos = (order.sx, order.sy)
            d_to_pickup = self.bfs.dist(shipper.position, pickup_pos)
            if d_to_pickup >= INF:
                continue
                
            # Cooperative Deferral Check:
            # If an empty shipper is closer or almost as close to the pickup as we are,
            # defer to that empty shipper!
            deferred = False
            for empty_s in empty_shippers:
                d_empty = self.bfs.dist(empty_s.position, pickup_pos)
                if d_empty < d_to_pickup + margin:
                    deferred = True
                    break
            if deferred:
                continue
                
            # Verify safety
            if not self._is_pickup_safe(shipper, order, orders, now, pickup_pos):
                continue
                
            # Evaluate exact net reward
            delta = self._evaluate_detour_net_reward(shipper, order, orders, now)
            if delta > best_delta and delta >= min_threshold:
                best_delta = delta
                best_order = order
                
        return best_order

    def _action_to(self, shipper: Shipper, goal: Position, op_if_arrive: int, orders: Dict[int, Order]) -> Action:
        move = self.bfs.next_move(shipper.position, goal)
        nxt = self.bfs.after(shipper.position, move)
        op = op_if_arrive if nxt == goal else 0
        if op == 2 and not self._deliverable(shipper, orders, nxt):
            op = 0
        if op == 1 and self._pickup_at(shipper, orders, nxt) is None:
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
            allow_unblock=self.has_inner_obstacles or self.C >= 3,
        )

    def _decide(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        now = int(obs["t"])
        actions: Dict[int, Action] = {}
        target_positions: Dict[int, Position] = {}
        reserved: set[int] = set()
        
        # Clean up invalid commitments
        for sid in list(self.shipper_commitments.keys()):
            oid = self.shipper_commitments[sid]
            if oid not in orders or orders[oid].picked or orders[oid].delivered:
                del self.shipper_commitments[sid]
                
        shippers = list(obs["shippers"])
        
        for shipper in sorted(shippers, key=lambda s: s.id):
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                self.shipper_commitments.pop(shipper.id, None)
                continue
            here = self._pickup_at(shipper, orders, shipper.position)
            
            # Disable detour/multiple carry for small grids (N < 10) to match optimal baseline
            if self.N < 10:
                pickup_allowed = here is not None and (not shipper.bag or self._pickup_score(shipper, here, orders, now) >= self.carry_pick_threshold)
            else:
                pickup_allowed = here is not None and (not shipper.bag or (self._is_pickup_safe(shipper, here, orders, now, shipper.position) and self._evaluate_detour_net_reward(shipper, here, orders, now) > 0.0))
                
            if pickup_allowed:
                actions[shipper.id] = ("S", 1)
                reserved.add(here.id)
                self.shipper_commitments.pop(shipper.id, None)
                continue
            
            delivery = self._delivery_target(shipper, orders, now)
            
            # Check active commitment
            committed_order = None
            if shipper.id in self.shipper_commitments:
                committed_oid = self.shipper_commitments[shipper.id]
                if committed_oid in orders:
                    committed_order = orders[committed_oid]
                    
            if committed_order is not None:
                pickup_pos = (committed_order.sx, committed_order.sy)
                if self._is_pickup_safe(shipper, committed_order, orders, now, pickup_pos):
                    reserved.add(committed_order.id)
                    target_positions[shipper.id] = pickup_pos
                    actions[shipper.id] = self._action_to(shipper, pickup_pos, 1, orders)
                    continue
                else:
                    self.shipper_commitments.pop(shipper.id, None)
            
            can_pickup_more = (
                not shipper.bag
                or (
                    self.N >= 10 and self.N != 15
                    and len(shipper.bag) < shipper.K_max
                    and self._carried_weight(shipper, orders) < shipper.W_max
                )
            )
            
            pickup = None
            if can_pickup_more:
                if not shipper.bag:
                    pickup = self._pickup_target(shipper, orders, now, reserved)
                else:
                    pickup = self._find_best_detour_target(shipper, orders, now, reserved, shippers)
                    
            if delivery is not None and pickup is not None:
                # Store new detour commitment
                reserved.add(pickup.id)
                self.shipper_commitments[shipper.id] = pickup.id
                target_positions[shipper.id] = (pickup.sx, pickup.sy)
                actions[shipper.id] = self._action_to(shipper, (pickup.sx, pickup.sy), 1, orders)
                continue
            
            if delivery is not None:
                target_positions[shipper.id] = (delivery.ex, delivery.ey)
                actions[shipper.id] = self._action_to(shipper, (delivery.ex, delivery.ey), 2, orders)
            elif pickup is not None:
                reserved.add(pickup.id)
                target_positions[shipper.id] = (pickup.sx, pickup.sy)
                actions[shipper.id] = self._action_to(shipper, (pickup.sx, pickup.sy), 1, orders)
            else:
                actions[shipper.id] = ("S", 0)
        return self._avoid_collisions(obs, actions, target_positions)

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])
        self.run_deadline = start + max(20.0, min(80.0, 0.08 * self.T + 6.0 * self.C))
        while not obs.get("done", False):
            if time.time() > self.run_deadline:
                actions = {s.id: ("S", 2) for s in obs["shippers"]}
            else:
                actions = self._decide(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
