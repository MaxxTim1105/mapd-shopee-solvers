from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from env import Order, Shipper, valid_next_pos


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
INF = 10**8

PickupAt = Callable[[Shipper, Dict[int, Order], Position], Optional[Order]]
Deliverable = Callable[[Shipper, Dict[int, Order], Position], bool]
Distance = Callable[[Position, Position], int]


def _complete_actions(shippers: List[Shipper], actions: Dict[int, Action]) -> Dict[int, Action]:
    return {shipper.id: actions.get(shipper.id, ("S", 0)) for shipper in shippers}


def _simulate_env_positions(
    shippers: List[Shipper],
    actions: Dict[int, Action],
    grid: List[List[int]],
) -> Tuple[Dict[int, Position], Dict[int, Position]]:
    old = {shipper.id: shipper.position for shipper in shippers}
    desired = {
        shipper.id: valid_next_pos(shipper.position, actions.get(shipper.id, ("S", 0))[0], grid)
        for shipper in shippers
    }
    occupied = set(old.values())
    actual: Dict[int, Position] = {}

    for shipper in sorted(shippers, key=lambda s: s.id):
        old_pos = old[shipper.id]
        target = desired[shipper.id]
        occupied.discard(old_pos)
        if target in occupied:
            target = old_pos
        occupied.add(target)
        actual[shipper.id] = target

    return desired, actual


def _move_for_actual(
    old_pos: Position,
    requested_move: Move,
    actual_pos: Position,
    grid: List[List[int]],
) -> Move:
    if actual_pos == old_pos:
        return "S"
    if valid_next_pos(old_pos, requested_move, grid) == actual_pos:
        return requested_move
    for move in MOVES:
        if valid_next_pos(old_pos, move, grid) == actual_pos:
            return move
    return "S"


def _normalized_actions(
    shippers: List[Shipper],
    actions: Dict[int, Action],
    grid: List[List[int]],
    orders: Dict[int, Order],
    pickup_at: PickupAt,
    deliverable: Deliverable,
) -> Dict[int, Action]:
    _, actual = _simulate_env_positions(shippers, actions, grid)
    fixed: Dict[int, Action] = {}

    for shipper in shippers:
        move, op = actions.get(shipper.id, ("S", 0))
        pos_after_move = actual[shipper.id]
        move = _move_for_actual(shipper.position, move, pos_after_move, grid)

        if op == 1:
            op = 1 if pickup_at(shipper, orders, pos_after_move) is not None else 0
        elif op == 2:
            op = 2 if deliverable(shipper, orders, pos_after_move) else 0
        else:
            op = 0
        fixed[shipper.id] = (move, op)

    return fixed


def _score_simulation(
    shippers: List[Shipper],
    actions: Dict[int, Action],
    grid: List[List[int]],
    target_positions: Optional[Dict[int, Position]],
    distance: Optional[Distance],
) -> Tuple[int, int, int, float]:
    old = {shipper.id: shipper.position for shipper in shippers}
    desired, actual = _simulate_env_positions(shippers, actions, grid)
    intended = [
        shipper.id
        for shipper in shippers
        if actions.get(shipper.id, ("S", 0))[0] != "S" and desired[shipper.id] != old[shipper.id]
    ]
    blocked = sum(1 for sid in intended if actual[sid] == old[sid])
    moved = sum(1 for sid in intended if actual[sid] != old[sid])
    ops = sum(1 for _, op in actions.values() if op in (1, 2))

    progress = 0.0
    if target_positions and distance:
        for sid, target in target_positions.items():
            if sid not in old:
                continue
            before = distance(old[sid], target)
            after = distance(actual.get(sid, old[sid]), target)
            if before < INF and after < INF:
                progress += before - after

    return -blocked, moved, ops, progress


def resolve_collisions_and_blocks(
    shippers: List[Shipper],
    actions: Dict[int, Action],
    grid: List[List[int]],
    orders: Dict[int, Order],
    pickup_at: PickupAt,
    deliverable: Deliverable,
    target_positions: Optional[Dict[int, Position]] = None,
    distance: Optional[Distance] = None,
    allow_unblock: bool = True,
) -> Dict[int, Action]:
    """Repair one-step moves using the same priority rule as DeliveryEnv.

    If the env simulation says movers would be blocked, try a few local
    side-steps/back-steps for blocked agents. This breaks corridor deadlocks
    without changing the solvers' task selection logic.
    """
    shippers = sorted(shippers, key=lambda s: s.id)
    fixed = _complete_actions(shippers, actions)
    if not allow_unblock:
        return _normalized_actions(shippers, fixed, grid, orders, pickup_at, deliverable)

    for _ in range(max(1, len(shippers))):
        old = {shipper.id: shipper.position for shipper in shippers}
        desired, actual = _simulate_env_positions(shippers, fixed, grid)
        blocked = [
            sid
            for sid in sorted(fixed, reverse=True)
            if fixed[sid][0] != "S" and desired[sid] != old[sid] and actual[sid] == old[sid]
        ]
        if not blocked:
            break

        base_score = _score_simulation(shippers, fixed, grid, target_positions, distance)
        best_score = base_score
        best_actions: Optional[Dict[int, Action]] = None

        by_id = {shipper.id: shipper for shipper in shippers}
        for sid in blocked:
            shipper = by_id[sid]
            current_move = fixed[sid][0]
            for move in MOVES:
                if move == current_move:
                    continue
                nxt = valid_next_pos(shipper.position, move, grid)
                if nxt == shipper.position:
                    continue
                trial = dict(fixed)
                trial[sid] = (move, 0)
                _, trial_actual = _simulate_env_positions(shippers, trial, grid)
                if trial_actual[sid] == shipper.position:
                    continue
                score = _score_simulation(shippers, trial, grid, target_positions, distance)
                if score[0] <= base_score[0]:
                    continue
                if score > best_score:
                    best_score = score
                    best_actions = trial

        if best_actions is None:
            break
        fixed = best_actions

    return _normalized_actions(shippers, fixed, grid, orders, pickup_at, deliverable)
