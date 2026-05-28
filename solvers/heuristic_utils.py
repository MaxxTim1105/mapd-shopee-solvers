from __future__ import annotations

import random
from typing import Dict, Tuple, Any

INF = 10**8

def estimate_average_shortest_path(grid: list[list[int]], bfs, sample_size: int = 40) -> float:
    """
    Estimates the Average Shortest Distance (ASD) between all free cells of the grid
    by sampling random pairs. This dynamically captures map diameter and bottleneck complexity.
    """
    free_cells = [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0]
    if len(free_cells) < 2:
        return 1.0
    
    # Stable seed for deterministic and reproducible evaluations
    rng = random.Random(42)
    total_dist = 0
    valid_pairs = 0
    
    for _ in range(sample_size):
        u = rng.choice(free_cells)
        v = rng.choice(free_cells)
        if u != v:
            d = bfs.dist(u, v)
            if d < INF:
                total_dist += d
                valid_pairs += 1
                
    if valid_pairs == 0:
        return float(len(grid))
    return total_dist / valid_pairs


def interpolate_multi(x: float, points: list[tuple[float, float]]) -> float:
    """
    Performs continuous multi-point linear interpolation for x on the given reference points.
    points must be a list of (x_val, y_val) sorted by x_val.
    """
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
        
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        if x1 <= x <= x2:
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            
    return points[-1][1]


def get_aco_params(n: int, c: int, t: int, grid: list[list[int]], bfs) -> Tuple[float, float, float]:
    """Computes dynamic d1_penalty, d2_penalty, and late_penalty for ACO Solver."""
    asd = estimate_average_shortest_path(grid, bfs)
    
    # Continuous interpolation across Small, Medium, Large maps
    d1 = interpolate_multi(asd, [
        (3.5, 0.20), (6.6, 0.20), (7.9, 0.40), (9.6, 0.30), (12.5, 0.55), (16.5, 0.65)
    ])
    d2 = interpolate_multi(asd, [
        (3.5, 0.05), (6.6, 0.05), (7.9, 0.15), (9.6, 0.10), (12.5, 0.22), (16.5, 0.18)
    ])
    late = interpolate_multi(asd, [
        (3.5, 0.50), (6.6, 0.50), (7.9, 1.00), (9.6, 0.80), (12.5, 1.35), (16.5, 1.20)
    ])
    
    # Scale down smoothly for extremely large hidden maps to prevent paralysis
    if asd > 16.5:
        scale = 16.5 / asd
        d1 *= scale
        d2 *= scale
        late *= scale
        
    return d1, d2, late


def get_greedy_params(n: int, c: int, t: int, grid: list[list[int]], bfs) -> Dict[str, float]:
    """Computes dynamic parameters for GreedyBFS solver."""
    asd = estimate_average_shortest_path(grid, bfs)
    
    pickup_priority_bonus = interpolate_multi(asd, [
        (3.5, 3.0), (6.6, 3.0), (7.9, 3.5), (12.5, 3.0), (16.5, 3.5)
    ])
    pickup_d1_penalty = interpolate_multi(asd, [
        (3.5, 0.35), (12.5, 0.45), (16.5, 0.35)
    ])
    pickup_d2_penalty = interpolate_multi(asd, [
        (3.5, 0.12), (12.5, 0.16), (16.5, 0.12)
    ])
    pickup_late_penalty = interpolate_multi(asd, [
        (3.5, 0.35), (6.6, 0.35), (7.9, 0.70), (12.5, 0.60), (16.5, 0.70)
    ])
    pickup_density_base = interpolate_multi(asd, [
        (3.5, 0.60), (6.6, 0.60), (7.9, 0.70), (12.5, 0.60), (16.5, 0.70)
    ])
    carry_pick_threshold = interpolate_multi(asd, [
        (3.5, 22.0), (4.9, 22.0), (5.0, 999.0)
    ])
    
    # Scale down smoothly for extremely large hidden maps
    if asd > 16.5:
        scale = 16.5 / asd
        pickup_d1_penalty *= scale
        pickup_d2_penalty *= scale
        pickup_late_penalty *= scale
        
    return {
        "pickup_priority_bonus": pickup_priority_bonus,
        "pickup_d1_penalty": pickup_d1_penalty,
        "pickup_d2_penalty": pickup_d2_penalty,
        "pickup_late_penalty": pickup_late_penalty,
        "pickup_density_base": pickup_density_base,
        "carry_pick_threshold": carry_pick_threshold,
    }


def get_vrp_params(n: int, c: int, t: int, grid: list[list[int]], bfs) -> Tuple[float, float]:
    """Computes dynamic travel_penalty and late_penalty for VRPOrToolsSolver."""
    asd = estimate_average_shortest_path(grid, bfs)
    
    if c <= 2:
        return 0.012, 0.20
        
    travel = interpolate_multi(asd, [
        (7.85, 0.04), (9.6, 0.06), (12.5, 0.075), (16.5, 0.135)
    ])
    late = interpolate_multi(asd, [
        (7.85, 0.30), (9.6, 0.35), (12.5, 0.38), (16.5, 0.48)
    ])
    
    # Scale down smoothly for extremely large hidden maps
    if asd > 16.5:
        scale = 16.5 / asd
        travel *= scale
        late *= scale
        
    return travel, late


def get_cbs_params(n: int, c: int, t: int, grid: list[list[int]], bfs) -> Dict[str, Any]:
    """Computes dynamic horizon and penalties for MAPDCBSSolver."""
    asd = estimate_average_shortest_path(grid, bfs)
    
    horizon = int(max(8, min(18, 5 + c + asd // 1.2)))
    route_travel_penalty = 0.05 + min(0.08, 0.02 * max(0, c - 2))
    route_late_penalty = 0.42 + min(0.35, 0.012 * asd)
    
    # Scale down route travel and late penalties if map is huge
    if asd > 16.5:
        scale = 16.5 / asd
        route_travel_penalty *= scale
        route_late_penalty *= scale
        
    return {
        "horizon": horizon,
        "route_travel_penalty": route_travel_penalty,
        "route_late_penalty": route_late_penalty,
    }
