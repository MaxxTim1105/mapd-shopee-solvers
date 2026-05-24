from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

from env import Order, Shipper, delivery_reward

Position = Tuple[int, int]
DistanceFn = Callable[[Position, Position], int]
CanTakeFn = Callable[[Shipper, Order, Dict[int, Order]], bool]

INF = 10**8

def is_pickup_safe(
    shipper: Shipper,
    pickup_order: Order,
    orders: Dict[int, Order],
    now: int,
    start_pos: Position,
    dist_fn: DistanceFn,
) -> bool:
    """Kiểm tra xem việc đi đường vòng để lấy thêm đơn hàng mới có làm các đơn hàng

    hiện có trong túi bị trễ hạn hoặc bị chậm trễ quá mức cho phép hay không.
    """
    if not shipper.bag:
        return True
    
    direct_times = {}
    for oid in shipper.bag:
        o = orders[oid]
        d = dist_fn(shipper.position, (o.ex, o.ey))
        direct_times[oid] = now + d
        
    temp_bag = list(shipper.bag) + [pickup_order.id]
    curr_pos = start_pos
    curr_time = now
    if curr_pos != shipper.position:
        curr_time += dist_fn(shipper.position, curr_pos)
        
    sim_times = {}
    while temp_bag:
        best_oid = None
        best_key = (INF, 0, 0, 0)
        for oid in temp_bag:
            o = orders[oid]
            d = dist_fn(curr_pos, (o.ex, o.ey))
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
            # Nếu đơn hàng vốn kịp giờ nay lại bị trễ -> Reject
            if direct_times[oid] <= o.et:
                return False
            # Nếu đơn hàng đã trễ sẵn, không cho phép bị chậm thêm quá 10 bước
            if sim_times[oid] - direct_times[oid] > 10:
                return False
    return True


def evaluate_detour_net_reward(
    shipper: Shipper,
    pickup_order: Order,
    orders: Dict[int, Order],
    now: int,
    dist_fn: DistanceFn,
    T: int,
) -> float:
    """Tính toán và so sánh hiệu số lợi ích thực tế ròng (phần thưởng giao hàng trừ đi

    chi phí nhiên liệu di chuyển dựa trên trọng tải) giữa việc đi giao trực tiếp (Phương án A)
    và đi vòng nhận thêm đơn rồi mới giao toàn bộ (Phương án B).
    """
    # Phương án A: Đi giao trực tiếp túi đồ hiện tại
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
            d = dist_fn(curr_pos, (o.ex, o.ey))
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
        r_direct += delivery_reward(orders[best_oid], curr_time, T)
        
        curr_pos = (orders[best_oid].ex, orders[best_oid].ey)
        temp_bag_direct.remove(best_oid)
        
    # Phương án B: Đi vòng lấy thêm đơn hàng mới trước, sau đó giao tất cả
    temp_bag_detour = list(shipper.bag) + [pickup_order.id]
    curr_pos = shipper.position
    curr_time = now
    r_detour = 0.0
    c_detour = 0.0
    
    d_to_pickup = dist_fn(shipper.position, (pickup_order.sx, pickup_order.sy))
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
            d = dist_fn(curr_pos, (o.ex, o.ey))
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
        r_detour += delivery_reward(orders[best_oid], curr_time, T)
        
        curr_pos = (orders[best_oid].ex, orders[best_oid].ey)
        temp_bag_detour.remove(best_oid)
        
    net_direct = r_direct + c_direct
    net_detour = r_detour + c_detour
    return net_detour - net_direct


def find_best_detour_target(
    shipper: Shipper,
    orders: Dict[int, Order],
    now: int,
    reserved: Set[int],
    shippers: List[Shipper],
    dist_fn: DistanceFn,
    T: int,
    can_take_fn: CanTakeFn,
) -> Optional[Order]:
    """Tìm kiếm đơn hàng tốt nhất để đi vòng tới lấy dựa trên các kiểm tra an toàn,

    so sánh lợi ích thực tế ròng và quy tắc nhường đơn cho các shipper trống túi khác gần hơn.
    """
    best_order = None
    best_delta = 0.0
    min_threshold = 0.5
    margin = 3
    
    empty_shippers = [s for s in shippers if not s.bag and s.id != shipper.id]
    
    for order in orders.values():
        if order.id in reserved or not can_take_fn(shipper, order, orders):
            continue
            
        pickup_pos = (order.sx, order.sy)
        d_to_pickup = dist_fn(shipper.position, pickup_pos)
        if d_to_pickup >= INF:
            continue
            
        # Cơ chế nhường đơn: Nếu một shipper trống túi ở gần hoặc gần ngang ngửa chúng ta, nhường đơn!
        deferred = False
        for empty_s in empty_shippers:
            d_empty = dist_fn(empty_s.position, pickup_pos)
            if d_empty < d_to_pickup + margin:
                deferred = True
                break
        if deferred:
            continue
            
        # Kiểm tra an toàn tiến trình
        if not is_pickup_safe(shipper, order, orders, now, pickup_pos, dist_fn):
            continue
            
        # Đánh giá lợi nhuận ròng
        delta = evaluate_detour_net_reward(shipper, order, orders, now, dist_fn, T)
        if delta > best_delta and delta >= min_threshold:
            best_delta = delta
            best_order = order
            
    return best_order
