import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import copy
import hashlib
from env import DeliveryEnv

# Ban do ten action de hien thi dep hon tren man hinh
ACTION_NAMES_INT = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "pick",
    6: "drop",
}

# Map chu viet tat di chuyen (kieu tuple) sang ten day du
ACTION_NAMES_CHAR = {
    'S': 'stay',
    'U': 'up',
    'D': 'down',
    'L': 'left',
    'R': 'right',
    'P': 'pick',
    'G': 'drop',  # Give / deliver
}

def _get_action_name(a):
    """
    Ho tro nhieu kieu action ma solver co the tra ve:
      - int:               0, 1, 2 ...
      - str:               'U', 'D', 'stay' ...
      - tuple (dir, oid):  ('D', 0), ('R', 2) — kieu GreedyBFS
      - tuple (int, oid):  (2, 0)
    """
    if isinstance(a, tuple):
        direction, oid = a[0], a[1] if len(a) > 1 else '?'
        if isinstance(direction, str):
            dir_name = ACTION_NAMES_CHAR.get(direction.upper(), direction)
        else:
            dir_name = ACTION_NAMES_INT.get(direction, str(direction))
        # Chi hien thi order_id neu khong phai stay/idle
        if dir_name == 'stay' or oid == 0:
            return dir_name
        return f"{dir_name}(ord={oid})"
    if isinstance(a, int):
        return ACTION_NAMES_INT.get(a, str(a))
    if isinstance(a, str):
        return ACTION_NAMES_CHAR.get(a.upper(), a)
    return str(a)


def visualize_simulation(solver_cls, cfg: dict, base_seed: int = 42, interval: int = 300):
    """
    Truc quan hoa dong tuong tac hoan chinh cho Online MAPD.
    - Fix: raise loi ro rang neu solver khong co _decide (thay vi fallback im lang).
    - Them: log actions ra console moi step.
    - Them: luu actions vao history va hien thi tren panel phai duoi.
    """

    # ------------------------------------------------------------------
    # 1. Kiem tra solver co _decide khong — fail som, fail ro rang
    # ------------------------------------------------------------------
    if not hasattr(solver_cls, '_decide') and not callable(getattr(solver_cls, '_decide', None)):
        # Thu khoi tao de kiem tra instance method
        pass  # se kiem tra sau khi khoi tao

    # ------------------------------------------------------------------
    # 2. Khoi tao moi truong giong Grader
    # ------------------------------------------------------------------
    digest = hashlib.md5(f"{base_seed}:{cfg.get('name', 'unknown')}".encode("utf-8")).hexdigest()
    config_seed = int(digest[:8], 16)

    env_cfg = copy.deepcopy(cfg)
    env = DeliveryEnv(env_cfg, seed=config_seed)
    solver = solver_cls(env)

    # Kiem tra instance method _decide
    if not hasattr(solver, '_decide'):
        available = [m for m in dir(solver) if not m.startswith('__')]
        raise AttributeError(
            f"[visualizer] Solver '{solver_cls.__name__}' khong co method '_decide'.\n"
            f"  Cac method hien co: {available}\n"
            f"  => Tat ca actions se la {{}} (shipper dung yen) — day la nguyen nhan chinh khien cac thuat toan trong giong nhau!\n"
            f"  => Vui long doi ten method thanh '_decide' hoac them wrapper."
        )

    # ------------------------------------------------------------------
    # 3. Thu thap lich su trang thai (Snapshots) + log actions
    # ------------------------------------------------------------------
    history = []
    obs = env.reset()
    history.append({
        "t": 0,
        "shippers": copy.deepcopy(env.shippers),
        "orders":   copy.deepcopy(env.orders),
        "actions":  {},   # frame dau tien chua co action
    })

    if hasattr(solver, '_configure_by_observation'):
        solver._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])

    print(f"\n{'='*60}")
    print(f"[visualizer] Bat dau chay: solver={solver_cls.__name__}  seed={config_seed}")
    print(f"{'='*60}")

    step_count = 0
    idle_streak = {}   # sid -> so buoc lien tiep dung yen

    while not obs.get("done", False):
        actions = solver._decide(obs)

        # --- LOG RA CONSOLE ---
        parts = []
        for sid, a in sorted(actions.items()):
            aname = _get_action_name(a)
            if aname == 'stay':
                idle_streak[sid] = idle_streak.get(sid, 0) + 1
            else:
                idle_streak[sid] = 0
            tag = f"[IDLE x{idle_streak[sid]}!]" if idle_streak.get(sid, 0) >= 10 else ""
            parts.append(f"S{sid}:{aname} {tag}".strip())
        actions_str = "  ".join(parts) if parts else "(khong co action)"
        print(f"  t={obs.get('t', step_count):>4d} | {actions_str}")

        obs, _, done, _ = env.step(actions)
        history.append({
            "t":        int(obs["t"]),
            "shippers": copy.deepcopy(env.shippers),
            "orders":   copy.deepcopy(env.orders),
            "actions":  copy.deepcopy(actions),   # luu action cua buoc nay
        })
        step_count += 1
        if done:
            break

    # Tong ket canh bao idle
    long_idle = {sid: s for sid, s in idle_streak.items() if s >= 20}
    if long_idle:
        print()
        for sid, streak in long_idle.items():
            print(f"  [CANH BAO] S{sid} dung yen {streak} buoc cuoi lien tiep — co the bug trong logic phan cong don hang!")

    delivered_count = sum(1 for o in env.orders.values() if o.delivered)
    print(f"{'='*60}")
    print(f"[visualizer] Ket thuc: {step_count} buoc  |  Don da giao: {delivered_count}/{len(env.orders)}")
    print(f"{'='*60}\n")

    grid        = np.array(env.grid)
    N           = env.N
    total_frames = len(history)

    # ------------------------------------------------------------------
    # 4. Khoi tao giao dien do hoa
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))

    # Khu vuc 1: Ban do mo phong (Trai)
    ax = fig.add_axes([0.05, 0.22, 0.48, 0.70])

    # Khu vuc 2: Dashboard Shippers + Orders (Phai - Tren)
    ax_info = fig.add_axes([0.56, 0.46, 0.41, 0.46])
    ax_info.axis('off')

    # Khu vuc 3: Bang chu giai (Phai - Giua)
    ax_legend = fig.add_axes([0.56, 0.29, 0.41, 0.15])
    ax_legend.axis('off')

    # Khu vuc 4 (MOI): Hien thi actions buoc hien tai (Phai - Duoi)
    ax_actions = fig.add_axes([0.56, 0.22, 0.41, 0.06])
    ax_actions.axis('off')

    cmap   = plt.cm.binary
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63', '#FFEB3B']

    state = {
        "current_idx": 0,
        "is_playing":  True,
        "interval":    interval,
        "active_timer": None,
    }

    # ------------------------------------------------------------------
    # 5. Ham ve tung frame
    # ------------------------------------------------------------------
    def _order_score(order, shippers, orders_dict, t):
        carrier = next((s for s in shippers if order.id in s.bag), None)
        ship = carrier if carrier is not None else (shippers[0] if shippers else None)
        if ship is None:
            return None
        for method_name in ['_heuristic', '_pickup_score', '_completion_score']:
            method = getattr(solver, method_name, None)
            if method is None:
                continue
            try:
                v = method(ship, order, orders_dict, t)
                if v is not None and abs(v) < 10**9:
                    return f"{v:.1f}"
            except Exception:
                continue
        return None

    def draw_frame(frame_idx):
        snapshot        = history[frame_idx]
        current_t       = snapshot["t"]
        current_shippers = snapshot["shippers"]
        current_orders  = snapshot["orders"]
        current_actions = snapshot["actions"]

        # ---- 5a. BAN DO ----
        ax.clear()
        ax.imshow(grid, cmap=cmap, origin='upper')
        ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(
            f"MO PHONG MAPD  |  Thuat toan: {solver_cls.__name__}\n"
            f"Time Step: {current_t} / {env.T}",
            fontsize=11, fontweight='bold'
        )

        for oid, order in current_orders.items():
            if order.appear_t <= current_t:
                if not order.picked:
                    ax.plot(order.sy, order.sx, marker='o', markersize=11, color='green', alpha=0.85)
                    ax.text(order.sy, order.sx, f"S{oid}", color='black', fontsize=7,
                            ha='center', va='center', fontweight='bold')
                if not order.delivered:
                    ax.plot(order.ey, order.ex, marker='X', markersize=11, color='red', alpha=0.85)
                    ax.text(order.ey, order.ex, f"E{oid}", color='white', fontsize=7,
                            ha='center', va='center', fontweight='bold')

        for idx, shipper in enumerate(current_shippers):
            color = colors[idx % len(colors)]
            ax.plot(shipper.c, shipper.r, marker='v', markersize=13,
                    color=color, markeredgecolor='black', zorder=5)
            bag_info = f"S{shipper.id}({len(shipper.bag)})"
            ax.text(shipper.c, shipper.r - 0.38, bag_info, color=color,
                    fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=6)

        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(N - 0.5, -0.5)

        # ---- 5b. ACTIONS PANEL (MOI) ----
        ax_actions.clear()
        ax_actions.axis('off')
        ax_actions.set_xlim(0, 10)
        ax_actions.set_ylim(0, 2)

        ax_actions.text(0, 1.75, "ACTIONS BUOC NAY:", fontsize=8.5,
                        fontweight='bold', color='darkorange')

        if current_actions:
            parts = []
            for sid, a in sorted(current_actions.items()):
                aname = _get_action_name(a)
                parts.append(f"S{sid}:{aname}")
            # Xep toi da 4 shipper moi dong
            lines = [" | ".join(parts[i:i+4]) for i in range(0, len(parts), 4)]
            actions_display = "\n".join(lines)
        else:
            actions_display = "(khong co action / buoc dau tien)"

        ax_actions.text(0, 0.9, actions_display, fontsize=8.5,
                        family='monospace', color='darkorange')

        # ---- 5c. DASHBOARD SHIPPERS ----
        ax_info.clear()
        ax_info.axis('off')
        ax_info.set_xlim(0, 10)
        ax_info.set_ylim(0, 10)

        ax_info.text(0, 9.6, "THONG SO NGUOI GIAO HANG (SHIPPERS)",
                     fontsize=10, fontweight='bold', color='darkgreen')
        headers_shipper = f"{'ID':<4} {'Vi tri':<9} {'So don':<8} {'Trong tai (Kg)':<17} {'Trang thai tai':<12}"
        ax_info.text(0, 9.1, headers_shipper, fontsize=8.5, fontweight='bold', family='monospace')
        ax_info.text(0, 8.9, "-" * 68, fontsize=8.5, family='monospace', color='gray')

        y_offset = 8.4
        for shipper in current_shippers:
            current_weight = sum(
                current_orders[oid].w for oid in shipper.bag if oid in current_orders
            )
            max_weight = getattr(shipper, 'W_max', 20.0)
            max_bag    = getattr(shipper, 'K_max', 5)

            pos_str    = f"({shipper.r},{shipper.c})"
            count_str  = f"{len(shipper.bag)}/{max_bag}"
            weight_str = f"{current_weight:.1f}/{max_weight:.1f}"

            if current_weight > max_weight or len(shipper.bag) > max_bag:
                load_status = "OVERLOADED!"
                text_color  = 'red'
            else:
                load_status = "An toan"
                text_color  = 'black'

            info_line = f"{shipper.id:<4} {pos_str:<9} {count_str:<8} {weight_str:<17} {load_status:<12}"
            ax_info.text(0, y_offset, info_line, fontsize=8.5, family='monospace', color=text_color)
            y_offset -= 0.45

        # ---- 5d. DASHBOARD ORDERS ----
        y_offset -= 0.1
        ax_info.text(0, y_offset, "THONG SO DON HANG (ORDERS)",
                     fontsize=10, fontweight='bold', color='darkred')
        y_offset -= 0.5

        headers_order = f"{'ID':<4} {'Nang':<8} {'Uu tien':<8} {'Xuat hien':<10} {'Han cuoi':<9} {'Diem':<7} {'Trang thai':<13}"
        ax_info.text(0, y_offset, headers_order, fontsize=8.5, fontweight='bold', family='monospace')
        y_offset -= 0.2
        ax_info.text(0, y_offset, "-" * 65, fontsize=8.5, family='monospace', color='gray')
        y_offset -= 0.4

        visible_orders = [o for o in current_orders.values() if o.appear_t <= current_t and not o.delivered]

        for order in visible_orders[-6:]:
            w_val = getattr(order, 'w', 0.0)
            p_val = getattr(order, 'p', 1)

            status     = "Cho lay"
            text_color = 'dimgray'

            if order.delivered:
                status     = "Da giao xong"
                text_color = 'green'
            elif order.picked:
                actual_shipper_id = "?"
                for s in current_shippers:
                    if order.id in s.bag:
                        actual_shipper_id = s.id
                        break
                if actual_shipper_id != "?":
                    status = f"Dang tho (S{actual_shipper_id})"
                else:
                    alt_id = getattr(order, 'carrier', -1)
                    status = f"Dang tho (S{alt_id})" if alt_id != -1 else "Dang tho"
                text_color = 'blue'

            w_str = f"{w_val:.1f}"
            p_str = f"P{p_val}"
            d_str = f"d={order.et}"
            score_str = _order_score(order, current_shippers, current_orders, current_t) or "-"
            info_line = f"#{order.id:<3} {w_str:<8} {p_str:<8} t={order.appear_t:<8} {d_str:<9} {score_str:<7} {status:<13}"
            ax_info.text(0, y_offset, info_line, fontsize=8.5, family='monospace', color=text_color)
            y_offset -= 0.45

        if len(visible_orders) > 6:
            ax_info.text(0, y_offset,
                         f"... va {len(visible_orders) - 6} don hang khac dang xu ly.",
                         fontsize=8, style='italic', color='dimgray')

        # ---- 5e. CHU GIAI ----
        ax_legend.clear()
        ax_legend.axis('off')
        ax_legend.set_xlim(0, 10)
        ax_legend.set_ylim(0, 4)

        ax_legend.text(0, 3.7, "BANG CHU GIAI KY HIEU", fontsize=9.5, fontweight='bold', color='darkblue')

        ax_legend.plot(0.5, 2.8, marker='s', markersize=11, color='white', markeredgecolor='gray')
        ax_legend.text(1.0, 2.8, "O trong (Hop le)", fontsize=8, va='center')
        ax_legend.plot(4.5, 2.8, marker='s', markersize=11, color='black')
        ax_legend.text(5.0, 2.8, "Vat can (Tuong chan)", fontsize=8, va='center')

        ax_legend.plot(0.5, 1.9, marker='o', markersize=10, color='green')
        ax_legend.text(0.5, 1.9, "S", color='black', fontsize=6, ha='center', va='center', fontweight='bold')
        ax_legend.text(1.0, 1.9, "S_id: Diem nhan hang", fontsize=8, va='center')
        ax_legend.plot(4.5, 1.9, marker='X', markersize=10, color='red')
        ax_legend.text(5.0, 1.9, "E_id: Diem giao hang", fontsize=8, va='center')

        ax_legend.plot(0.5, 1.0, marker='v', markersize=11, color='#2196F3', markeredgecolor='black')
        ax_legend.text(1.0, 1.0, "Vi tri Shipper", fontsize=8, va='center')
        ax_legend.text(4.5, 1.0, "S_id(K): ID(So don trong tui)", fontsize=8,
                       fontweight='bold', color='#2196F3', va='center')

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 6. Widgets dieu khien
    # ------------------------------------------------------------------
    ax_slider_t  = fig.add_axes([0.08, 0.13, 0.42, 0.03])
    slider_t     = Slider(ax_slider_t, 'Timeline (t)', 0, total_frames - 1,
                          valinit=0, valfmt='%d', color='skyblue')

    ax_slider_sp = fig.add_axes([0.08, 0.08, 0.42, 0.03])
    slider_speed = Slider(ax_slider_sp, 'Toc do (ms)', 50, 1500,
                          valinit=interval, valfmt='%d ms', color='orange')

    ax_btn_prev = fig.add_axes([0.08, 0.02, 0.08, 0.04])
    ax_btn_play = fig.add_axes([0.18, 0.02, 0.10, 0.04])
    ax_btn_next = fig.add_axes([0.30, 0.02, 0.08, 0.04])

    btn_prev = Button(ax_btn_prev, '◀ Back',   color='lightgray', hovercolor='silver')
    btn_play = Button(ax_btn_play, '⏸ Pause',  color='lightgray', hovercolor='silver')
    btn_next = Button(ax_btn_next, 'Next ▶',   color='lightgray', hovercolor='silver')

    def stop_active_timer():
        if state["active_timer"] is not None:
            try:
                state["active_timer"].stop()
            except Exception:
                pass
            state["active_timer"] = None

    def update_via_slider(val):
        state["current_idx"] = int(val)
        draw_frame(state["current_idx"])

    slider_t.on_changed(update_via_slider)

    def change_speed(val):
        state["interval"] = int(val)
        if state["is_playing"]:
            stop_active_timer()
            restart_anim()

    slider_speed.on_changed(change_speed)

    def on_prev(event):
        state["is_playing"] = False
        btn_play.label.set_text('▶ Play')
        stop_active_timer()
        state["current_idx"] = max(0, state["current_idx"] - 1)
        slider_t.set_val(state["current_idx"])

    def on_next(event):
        state["is_playing"] = False
        btn_play.label.set_text('▶ Play')
        stop_active_timer()
        state["current_idx"] = min(total_frames - 1, state["current_idx"] + 1)
        slider_t.set_val(state["current_idx"])

    def on_play(event):
        if state["is_playing"]:
            state["is_playing"] = False
            btn_play.label.set_text('▶ Play')
            stop_active_timer()
        else:
            state["is_playing"] = True
            btn_play.label.set_text('⏸ Pause')
            if state["current_idx"] >= total_frames - 1:
                state["current_idx"] = 0
            restart_anim()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_play.on_clicked(on_play)

    def anim_worker():
        stop_active_timer()
        if state["is_playing"]:
            if state["current_idx"] < total_frames - 1:
                state["current_idx"] += 1
                slider_t.eventson = False
                slider_t.set_val(state["current_idx"])
                slider_t.eventson = True
                draw_frame(state["current_idx"])

                timer = fig.canvas.new_timer(
                    interval=state["interval"],
                    callbacks=[(anim_worker, (), {})]
                )
                state["active_timer"] = timer
                timer.start()
            else:
                state["is_playing"] = False
                btn_play.label.set_text('▶ Replay')

    def restart_anim():
        anim_worker()

    draw_frame(0)
    anim_worker()
    plt.show()
