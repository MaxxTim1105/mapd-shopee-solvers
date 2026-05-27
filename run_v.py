import argparse
import sys
import os
from env import load_config, SEED
from run_test import load_solver_class, SOLVER_SOURCES
from visualizer import visualize_simulation

def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa hành trình MAPD")
    parser.add_argument("--config", required=True, help="Đường dẫn file test_config.txt")
    parser.add_argument("--config_name", default="C1", help="Tên config cụ thể muốn chạy (ví dụ: C1)")
    parser.add_argument("--method", default="GreedyBFS", help="Thuật toán muốn chạy: GreedyBFS, VRPOrToolsSolver, ACOSolver, MAPDCBSSolver")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--interval", type=int, default=300, help="Tốc độ chuyển bước (ms)")
    args = parser.parse_args()

    # 1. Tìm thông tin file nguồn của solver được yêu cầu
    solver_info = next((item for item in SOLVER_SOURCES if item[0] == args.method), None)
    if not solver_info:
        sys.exit(f"[ERROR] Phương pháp '{args.method}' không hợp lệ.")
        
    print(f"-> Đang tải thuật toán: {args.method}")
    solver_cls = load_solver_class(solver_info[0], solver_info[1])

    # 2. Đọc danh sách cấu hình ma trận bản đồ từ file config
    configs = load_config(args.config)
    target_cfg = next((c for c in configs if c.get("name") == args.config_name), None)
    if not target_cfg:
        sys.exit(f"[ERROR] Không tìm thấy config có tên '{args.config_name}' trong file {args.config}.")

    print(f"-> Đang khởi chạy mô phỏng trực quan trên cấu hình: {args.config_name}")
    # 3. Gọi hàm hiển thị giao diện động
    visualize_simulation(solver_cls, target_cfg, base_seed=args.seed, interval=args.interval)

if __name__ == "__main__":
    main()