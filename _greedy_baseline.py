from env import load_config, DeliveryEnv
from solvers.greedy_bfs import GreedyBFS
import time

configs = load_config('valid_config.txt')
for ci in [0, 3, 7]:
  cfg = configs[ci]
  start = time.time()
  env = DeliveryEnv(cfg, seed=42)
  solver = GreedyBFS(env)
  result = solver.run()
  elapsed = time.time() - start
  name = cfg['name']
  print(f'{name}: net={result["net_reward"]} del={result["delivered"]}/{result["total_orders"]} on_time={result["on_time"]} late={result["late"]} missed={result["missed"]} t={elapsed:.2f}s')
