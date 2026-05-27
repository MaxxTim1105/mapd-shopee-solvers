from env import load_config, DeliveryEnv
from solvers.aco_solver import ACOSolver
import time

configs = load_config('valid_config.txt')
import sys
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 7
cfg = configs[idx]
print(f'Config: {cfg["name"]} N={cfg["N"]} C={cfg["C"]} G={cfg["G"]} T={cfg["T"]}')

start = time.time()
env = DeliveryEnv(cfg, seed=42)

budget = max(20.0, min(110.0, 0.12 * cfg["T"] + 9.0 * cfg["C"]))

obs = env.reset()
solver = ACOSolver(env)
solver._configure_by_observation(int(obs["N"]), int(obs["C"]), int(obs["T"]), obs["grid"])
solver.run_deadline = time.time() + budget

step = 0
while not obs.get("done", False):
    step += 1
    if step % 200 == 0:
        elapsed = time.time() - start
        delivered = sum(1 for o in obs['orders'].values() if o.delivered)
        print(f'  step {step}/{cfg["T"]} t={elapsed:.1f}s delivered={delivered}')
    if time.time() > solver.run_deadline:
        actions = {s.id: ("S", 2) for s in obs["shippers"]}
    else:
        actions = solver._decide(obs)
    obs, _, done, _ = env.step(actions)

result = solver.env.result(solver.method_name, time.time() - start)
elapsed = time.time() - start
print(f'Result: net_reward={result["net_reward"]} delivered={result["delivered"]}/{result["total_orders"]}')
print(f'on_time={result["on_time"]} late={result["late"]} missed={result["missed"]}')
print(f'elapsed={elapsed:.2f}s')
