import sys; sys.path.insert(0,'/home/murray/simworld_nav')
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier
NET=build_road_network(Path('vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris'),map_name='citycore-paris')
S=Path('/data/murray/paris_streets_v2/citycore-paris');G=Path('/data/murray/paris_signals_kerb/citycore-paris');O=Path('/data/murray/paris_obstacles/citycore-paris')
print(f"{'tier':7}{'stride':10}{'blind':>10}{'sighted':>10}")
for tier in ('solo','pair'):
  for stride in ('waypoint','block'):
    out=[]
    for sighted in (False,True):
      d=i=0
      for seed in range(20):
        e=CourierEnv(NET,seed=seed,difficulty=tier,stride=stride,album_root=S,signal_album_root=G,obstacle_album_root=O)
        e.reset(); ObservationOnlyCourier(e,max_steps=9000,sighted=sighted).run(seed)
        s=e.summary(); d+=s.get('delivered',0); i+=s.get('orders_issued',0)
      out.append(f"{d}/{i} ({100*d/i:.0f}%)")
    print(f"{tier:7}{stride:10}{out[0]:>10}{out[1]:>10}")
