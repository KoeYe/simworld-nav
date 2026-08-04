import sys,re; sys.path.insert(0,'/home/murray/simworld_nav')
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network, project_to_polyline
from embodiedbench.runtime.city.courier_env import CourierEnv
NET=build_road_network(Path('vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris'),map_name='citycore-paris')
e=CourierEnv(NET,seed=0,difficulty='solo',stride='block',album_root=Path('/data/murray/paris_streets_v2/citycore-paris'))
e.reset()
side_ok=side_bad=disp_ok=disp_bad=0; worst=[]
for st in NET.streets:
    nodes=[n for n in NET.nodes.values() if n.street_index==st.index]
    if len(nodes)<5: continue
    ordered=sorted(nodes,key=lambda n: project_to_polyline(n.position,st.polyline)[1])
    seq=[[int(x) for x in re.findall(r'\d+',e.house_numbers_near(n.id) or '')] for n in ordered]
    odds=[x for s in seq for x in s if x%2]; evens=[x for s in seq for x in s if x%2==0]
    if odds==sorted(odds): side_ok+=1
    else: side_bad+=1
    if evens==sorted(evens): side_ok+=1
    else: side_bad+=1
    flat=[(i,x) for i,s in enumerate(seq) for x in s]
    inv=sum(1 for a in range(len(flat)) for b in range(a+1,len(flat)) if flat[b][1]<flat[a][1])
    if inv: disp_bad+=1
    else: disp_ok+=1
    # how many junctions apart are odd n and even n+1
    pos={}
    for i,s in enumerate(seq):
        for x in s: pos.setdefault(x,i)
    lags=[abs(pos[k]-pos[k+1]) for k in pos if k%2 and k+1 in pos]
    if lags: worst.append((max(lags),st.name,seq[:10]))
print(f'per-side monotone: {side_ok} ok / {side_bad} broken  (of {side_ok+side_bad} sides)')
print(f'displayed sequence monotone: {disp_ok} ok / {disp_bad} has inversions  (of {disp_ok+disp_bad} streets)')
worst.sort(reverse=True)
print('\nworst odd/even lag (junctions between number N and N+1):')
for lag,name,s in worst[:6]: print(f'  {lag:2d}  {name}')
