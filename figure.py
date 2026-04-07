import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 层命名（按索引 0..5）=====
labels = ['Attn1','Attn2','Linear1','Linear2','FFN1','FFN2']

# ====== OS 数据（填入你上一次 OS 结果）======
os = [
    {"layer":0, "total":120594, "compute":9183,   "stall":0,      "util":11.15, "map_eff":100.00,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.000},
    {"layer":1, "total":117546, "compute":6135,   "stall":0,      "util":16.69, "map_eff":25.00,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":9.999},
    {"layer":2, "total":3513227,"compute":3117515,"stall":2957156,"util":3.85,  "map_eff":98.68,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":1.554},
    {"layer":3, "total":426936, "compute":59079,  "stall":0,      "util":67.71, "map_eff":89.29,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":15.625},
    {"layer":4, "total":1029263,"compute":609833, "stall":508554, "util":12.59, "map_eff":100.00,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":4.631},
    {"layer":5, "total":1016591,"compute":637073, "stall":536778, "util":12.06, "map_eff":89.29,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":15.625}
]
os_df = pd.DataFrame(os)
os_df['name'] = labels
os_df['compute_pct'] = os_df['compute']/os_df['total']*100
os_df['stall_pct']   = os_df['stall']  /os_df['total']*100

# ====== WS 数据（填入你这次 WS 结果）======
ws = [
    {"layer":0, "total":138231,"compute":7159,   "stall":0,      "util":14.30, "map_eff":25.00,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.000},
    {"layer":1, "total":138231,"compute":7159,   "stall":0,      "util":14.30, "map_eff":25.00,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.000},
    {"layer":2, "total":4077729,"compute":3573106,"stall":3335037,"util":3.36, "map_eff":88.11,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":9.162},
    {"layer":3, "total":1352006,"compute":862125,"stall":774416, "util":4.64, "map_eff":79.72,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.844},
    {"layer":4, "total":2426033,"compute":1901746,"stall":1751387,"util":4.04, "map_eff":89.29,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.467},
    {"layer":5, "total":2204424,"compute":1706352,"stall":1555993,"util":4.50, "map_eff":89.29,
     "ifmap_dram_bw":10.000, "filter_dram_bw":10.000, "ofmap_dram_bw":10.446}
]
ws_df = pd.DataFrame(ws)
ws_df['name'] = labels
ws_df['compute_pct'] = ws_df['compute']/ws_df['total']*100
ws_df['stall_pct']   = ws_df['stall']  /ws_df['total']*100

# ====== 作图（按新命名）======
x = np.arange(len(labels))
width = 0.35

# 1) Overall Utilization：OS vs WS
plt.figure(figsize=(10,4))
plt.bar(x - width/2, os_df['util'], width, label='OS', color='#4e79a7')
plt.bar(x + width/2, ws_df['util'], width, label='WS', color='#f28e2b')
plt.xticks(x, labels); plt.ylabel('Overall Utilization (%)')
plt.title('Overall Utilization: OS vs WS')
plt.legend(); plt.tight_layout()
plt.savefig('renamed_cmp_util_os_ws.png', dpi=200)

# 2) Stall Ratio：OS vs WS
plt.figure(figsize=(10,4))
plt.bar(x - width/2, os_df['stall_pct'], width, label='OS', color='#e15759')
plt.bar(x + width/2, ws_df['stall_pct'], width, label='WS', color='#76b7b2')
plt.xticks(x, labels); plt.ylabel('Stall (% total cycles)')
plt.title('Stall Ratio: OS vs WS')
plt.legend(); plt.tight_layout()
plt.savefig('renamed_cmp_stall_os_ws.png', dpi=200)

# 3) Mapping Efficiency：OS vs WS
plt.figure(figsize=(10,4))
plt.plot(x, os_df['map_eff'], marker='o', label='OS', color='#59a14f')
plt.plot(x, ws_df['map_eff'], marker='s', label='WS', color='#edc948')
plt.xticks(x, labels); plt.ylabel('Mapping Efficiency (%)')
plt.title('Mapping Efficiency: OS vs WS')
plt.legend(); plt.tight_layout()
plt.savefig('renamed_cmp_mapping_os_ws.png', dpi=200)

# 4) OFMAP DRAM 带宽：OS vs WS
plt.figure(figsize=(10,4))
plt.plot(x, os_df['ofmap_dram_bw'], marker='o', label='OS OFMAP BW', color='#e15759')
plt.plot(x, ws_df['ofmap_dram_bw'], marker='s', label='WS OFMAP BW', color='#b07aa1')
plt.axhline(10.0, color='gray', linestyle='--', linewidth=1, label='Interface Limit (10)')
plt.xticks(x, labels); plt.ylabel('OFMAP DRAM BW (words/cycle)')
plt.title('OFMAP DRAM BW: OS vs WS')
plt.legend(); plt.tight_layout()
plt.savefig('renamed_cmp_ofmap_bw_os_ws.png', dpi=200)

# 5) 导出对比表（含层名）
delta = os_df[['name','layer','total','util','map_eff','stall_pct','ofmap_dram_bw']].merge(
    ws_df[['name','layer','total','util','map_eff','stall_pct','ofmap_dram_bw']],
    on=['name','layer'], suffixes=('_OS','_WS')
)
delta['delta_util']      = delta['util_WS']      - delta['util_OS']
delta['delta_stall_pct'] = delta['stall_pct_WS'] - delta['stall_pct_OS']
delta['delta_map_eff']   = delta['map_eff_WS']   - delta['map_eff_OS']

delta[[
    'name','layer','total_OS','total_WS',
    'util_OS','util_WS','delta_util',
    'stall_pct_OS','stall_pct_WS','delta_stall_pct',
    'map_eff_OS','map_eff_WS','delta_map_eff',
    'ofmap_dram_bw_OS','ofmap_dram_bw_WS'
]].to_csv('renamed_summary_OS_vs_WS.csv', index=False)

print('Saved: renamed_cmp_*.png, renamed_summary_OS_vs_WS.csv')