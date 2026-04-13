import pandas as pd
import os
from config import *
from config2 import *

def selection(df):
    conditions = []

    if 'pt_1' in df.columns:
        conditions.append(df['pt_1'] > PT_1_CUT)
    if 'pt_2' in df.columns:
        conditions.append(df['pt_2'] > PT_2_CUT)
    if 'eta_1' in df.columns:
        conditions.append(abs(df['eta_1']) < ETA_1_CUT)
    if 'eta_2' in df.columns:
        conditions.append(abs(df['eta_2']) < ETA_2_CUT)
    if 'decayModePNet_2' in df.columns:
        conditions.append(df['decayModePNet_2'] == DECAYMODE_2_CUT)
    if 'idDeepTau2018v2p5VSjet_2' in df.columns:
        conditions.append(df['idDeepTau2018v2p5VSjet_2'] >= IDJET_2_CUT)
    if 'idDeepTau2018v2p5VSe_2' in df.columns:
        conditions.append(df['idDeepTau2018v2p5VSe_2'] >= IDE_2_CUT)
    if 'idDeepTau2018v2p5VSmu_2' in df.columns:
        conditions.append(df['idDeepTau2018v2p5VSmu_2'] >= IDMU_2_CUT)
    if 'iso_1' in df.columns:
        conditions.append(df['iso_1'] < ISO_1_CUT)
    if 'ip_LengthSig_1' in df.columns:
        conditions.append(abs(df['ip_LengthSig_1']) > IP_LENSIG_1_CUT)

    if not conditions:
        print("No conditions were found")
        return pd.Series(True, index=df.index)

    mask = conditions[0].astype(bool)
    for cond in conditions[1:]:
        mask &= cond.astype(bool)
    return mask

def SELECT(df_or_path):
    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path.copy()
    return df[selection(df)]

def plotting(df):
    columns_control = [col for col in CONTROL if col in df.columns]
    columns_resol = [col for col in RESOL if col in df.columns]

    if not columns_control and not columns_resol:
        print("No plotting columns were found in this DataFrame")

    return {"control": columns_control, "resolution": columns_resol}


def Stack_together(processes, nominal):
    results = {}
    
    for process in processes:
        folder_path = os.path.join(BASE_PATH, process, nominal)
        
        if not os.path.exists(folder_path):
            print(f"Folder nie istnieje: {folder_path}")
            results[process] = False
            continue
        
        parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        results[process] = len(parquet_files) > 1
    
    return results