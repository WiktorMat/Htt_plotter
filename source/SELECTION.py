import pandas as pd
from config import *

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
        return conditions

    mask = conditions[0]
    for cond in conditions[1:]:
        mask &= cond
    return mask

def SELECT(df_or_path):
    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path.copy()
    return df[selection(df)]