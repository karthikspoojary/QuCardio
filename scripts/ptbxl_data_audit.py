import pandas as pd, ast, numpy as np

db = pd.read_csv('data/ptb-xl/ptbxl_database.csv')
scp_df = pd.read_csv('data/ptb-xl/scp_statements.csv', index_col=0)

# Detect superclass column
col = 'diagnostic_class' if 'diagnostic_class' in scp_df.columns else 'superclass'

SUPERCLASS_PRIORITY = ['NORM', 'MI', 'CD', 'STTC']

def get_sc(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
    except:
        return None
    present = set()
    for code in codes:
        if code in scp_df.index:
            sc = scp_df.loc[code, col]
            if pd.notna(sc) and sc in SUPERCLASS_PRIORITY:
                present.add(sc)
    for sc in SUPERCLASS_PRIORITY:
        if sc in present:
            return sc
    return None

db['superclass'] = db['scp_codes'].apply(get_sc)
db = db.dropna(subset=['superclass'])  # drop HYP-only

print('=== Total after mapping (all folds) ===')
print(db['superclass'].value_counts())

print('\n=== Per fold counts ===')
for fold in range(1, 11):
    sub = db[db['strat_fold'] == fold]
    print(f'  Fold {fold:2d}: {len(sub):4d}  NORM={sub[sub.superclass=="NORM"].shape[0]}  MI={sub[sub.superclass=="MI"].shape[0]}  CD={sub[sub.superclass=="CD"].shape[0]}  STTC={sub[sub.superclass=="STTC"].shape[0]}')

print('\n=== Folds 1-8 total ===', len(db[db['strat_fold'].isin(range(1,9))]))
print('=== Folds 9-10 total ===', len(db[db['strat_fold'].isin([9,10])]))

# --- Stratified sample sizes for various N ---
train_df = db[db['strat_fold'].isin(range(1, 9))]
print('\n=== Class distribution in folds 1-8 ===')
print(train_df['superclass'].value_counts())

# What papers use — typical train sizes
for n in [500, 800, 1000, 1200, 2000]:
    per_class = n // 4
    available = train_df['superclass'].value_counts()
    feasible = all(available.get(sc, 0) >= per_class for sc in SUPERCLASS_PRIORITY)
    print(f'  n_train={n:5d} ({per_class}/class): {"OK" if feasible else "TOO FEW for some class"}')
