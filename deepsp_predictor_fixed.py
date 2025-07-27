#!/usr/bin/env python3


import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json


# 1) Parse arguments
parser = argparse.ArgumentParser(description="DeepSP predictor (fixed)")
parser.add_argument(
    "--input_file", required=True,
    help="CSV with columns Name/VH/VL or Id/Heavy_Chain/Light_Chain"
)
args = parser.parse_args()


# 2) Load & normalize input CSV
df = pd.read_csv(args.input_file, dtype=str)
df = df.rename(columns={
    "Name": "Id",
    "VH":   "Heavy_Chain",
    "VL":   "Light_Chain"
})
for col in ("Id","Heavy_Chain","Light_Chain"):
    if col not in df.columns:
        sys.exit(f"ERROR: missing column '{col}' in {args.input_file}")


# 3) Write FASTA files
with open("seq_H.fasta","w") as fh, open("seq_L.fasta","w") as fl:
    for _, row in df.iterrows():
        fh.write(f">{row['Id']}\n{row['Heavy_Chain']}\n")
        fl.write(f">{row['Id']}\n{row['Light_Chain']}\n")
print("FASTA written")


# 4) Run ANARCI to produce IMGT-numbered CSVs
if os.system("ANARCI -i seq_H.fasta  -s imgt -o seq_aligned_H  --csv") != 0 \
or os.system("ANARCI -i seq_L.fasta  -s imgt -o seq_aligned_KL --csv") != 0:
    sys.exit("ERROR: ANARCI failed")
print("ANARCI done")


# 5) Read & cleanup IMGT alignments
H_aln = pd.read_csv("seq_aligned_H.csv",  index_col="Id", dtype=str).fillna("-").astype(str)
L_aln = pd.read_csv("seq_aligned_KL.csv", index_col="Id", dtype=str).fillna("-").astype(str)


# 6) Build combined HL alignment file
with open("seq_aligned_HL.txt","w") as out:
    for nm in df["Id"]:
        Hseq = "".join(H_aln.loc[nm].values.tolist())
        Lseq = "".join(L_aln.loc[nm].values.tolist())
        out.write(f"{nm} {Hseq + Lseq}\n")
print("HL alignment built")


# 7) Load DeepSP models (register Sequential)
def load_model(json_fp, weights_fp):
    custom_objs = {"Sequential": tf.keras.Sequential}
    with open(json_fp) as jf:
        model = model_from_json(jf.read(), custom_objects=custom_objs)
    model.load_weights(weights_fp)
    model.compile(optimizer="adam", loss="mae")
    return model

sap_model  = load_model("Conv1D_regressionSAPpos.json",  "Conv1D_regression_SAPpos.h5")
scmn_model = load_model("Conv1D_regressionSCMneg.json", "Conv1D_regression_SCMneg.h5")
scmp_model = load_model("Conv1D_regressionSCMpos.json", "Conv1D_regression_SCMpos.h5")
print("DeepSP models loaded")


# 8) Prepare one-hot arrays at the fixed input length
L = sap_model.input_shape[1]  
AA2IDX = {aa:i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}

def one_hot(seq):
    arr = np.zeros((L, len(AA2IDX)), dtype=np.float32)
    gap_idx = AA2IDX["-"]
    for i, aa in enumerate(seq[:L]):
        arr[i, AA2IDX.get(aa, gap_idx)] = 1.0
    return arr

names, arrs = [], []
with open("seq_aligned_HL.txt") as inf:
    for line in inf:
        nm, seq = line.strip().split()
        # pad/truncate to length L
        seq = (seq + "-"*L)[:L]
        names.append(nm)
        arrs.append(one_hot(seq))

X = np.stack(arrs, axis=0)
print(f" One-hot encoded {len(names)} sequences → shape {X.shape}")


# 9) Predict descriptors
sap_out  = sap_model.predict(X,  verbose=0)
scmn_out = scmn_model.predict(X, verbose=0)
scmp_out = scmp_model.predict(X, verbose=0)
print("Predictions done")


# 10) Assemble output DataFrame & write CSV
cdrs = ["CDRH1","CDRH2","CDRH3","CDRL1","CDRL2","CDRL3","CDR","Hv","Lv","Fv"]
sap_cols  = [f"SAP_pos_{n}"  for n in cdrs]
scmn_cols = [f"SCM_neg_{n}" for n in cdrs]
scmp_cols = [f"SCM_pos_{n}" for n in cdrs]

out_df = pd.DataFrame({"Id": names})
for i, col in enumerate(sap_cols):   out_df[col] = sap_out[:, i]
for i, col in enumerate(scmn_cols):  out_df[col] = scmn_out[:, i]
for i, col in enumerate(scmp_cols):  out_df[col] = scmp_out[:, i]

out_df.to_csv("DeepSP_descriptors.csv", index=False)
print(f"Wrote DeepSP_descriptors.csv ({out_df.shape[0]}×{out_df.shape[1]})")


