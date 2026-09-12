#!/usr/bin/env python3
import argparse,subprocess,sys
p=argparse.ArgumentParser(); p.add_argument('--descriptor-table',required=True); p.add_argument('--assay-table'); p.add_argument('--assay-config',default='data/assay_config_template.csv'); p.add_argument('--vibe-gate',type=float); a=p.parse_args(); py=sys.executable
def run(xs): print('\n$',' '.join(map(str,xs))); subprocess.run(xs,check=True)
run([py,'analysis/00_validate_inputs.py',a.descriptor_table]); run([py,'analysis/01_prepare_analysis_table.py',a.descriptor_table]); table='results/analysis_table.csv'; run([py,'analysis/02_cohort_eda.py',table]); cmd=[py,'analysis/04_outcome_threshold.py',table]; cmd += [] if a.vibe_gate is None else ['--gate',str(a.vibe_gate)]; run(cmd); cmd=[py,'analysis/05_descriptor_analysis.py',table]; cmd += [] if a.vibe_gate is None else ['--gate',str(a.vibe_gate)]; run(cmd); run([py,'analysis/06_descriptor_pca.py',table]); cmd=[py,'analysis/07_surrogate_model_cv.py',table]; cmd += [] if a.vibe_gate is None else ['--gate',str(a.vibe_gate)]; run(cmd)
if a.assay_table and a.vibe_gate is not None: run([py,'analysis/03_assay_complementarity.py',a.assay_table,'--assay-config',a.assay_config,'--vibe-gate',str(a.vibe_gate)])
elif a.assay_table: print('Assay stage skipped: provide --vibe-gate explicitly so a threshold is not silently derived from the wrong subset.')
