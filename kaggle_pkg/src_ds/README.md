Staging folder of the Kaggle dataset `waelyt/tfb-src`, which the `tfb-assemble` kernel reads.
The pipeline writes its artifacts here: queue_*.pkl, t4_l2.pkl, t1_shock_ps.npz, t1nn_post5.npz, asm_jobs.txt, plus a copy of src/*.py.
The artifacts are not committed. For the final submission, asm_jobs.txt is the single line:

    msmU_stF 0.9 0.07 0.03 q=0.8 s=0.8 n=0.85,0.5 g=1.0 nf=5
