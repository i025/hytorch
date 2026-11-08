# Frozen acceptance (before candidate execution)

Original regression inputs/references/tolerances remain unchanged. Historical GEMM strict failures are disclosed separately.

D: A = original 6D im2col+tensordot; B = fresh native packed im2col + CuPy G@col; R = independent NumPy CPU im2col in FP64 using the SAME FP32 input values. Gate B vs R: abs(B-R)<=2e-5+2e-4*abs(R). Report A/R, B/R, B/A, bitwise element fraction, absolute maximum, relative L2, failing counts/coordinates; also original diagnostic 1e-5/1e-4 and 2e-6/2e-5. B/A failure alone is not blocking.

Fixed data: exact previous 4D-1 10 geometry specs, seeds42,123,501 and contiguous/negative/slice layouts =90 cases,30 independent shape/seed. Holdout: all10 plus new (N,C,H,W,OC,KH,KW,SH,SW,PH,PW)=(2,4,21,29,13,2,5,3,2,1,2),(1,9,12,19,17,4,2,2,3,2,1),(3,2,15,11,6,3,1,1,2,1,0), seeds2026,31415, same3 layouts =78 cases. Unit-normal FP32 x/gy. Real data: B0 DonkeyNet step0,95,191, all5 layers x/gy captured outside performance runs. No failed case deletion. FP64 is validation only; record longdouble precision and independently scalar-check representative errors.

Training: original final-weight contract atol1e-4/rtol1e-3 (prior scripts/reports); all per-parameter elements. Fixed single-step gradient/weight comparisons atol2e-5/rtol2e-4 (existing DonkeyNet dtype tests); final gradients also report/check same. Full loss trajectory finite; compare every point atol1e-4/rtol1e-3, not hashes. Separate exact SGD validation identical p/g (including signbits for finite values, NaN/Inf classification for nonfinite), dynamic lr, hooks, aliases, subclasses, stream, tensor counts0,1,16,32,33,65.

4 configs B0,D,S,DS: shared GEMM, im2col forward, native pool, optimized native col2im ALL configs; default globally unchanged. DonkeyNet256 samples/batch4/epochs3/192steps,seed42,dropout0,lr.001,TF32off. 12 rounds Williams balanced order prewritten raw/order.json. Warmup10 each then reset parameters/grads/RNG. No timed per-step loss read, sync or snapshots; queue loss copies on GPU identically and retrieve after timing. Pair delta B0-candidate 95% t(df11) CI; no optional stopping or case exclusion.

Supplement: Donkey batch1,16 (same256samples/3epochs), TinyNet batch4 same256samples/3epochs,3 paired rounds B0 vs final recommendation. CuPy and Torch FP32 reference5 rounds each same main workload, normal torch SGD foreach/fused defaults None,cuDNN enabled,TF32off. Record actual configuration.

Failure: correct/disable candidate and continue other candidate/report. Performance acceptance requires repeatable unprofiled wall evidence; no automatic defaults changed. Profiles separately alternating B0,D,S,DS twice; counts and elapsed not summed across CPU/GPU domains. No numerical forward/backward cache. No additional algorithms.
