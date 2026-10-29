"""Full output matrices against GPU FP64, with fsum cross-checks and raw arrays."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
import cupy as cp
import torch
from common import ROOT, setup, metadata, shared_call, inputs, write_csv, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', required=True)
p.add_argument('--correctness', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--raw', type=Path, required=True)
args = p.parse_args()
assert json.loads((args.correctness/'metadata.json').read_text())['status'] == 'PASSED'
args.output.mkdir(parents=True, exist_ok=False)
args.raw.mkdir(parents=True, exist_ok=False)
kernel = setup(args.library)
methods = ('shared32', 'block32_tree', 'cupy', 'torch', 'numpy_fp32')
all_metrics, histograms, position_rows, changes, cross_checks, cross_methods, semantic = [], [], [], [], [], [], []
max_inputs = []
old_samples = list(csv.DictReader((ROOT/'docs/experiments/gemm-fp32-accuracy/results/samples.csv').open()))
old_positions = json.loads((ROOT/'docs/experiments/gemm-fp32-accuracy/results/sample-positions.json').read_text())
absolute_bins = np.array([0,1e-14,1e-12,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,np.inf])
ulp_bins = np.array([0,1,2,4,8,16,32,64,128,1024,65536,np.inf])


def ordered32(values):
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.int64)
    return np.where(bits & 0x80000000, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)


def checkpoints():
    # Individual case files are final and exclusive; this index is a progress snapshot.
    (args.output/'progress.json').write_text(json.dumps(dict(cases_completed=len(changes),
        full_matrix_comparisons=len(all_metrics), reference_cross_checks=len(cross_checks)), indent=2))


for n in (1024,2048,4096):
    for seed_base in (42,1729,20261029):
        case = f'n{n}-seed{seed_base+n}'
        raw = args.raw/case
        raw.mkdir()
        a,b = inputs(n,seed_base)
        ca,cb = cp.asarray(a),cp.asarray(b)
        a64,b64 = ca.astype(cp.float64),cb.astype(cp.float64)
        reference = cp.asnumpy(cp.matmul(a64,b64))
        sum_abs = cp.asnumpy(cp.matmul(cp.abs(a64),cp.abs(b64)))
        assert reference.dtype == np.float64 and np.isfinite(reference).all()
        numpy32 = np.matmul(a,b)
        output = cp.empty((n,n),cp.float32)
        host = {'numpy_fp32':numpy32}
        shared_call(ca,cb,output)()
        host['shared32'] = cp.asnumpy(output)
        kernel.call(ca,cb,output)
        host['block32_tree'] = cp.asnumpy(output)
        cp.matmul(ca,cb,out=output)
        host['cupy'] = cp.asnumpy(output)
        ta,tb = torch.from_numpy(a).cuda(0),torch.from_numpy(b).cuda(0)
        ty = torch.empty((n,n),device='cuda',dtype=torch.float32)
        torch.matmul(ta,tb,out=ty)
        host['torch'] = ty.cpu().numpy()
        del ta,tb,ty,a64,b64,ca,cb,output
        for name,value in dict(a=a,b=b,reference_fp64=reference,sum_abs_products=sum_abs,**host).items():
            assert np.isfinite(value).all(), (case,name)
            with (raw/f'{name}.npy').open('xb') as stream:
                np.save(stream,value,allow_pickle=False)
        condition = np.divide(sum_abs,np.abs(reference),out=np.full_like(reference,np.inf),where=reference!=0)
        masks = {'ALL':np.ones((n,n),bool),'NEAR_ZERO':np.abs(reference)<=0.01,'SEVERE_CANCELLATION':condition>=1e4}
        old_error = np.abs(host['shared32'].astype(np.float64)-reference)
        new_error = np.abs(host['block32_tree'].astype(np.float64)-reference)
        change = np.where(new_error<old_error,2,np.where(new_error>old_error,0,1)).astype(np.uint8)
        np.save(raw/'improvement-map.npy',change,allow_pickle=False)
        # Every row-major output has two bits: 0=worse, 1=equal, 2=better vs FP64.
        bits = np.stack((change.ravel()&1,change.ravel()>>1),axis=1).ravel()
        with gzip.open(args.output/f'{case}-improvement-map.bin.gz','xb') as stream:
            stream.write(np.packbits(bits,bitorder='little').tobytes())
        old_fail = old_error>1e-4+1e-4*np.abs(reference)
        new_fail = new_error>1e-4+1e-4*np.abs(reference)
        original_old_fail = np.abs(host['shared32'].astype(np.float64)-numpy32)>1e-4+1e-4*np.abs(numpy32.astype(np.float64))
        original_new_fail = np.abs(host['block32_tree'].astype(np.float64)-numpy32)>1e-4+1e-4*np.abs(numpy32.astype(np.float64))
        change_record = dict(n=n,seed_base=seed_base,seed=seed_base+n,elements=n*n,
            improved=int(np.count_nonzero(change==2)),equal=int(np.count_nonzero(change==1)),worsened=int(np.count_nonzero(change==0)),
            old_fail_numpy=int(original_old_fail.sum()),new_fail_numpy=int(original_new_fail.sum()),
            resolved_numpy=int(np.count_nonzero(original_old_fail&~original_new_fail)),
            introduced_numpy=int(np.count_nonzero(~original_old_fail&original_new_fail)),
            old_fail_fp64=int(old_fail.sum()),new_fail_fp64=int(new_fail.sum()),
            resolved_fp64=int(np.count_nonzero(old_fail&~new_fail)),introduced_fp64=int(np.count_nonzero(~old_fail&new_fail)))
        changes.append(change_record)
        for index,left in enumerate(methods[:-1]):
            for right in methods[index+1:-1]:
                cross_methods.append(dict(n=n,seed_base=seed_base,a=left,b=right,
                    unequal_elements=int(np.count_nonzero(host[left]!=host[right])),
                    max_abs_difference=float(np.max(np.abs(host[left].astype(np.float64)-host[right])))))
        important = np.zeros((n,n),bool)
        maxima = set()
        case_metrics = []
        for method in methods:
            actual = host[method].astype(np.float64)
            error = np.abs(actual-reference)
            numpy_error = np.abs(actual-numpy32)
            fail64 = error>1e-4+1e-4*np.abs(reference)
            fail32 = numpy_error>1e-4+1e-4*np.abs(numpy32.astype(np.float64))
            important |= fail64 | fail32
            ulps = np.abs(ordered32(host[method])-ordered32(reference))
            max_locations = {}
            for ref_name,errors in [('fp64',error),('numpy',numpy_error)]:
                pos = tuple(int(v) for v in np.unravel_index(np.argmax(errors),errors.shape))
                maxima.add(pos)
                max_locations[ref_name] = pos
            for scope,mask in masks.items():
                errors = error[mask]
                ref = reference[mask]
                ulp = ulps[mask]
                row = dict(n=n,seed_base=seed_base,seed=seed_base+n,method=method,scope=scope,elements=int(mask.sum()),
                    mean_abs_error=float(np.mean(errors)),max_abs_error=float(np.max(errors)),
                    relative_l2_error=float(np.linalg.norm(errors)/np.linalg.norm(ref)),
                    rmse=float(np.sqrt(np.mean(errors**2))),
                    ulp_p50=float(np.percentile(ulp,50)),ulp_p95=float(np.percentile(ulp,95)),ulp_p99=float(np.percentile(ulp,99)),ulp_max=int(ulp.max()),
                    fail_numpy=int(np.count_nonzero(fail32&mask)),fail_fp64=int(np.count_nonzero(fail64&mask)),
                    mean_abs_vs_numpy=float(np.mean(numpy_error[mask])),max_abs_vs_numpy=float(np.max(numpy_error[mask])),
                    unequal_to_numpy=int(np.count_nonzero((host[method]!=numpy32)&mask)),
                    max_fp64_row=max_locations['fp64'][0] if scope=='ALL' else None,
                    max_fp64_col=max_locations['fp64'][1] if scope=='ALL' else None,
                    max_numpy_row=max_locations['numpy'][0] if scope=='ALL' else None,
                    max_numpy_col=max_locations['numpy'][1] if scope=='ALL' else None)
                case_metrics.append(row)
                for metric,values,edges in [('abs_error_fp64',errors,absolute_bins),('ulp',ulp,ulp_bins)]:
                    counts,_ = np.histogram(values,edges)
                    for lo,hi,count in zip(edges[:-1],edges[1:],counts):
                        histograms.append(dict(n=n,seed_base=seed_base,method=method,scope=scope,metric=metric,
                            lower=float(lo),upper=float(hi) if np.isfinite(hi) else 'inf',count=int(count)))
        all_metrics.extend(case_metrics)
        for i,j in sorted(maxima):
            important[i,j] = True
            max_inputs.append(dict(n=n,seed_base=seed_base,row=i,col=j,a_row=a[i].tolist(),b_col=b[:,j].tolist(),
                reference_fp64=float(reference[i,j]),outputs={name:float(value[i,j]) for name,value in host.items()}))
        old_failure_path=ROOT/f'docs/experiments/gemm-fp32-accuracy/results/gpu-01/shared32-{n}-failure-coordinates.csv'
        historical = {(int(r['row']),int(r['col'])) for r in csv.DictReader(old_failure_path.open())} if seed_base==42 else set()
        if seed_base==42:
            assert historical == {tuple(map(int,p)) for p in np.argwhere(original_old_fail)}, 'Historical full-matrix failures changed'
            selected=[r for r in old_samples if int(r['n'])==n and r['strategy']=='block32']
            mismatches=sum(np.float32(float(r['value'])).view(np.uint32)!=host['block32_tree'][int(r['row']),int(r['col'])].view(np.uint32) for r in selected)
            semantic.append(dict(n=n,samples=len(selected),bitwise_mismatches=int(mismatches)))
            assert mismatches==0,'Large sampled GPU output does not match 3A block32 results'
        for i,j in np.argwhere(important):
            point=dict(n=n,seed_base=seed_base,row=int(i),col=int(j),reference_fp64=float(reference[i,j]),
                sum_abs_products=float(sum_abs[i,j]),cancellation_condition=float(condition[i,j]),near_zero=bool(masks['NEAR_ZERO'][i,j]),
                historical_failure=(int(i),int(j)) in historical,maximum_error_position=(int(i),int(j)) in maxima,
                new_vs_old_fp64=('worse','equal','better')[int(change[i,j])])
            for method,value in host.items():
                val=float(value[i,j]);ref=float(reference[i,j]);ref32=float(numpy32[i,j])
                point.update({method:val,method+'_abs_fp64':abs(val-ref),method+'_abs_numpy':abs(val-ref32),
                    method+'_fail_fp64':abs(val-ref)>1e-4+1e-4*abs(ref),method+'_fail_numpy':abs(val-ref32)>1e-4+1e-4*abs(ref32)})
            position_rows.append(point)
        # Cross-check all old failures plus near-zero and seeded normal positions.
        checks={position:'historical_failure' for position in historical}
        for point in old_positions:
            if point['n']==n and point['stratum']=='historical_worst':
                checks.setdefault((point['row'],point['col']),'historical_coordinate')
        near=np.argpartition(np.abs(reference).ravel(),127)[:128]
        for index in near:
            checks.setdefault(tuple(int(v) for v in np.unravel_index(index,(n,n))),'near_zero')
        rng=np.random.default_rng(seed_base+n+900000)
        for index in rng.choice(n*n,128,replace=False):
            checks.setdefault(tuple(int(v) for v in np.unravel_index(index,(n,n))),'random')
        for (i,j),stratum in checks.items():
            fsum=math.fsum(float(x)*float(y) for x,y in zip(a[i],b[:,j]))
            dot=float(np.dot(a[i].astype(np.float64),b[:,j].astype(np.float64)))
            diff=abs(float(reference[i,j])-fsum)
            cross_checks.append(dict(n=n,seed_base=seed_base,row=i,col=j,stratum=stratum,
                gpu_fp64=float(reference[i,j]),cpu_fsum=fsum,cpu_fp64_dot=dot,
                abs_gpu_fsum=diff,abs_dot_fsum=abs(dot-fsum),
                rounded32_match=bool(np.float32(reference[i,j])==np.float32(fsum)),
                condition=float(condition[i,j])))
            assert diff<=1e-10,'GPU FP64 reference cross-check exceeded 1e-10 absolute error'
        write_csv(args.output/f'{case}-summary.csv',case_metrics)
        write_json(raw/'case.json',dict(n=n,seed_base=seed_base,seed=seed_base+n,
            input_generation='default_rng(seed); standard_normal float32 A then B',files=sorted(p.name for p in raw.glob('*.npy'))))
        checkpoints()
        print(json.dumps(change_record),flush=True)
        del a,b,host,reference,numpy32,sum_abs,condition,masks,old_error,new_error,change,important
        cp.get_default_memory_pool().free_all_blocks()

write_csv(args.output/'full-matrix-summary.csv',all_metrics)
write_csv(args.output/'error-distributions.csv',histograms)
write_csv(args.output/'error-positions.csv',position_rows)
write_csv(args.output/'improvement-summary.csv',changes)
write_csv(args.output/'reference-cross-checks.csv',cross_checks)
write_csv(args.output/'framework-comparisons.csv',cross_methods)
with gzip.open(args.output/'max-error-inputs.json.gz','xt') as stream:
    json.dump(max_inputs,stream,allow_nan=False)
write_json(args.output/'summary.json',dict(**metadata(),status='COMPLETED',gpu_status='EXECUTED',
    scope='FULL_MATRIX',cases=len(changes),elements_compared_per_method=sum(c['elements'] for c in changes),
    seed_bases=[42,1729,20261029],sizes=[1024,2048,4096],atol=1e-4,rtol=1e-4,
    new_all_numpy_tolerance_pass=all(c['new_fail_numpy']==0 for c in changes),
    new_all_fp64_tolerance_pass=all(c['new_fail_fp64']==0 for c in changes),
    large_stage3a_semantic_checks=semantic,reference_cross_checks=len(cross_checks),
    max_gpu_fp64_fsum_difference=max(c['abs_gpu_fsum'] for c in cross_checks),
    rounded32_reference_disagreements=sum(not c['rounded32_match'] for c in cross_checks),
    raw_directory=str(args.raw.resolve()),map_format='gzip of 2-bit row-major statuses, little-bit-order: 0=worse,1=equal,2=better; relative to FP64',
    near_zero_definition='abs(FP64)<=0.01',severe_cancellation_definition='sum(abs(a*b))/abs(FP64)>=10000'))
print('COMPLETED: full matrices and all per-position change maps retained',flush=True)
