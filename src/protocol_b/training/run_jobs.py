"""Run up to two independent folds, assigning one process to each visible GPU."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import json
import os
import subprocess
import sys
import time
from corrected_core import MODELS, atomic_json


def parse_job(value):
    model,fold=value.split(':')
    fold=int(fold)
    if model not in MODELS or fold not in range(1,6):raise ValueError(f'Invalid job: {value}')
    return model,fold


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',nargs='+',required=True)
    p.add_argument('--expected-manifest-sha',required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--output-root',type=Path,default=Path('/kaggle/working/pb_corrected'))
    p.add_argument('--max-hours',type=float,default=4.5)
    p.add_argument('--require-resume',action='store_true')
    args=p.parse_args(argv)
    jobs=[parse_job(v) for v in args.jobs]
    if not 1<=len(jobs)<=2 or len(set(jobs))!=len(jobs):raise ValueError('Use one or two different model/fold jobs.')
    if not .5<=args.max_hours<=8:raise ValueError('Choose a session budget from 0.5 to 8 hours.')
    import torch
    count=torch.cuda.device_count()
    if not count:raise RuntimeError('GPU required. Select T4 x2 for the paired notebooks.')
    available=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if os.environ.get('CUDA_VISIBLE_DEVICES') else [str(i) for i in range(count)]
    available=available[:min(2,count)]
    args.output_root.mkdir(parents=True,exist_ok=True)
    logs=args.output_root/'logs';logs.mkdir(exist_ok=True)
    deadline=time.monotonic()+args.max_hours*3600
    pending=list(jobs);active=[];outcomes=[];last_status=0
    while pending or active:
        occupied={item['gpu'] for item in active}
        for gpu in available:
            if not pending or gpu in occupied or deadline-time.monotonic()<1800:continue
            model,fold=pending.pop(0)
            logfile=logs/f'{model}_fold{fold}.log'
            stream=logfile.open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED='42',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            command=[sys.executable,str(Path(__file__).parent/'train_fold.py'),'--model',model,'--fold',str(fold),
                     '--input-root',str(args.input_root),'--output-root',str(args.output_root),
                     '--expected-manifest-sha',args.expected_manifest_sha,
                     '--max-hours',str(max(.01,(deadline-time.monotonic()-600)/3600))]
            if args.require_resume:command.append('--require-resume')
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            active.append({'proc':proc,'stream':stream,'model':model,'fold':fold,'gpu':gpu,'log':str(logfile)})
            print(f'START: {model} fold {fold} on GPU {gpu}',flush=True)
        for item in list(active):
            code=item['proc'].poll()
            if code is None:continue
            item['stream'].close()
            status=args.output_root/item['model']/f"fold{item['fold']}"/'status.json'
            result=json.loads(status.read_text()) if status.exists() else {'status':'failed_before_checkpoint'}
            outcomes.append({'model':item['model'],'fold':item['fold'],'exit_code':code,'status':result,'log':item['log']})
            print(f"FINISH: {item['model']} fold {item['fold']}: {result.get('status')}; exit code {code}",flush=True)
            if code:
                print(Path(item['log']).read_text(encoding='utf-8')[-7000:],flush=True)
            active.remove(item)
        if time.monotonic()-last_status>90:
            for item in active:
                path=args.output_root/item['model']/f"fold{item['fold']}"/'status.json'
                status=json.loads(path.read_text()) if path.exists() else {'status':'preflight/loading'}
                print(f"PROGRESS: {item['model']} fold {item['fold']}: {status}",flush=True)
            last_status=time.monotonic()
        if not active and pending and deadline-time.monotonic()<1800:
            outcomes.extend({'model':m,'fold':f,'exit_code':None,'status':{'status':'not_started_time_budget'}} for m,f in pending)
            pending=[]
        # A hung/failed child must not keep a notebook running to the platform cap.
        if time.monotonic()>deadline+600:
            for item in active:
                item['proc'].terminate();item['proc'].wait();item['stream'].close()
                outcomes.append({'model':item['model'],'fold':item['fold'],'exit_code':'time_guard',
                                 'status':{'status':'interrupted_last_complete_checkpoint_retained'}})
            active=[]
        if active:time.sleep(10)
    ready=all(o['exit_code']==0 and o['status'].get('status')=='complete' for o in outcomes)
    atomic_json(args.output_root/'round_status.json',{'all_requested_jobs_complete':ready,
                'input_manifest_sha256':args.expected_manifest_sha,'jobs':outcomes,
                'instruction':'Full .pt/.json outputs are required for resume. Small results ZIP excludes weights.'})
    token='__'.join(f'{model}_f{fold}' for model,fold in jobs)
    bundle=args.output_root.parent/f'PB_Corrected_{token}_RESULTS_ONLY.zip'
    with ZipFile(bundle,'w',compression=ZIP_DEFLATED) as zipped:
        zipped.write(args.output_root/'round_status.json','pb_corrected/round_status.json')
        for model,fold in jobs:
            directory=args.output_root/model/f'fold{fold}'
            for path in sorted(directory.glob('*')):
                if path.is_file() and path.suffix in ['.json','.csv','.npy'] and '_resume_' not in path.name:
                    zipped.write(path,'pb_corrected/'+path.relative_to(args.output_root).as_posix())
    print('ROUND COMPLETE' if ready else 'ROUND SAVED: one or more jobs need resuming or attention. Read round_status.json and logs.',flush=True)
    print(f'Small results ZIP: {bundle}',flush=True)
    print('Keep the FULL saved notebook outputs for all .pt model weights and resume checkpoints.',flush=True)
    # Finish normally so Kaggle can preserve useful checkpoints even when a child failed.
    # Completion is established only by validated fold_complete.json files, never this exit code.
    return 0


if __name__=='__main__':raise SystemExit(main())
