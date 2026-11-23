"""Reporting-only scope overlay. Frozen experiment modules are read, not edited."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from supplemental_common import sha,digest,verify_inputs,extract_results,write_json,bundle,export_tables
import supplemental_report as original
import additional_report as dose_reporting
from extension_common import verify_extension

PARENT_SHA='8d6fe11c06d5ce9af25f227d99fd63fdc4854dba924e01a584b0126efad4c5f5'
EXTENSION_SHA='c464d0e82def618ee5caadf394c6894b52184d5f5e6b6a27994a6fed1157d371'
SCOPE_STATUS='not performed—scope decision'
LIMITATION=('Reliance on eye bars, watermarks and other source marks was not directly evaluated using manually verified annotations. '
    'Targeted artifact-masking experiments were not performed as a scope decision. '
    'The remaining saliency and perturbation checks cannot rule out artifact-related confounding or establish clinically meaningful lesion localisation.')
SIX_RESULTS={
 'SUPP_alve_AUTO_RESULTS_ONLY.zip':('SUPP_alve_AUTO','auto_progress.json'),
 'SUPP_friend1_AUTO_RESULTS_ONLY.zip':('SUPP_friend1_AUTO','auto_progress.json'),
 'SUPP_friend2_AUTO_RESULTS_ONLY.zip':('SUPP_friend2_AUTO','auto_progress.json'),
 'SUPP_CPU_Audit_RESULTS_ONLY.zip':('SUPP_CPU_Audit','audit_status.json'),
 'SUPP_XAI_RESULTS_ONLY.zip':('SUPP_XAI','all_status.json'),
 'SUPP_Dose_Response_RESULTS_ONLY.zip':('SUPP_Dose_Response','dose_progress.json'),
}
SCOPE={
 'schema':'final_report_scope_decision_v1','date':'2026-11-17',
 'decision':'Omit optional manual eye-bar/watermark annotation and targeted artifact-masking experiment.',
 'authority':'User explicit scope decision in this chat; reporting change only.',
 'targeted_artifact_testing':SCOPE_STATUS,'manual_artifact_annotation_required':False,
 'artifact_notebook_required':False,'artifact_result_zip_required':False,
 'required_result_archives':list(SIX_RESULTS),'required_result_sets':6,
 'original_supplemental_manifest_sha256':PARENT_SHA,'original_extension_manifest_sha256':EXTENSION_SHA,
 'unchanged_experiments':['Current GPU Ensemble Validation/recovery','Friend2 additional contamination doses','Completed main and supplemental training'],
 'manuscript_limitation':LIMITATION,
}

def inventory(roots):
    rows=[]
    for archive,(folder,marker) in SIX_RESULTS.items():
        found={}
        for root in roots:
            for p in Path(root).rglob(folder):
                if not p.is_dir() or 'xai_recovery_import' in p.parts:continue
                status=p/marker
                if status.exists():
                    value=json.loads(status.read_text());found.setdefault(digest(value),[]).append(status)
        if len(found)>1:raise ValueError('Conflicting result versions for '+archive+'. Attach only the final cumulative result.')
        paths=next(iter(found.values()),[]);value=json.loads(paths[0].read_text()) if paths else {}
        complete=False
        if folder.endswith('_AUTO'):
            counts={'SUPP_alve_AUTO':(50,100),'SUPP_friend1_AUTO':(31,56),'SUPP_friend2_AUTO':(29,54)}[folder]
            complete=value.get('status')=='complete' and value.get('completed_jobs')==counts[0] and value.get('selected_weights' if folder=='SUPP_alve_AUTO' else 'weights_registered')==counts[1]
        elif folder=='SUPP_CPU_Audit':
            complete=value.get('images_verified')==755 and bool(paths) and (paths[0].parent/'duplicate_candidates_REVIEW_REQUIRED.csv').is_file()
            if complete:complete=sha(paths[0].parent/'duplicate_candidates_REVIEW_REQUIRED.csv')==value.get('candidates_sha256')
        elif folder=='SUPP_XAI':complete=value.get('status')=='complete' and value.get('mode')=='all' and value.get('analysis_plan_sha256')=='7823124dd464c09f09506973ee37946e8304fc70f996d5e74fa4341a7f0f8c95'
        else:complete=value.get('status')=='complete' and value.get('completed_jobs')==10 and value.get('extension_manifest_sha256')==EXTENSION_SHA
        rows.append({'archive':archive,'present':bool(paths),'source_status_complete':bool(complete),
            'source_status_sha256':sha(paths[0]) if paths else None,'source_status_file':str(paths[0]) if paths else None})
    return rows

def scope_completeness(base,missing_dose,source_inventory):
    dose_complete=not missing_dose
    additional={'dose_new_conditions_complete':10-len(missing_dose),'dose_new_conditions_required':10,
        'artifact_reliance':SCOPE_STATUS,'targeted_artifact_testing':SCOPE_STATUS,
        'manual_artifact_annotation_required':False,'artifact_result_zip_required':False,
        'required_additional_checks_complete':dose_complete,'extension_manifest_sha256':EXTENSION_SHA}
    missing=[]
    for row in source_inventory:
        if not row['present']:missing.append('Missing result set: '+row['archive'])
        elif not row['source_status_complete']:missing.append('Incomplete/unverified source status: '+row['archive'])
    if not base.get('training_complete') or base.get('completed_new_A_jobs')!=100 or base.get('completed_ablation_jobs')!=10:
        missing.append('Original supplemental training: require100 new A jobs and10 matched ablation conditions.')
    for field in ('XAI_sensitivity','XAI_randomization'):
        if base.get(field)!='complete':missing.append(field+' is not complete.')
    if not dose_complete:missing.append(f'Additional dose conditions: {10-len(missing_dose)}/10 complete.')
    result={k:v for k,v in base.items() if k not in ('additional_checks','both_additional_checks_complete')}
    result.update(report_scope='six_result_zip_scope_revision_v1',report_scope_decision_sha256=digest(SCOPE),
        targeted_artifact_testing=SCOPE_STATUS,manual_artifact_annotation_required=False,
        required_result_sets=6,result_sets_present=sum(r['present'] for r in source_inventory),
        additional_checks=additional,required_experiments_complete=not missing,
        missing_required_analyses=missing,report_scope_status='complete' if not missing else 'incomplete_required_analyses')
    return result

def annotate_report_copies(report):
    """Only generated report views change; source status JSON stays untouched."""
    changed=[]
    for p in (Path(report)/'XAI').rglob('status.json'):
        v=json.loads(p.read_text());before=sha(p)
        v['targeted_artifact_masks']=SCOPE_STATUS
        v['reporting_scope_annotation']={'decision_sha256':digest(SCOPE),'report_status_before_overlay_sha256':before,
            'source_experiment_outputs_modified':False,'required_for_report_completion':False}
        write_json(p,v);changed.append(str(p.relative_to(report)))
    return changed

def check_reporting_scope_inputs(search):
    if any(Path(search).rglob('SUPP_Artifact_Reliance*RESULTS_ONLY.zip')) or any(Path(search).rglob('artifact_complete.json')):
        raise ValueError('This report uses six result sets and excludes targeted-artifact results. Attach a fresh six-ZIP results dataset; preserve older datasets separately.')

def main(args):
    verify_inputs(args.inputs,PARENT_SHA);verify_extension(args.extension,EXTENSION_SHA)
    check_reporting_scope_inputs(args.input_root)
    roots=extract_results(args.input_root,Path(args.work)/'scope_result_cache')
    source_inventory=inventory(roots)
    absent=[r['archive'] for r in source_inventory if not r['present']]
    if absent:raise FileNotFoundError('Attach all six final result ZIPs (raw or Kaggle-expanded). Missing: '+', '.join(absent))
    print('PREFLIGHT PASS: six result sets found; manual artifact annotation and targeted masking are not required.',flush=True)
    for row in source_inventory:print(row['archive'],'source complete:',row['source_status_complete'],flush=True)
    if args.check_only:return
    report=Path(args.work)/'SUPP_Final_Report'
    fingerprint=digest({'scope':SCOPE,'reporting_script':sha(__file__),
        'base_report_script':sha(original.__file__),'dose_report_script':sha(dose_reporting.__file__)})
    identity_path=report/'REPORT_SCOPE_IDENTITY.json'
    if report.exists() and any(report.iterdir()):
        if not identity_path.exists() or json.loads(identity_path.read_text()).get('fingerprint')!=fingerprint:
            raise ValueError('Use a fresh notebook session/output directory. An existing report uses a different reporting scope; it will not be overwritten.')
    report.mkdir(parents=True,exist_ok=True)
    write_json(identity_path,{'fingerprint':fingerprint,'scope_decision_sha256':digest(SCOPE),'reporting_script_sha256':sha(__file__)})
    write_json(report/'REPORT_SCOPE_DECISION.json',SCOPE)
    try:
        # Execute the unchanged, previously verified base analysis. No training,
        # annotation or artifact-detection code runs in this reporting overlay.
        original.main(argparse.Namespace(inputs=args.inputs,expected_sha=PARENT_SHA,input_root=args.input_root,work=args.work))
        records,missing=dose_reporting.collect(args.inputs,args.extension,roots,EXTENSION_SHA)
        tables=dose_reporting.dose_tables(records)
        dest=report/'AdditionalChecks';dest.mkdir(exist_ok=True)
        export_tables(tables,dest/'Dose_Response/tables')
        dose_reporting.figures(tables,None,dest/'figures')
        pd.DataFrame(missing).to_csv(dest/'MISSING_DOSE_JOBS.csv',index=False)
        base=json.loads((report/'COMPLETENESS.json').read_text());status=scope_completeness(base,missing,source_inventory)
        write_json(dest/'ADDITIONAL_COMPLETENESS.json',status['additional_checks'])
        write_json(report/'COMPLETENESS.json',status)
        write_json(report/'MISSING_REQUIRED_ANALYSES.json',{'missing_required_analyses':status['missing_required_analyses'],
            'targeted_artifact_testing':SCOPE_STATUS,'artifact_testing_required':False})
        pd.DataFrame(source_inventory).to_csv(report/'REQUIRED_RESULT_INVENTORY.csv',index=False)
        altered=annotate_report_copies(report)
        write_json(report/'REPORTING_SCOPE_PROVENANCE.json',{'fingerprint':fingerprint,'generated_XAI_status_views_annotated':altered,
            'source_experiment_inputs_code_identities_and_results_modified':False,'frozen_plan_is_retained_as_history':True,
            'scope_decision_takes_precedence_for_reporting_optional_artifact_branch':True})
        (report/'MANUSCRIPT_ARTIFACT_LIMITATION.md').write_text(LIMITATION+'\n',encoding='utf-8')
        (dest/'INTERPRETATION.md').write_text('The additional two- and four-relative conditions retain the frozen dose design, matched cohort, initialization, recipe and training budget. Report all four levels (0,2,4,8), all five seeds and the original uncertainty/multiplicity definitions. These are supplemental internal sensitivity analyses.\n\n'+LIMITATION+'\n\nTargeted artifact testing status: '+SCOPE_STATUS+'. This is not an unfinished required experiment. External validation, authoritative patient identities, expert lesion masks, unfinished duplicate adjudication and historical dataset reuse remain limitations.\n',encoding='utf-8')
        with (report/'INTERPRETATION.md').open('a',encoding='utf-8') as f:
            f.write('\n## Reporting scope decision, 17 November 2026\n\n'+LIMITATION+'\n\nTargeted artifact testing: **'+SCOPE_STATUS+'**. Reviewing all755 images for source marks and running the optional artifact-reliance notebook are not required. Original experimental records remain unchanged.\n')
        print(json.dumps(status,indent=2,ensure_ascii=False),flush=True)
    except Exception as error:
        marker=report/'COMPLETENESS.json';status=json.loads(marker.read_text()) if marker.exists() else {}
        status.update(required_experiments_complete=False,report_scope_status='failed',targeted_artifact_testing=SCOPE_STATUS,
            manual_artifact_annotation_required=False,reporting_error=str(error))
        write_json(marker,status);raise
    finally:bundle(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');p.add_argument('--check-only',action='store_true');main(p.parse_args())
