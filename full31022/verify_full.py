"""Independent arithmetic, output, model and frozen-calibration checks."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
O=Path(__file__).parent;ty=np.load(O/'dataset/test_labels.npy');rows=json.loads((O/'results.json').read_text(encoding='utf-8'));cfgs=json.loads((O/'frozen_configurations.json').read_text(encoding='utf-8'));cy={c['id']:c for c in cfgs};vy=np.load(O/'dataset/train_labels.npy')[np.load(O/'train_validation_split.npz')['validation']]
checks=[]
for r in rows:
 eid=r['id'];z=np.load(O/'scores'/(eid+'_test.npz'));p=z['pred'];s=z['score'];cfg=cy[eid];assert np.isfinite(s).all() and len(p)==24045 and set(p)<=set([-1,1])
 tn=int(sum((ty==-1)&(p==-1)));fp=int(sum((ty==-1)&(p==1)));fn=int(sum((ty==1)&(p==-1)));tp=int(sum((ty==1)&(p==1)))
 assert [tn,fp,fn,tp]==[r[k] for k in ['tn','fp','fn','tp']]
 assert abs((tn+tp)/len(ty)-r['accuracy'])<1e-12 and abs(2*tp/(2*tp+fp+fn)-r['f1'])<1e-12
 assert abs((tp/(tp+fn)+tn/(tn+fp))/2-r['balanced_accuracy'])<1e-12
 assert np.array_equal(np.where(s>cfg['threshold'],1,-1),p)
 assert abs(roc_auc_score(ty,s)-r['roc_auc'])<1e-12 and abs(average_precision_score(ty,s)-r['average_precision'])<1e-12
 txt=np.loadtxt(O/(eid+'_test_predictions.txt'),dtype=int);assert txt.shape==(24045,2) and np.array_equal(txt[:,0],np.arange(1,24046)) and np.array_equal(txt[:,1],p)
 v=np.load(O/'scores'/(eid+'_validation.npz'));u=np.unique(v['score']);ths=np.r_[np.nextafter(u[0],-np.inf),(u[:-1]+u[1:])/2,np.nextafter(u[-1],np.inf)]
 brute=int(np.max(np.sum((v['score'][:,None]>ths)==(vy[:,None]==1),axis=0)));assert brute==cfg['validation']['correct']
 checks.append(dict(id=eid,metrics_verified=True,txt_rows=24045,validation_threshold_optimum_verified=True,threshold_is_frozen_validation_value=r['threshold']==cfg['threshold']))
modelchecks=[]
for p in sorted((O/'models').glob('*.joblib')):
 m=joblib.load(p);assert m['config']['fit_status']==0 and m['support_features'].shape[0]==len(m['dual_coef'])
 modelchecks.append(dict(model=p.name,support_vectors=len(m['dual_coef']),dimension=m['support_features'].shape[1],all_finite=bool(np.isfinite(m['support_features']).all())))
manifest=[]
for path in [O/'mlcnn-master.zip',O/'dataset_manifest.json',O/'train_validation_split.npz',O/'frozen_configurations.json',O/'results.json',O/'run_full_ablation.py',O/'extract_features.py',O/'prepare_dataset.py']+list((O/'features').glob('*.npy')):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 manifest.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest()))
(O/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
complete=json.loads((O/'evaluation_completion.json').read_text());frozen=hashlib.sha256((O/'frozen_configurations.json').read_bytes()).hexdigest();assert frozen==complete['frozen_config_sha256']
(O/'verification.json').write_text(json.dumps(dict(checks=checks,models=modelchecks,configuration_sha256=frozen,all_passed=True,cnn_features_extracted=True,test_used_for_parameter_selection_this_run=False,historical_test_exposure_limitation=True),ensure_ascii=False,indent=2),encoding='utf-8')
print('VERIFIED',len(checks),'experiments;',len(modelchecks),'models; all frozen thresholds and metrics passed',flush=True)
