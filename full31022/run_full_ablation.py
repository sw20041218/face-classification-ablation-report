"""Full MIT-CBCL ablation: fit/calibrate first, evaluate frozen systems separately."""
import argparse,csv,gc,hashlib,json,platform,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np,scipy,sklearn,joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
O=Path(__file__).parent;R=O.parent
sys.path.insert(0,str(R/'common'))
from face_model import features,hog,contrast,unit
P=argparse.ArgumentParser();P.add_argument('--stage',choices=['fit','evaluate'],required=True);args=P.parse_args();threadpool_limits(4)
for n in ['models','scores']:(O/n).mkdir(exist_ok=True)
TZ=timezone(timedelta(hours=8))
def dump(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
def log(s):
 s=f'{datetime.now(TZ).isoformat()} {s}';print(s,flush=True)
 with (O/'svm_run.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
keys=['resnet18_160','resnet50_96','resnet50_160']
for key in keys:
 pr=json.loads((O/'features'/(key+'_progress.json')).read_text())
 assert pr['done']==31022,'Incomplete CNN cache '+key
cache={name:{mode:np.load(O/'features'/(name+'_'+mode+'.npy'),mmap_mode='r') for mode in ['original','mirror']} for name in keys}
images={sp:np.load(O/'dataset'/(sp+'_images.npy'),mmap_mode='r') for sp in ['train','test']}
y=np.load(O/'dataset/train_labels.npy');split=np.load(O/'train_validation_split.npz');fitidx=split['fit'];validx=split['validation'];vy=y[validx]
trad={}
for sp,im in images.items():
 trad[sp]={}
 for mode in ['original','mirror']:
  a=np.asarray(im[:, :, ::-1] if mode=='mirror' else im).transpose(0,2,1).reshape(-1,361).astype(np.float64)/255
  trad[sp][mode]={'full':features(a,'hog_1.0'),'hog':hog(a),'contrast':contrast(a)}
  del a
params={'A':dict(C=100.,gamma=.1,amplitude=4.),'B':dict(C=1.,gamma=2.,amplitude=.5)}
def feature(b,traditional,source,sp,idx,mode):
 offset=0 if sp=='train' else 6977;ind=idx+offset
 if source=='multi':deep=unit(np.concatenate([cache['resnet50_160'][mode][ind],cache['resnet50_96'][mode][ind]],axis=1))
 else:deep=cache[{'160':'resnet50_160','96':'resnet50_96','r18':'resnet18_160'}[source]][mode][ind]
 return np.asarray(deep,dtype=np.float64) if traditional=='none' else unit(np.concatenate([trad[sp][mode][traditional][idx],params[b]['amplitude']*deep],axis=1))
def kernel(a,b,gamma):return np.exp(-gamma*np.maximum(2-2*(a@b.T),0))
def predict_artifact(model,traditional,source,sp,idx):
 out=[];b=model['config']['branch'];g=model['config']['gamma']
 for mode in ['original','mirror']:
  scores=[]
  for start in range(0,len(idx),256):
   f=feature(b,traditional,source,sp,idx[start:start+256],mode)
   scores.append(kernel(f,model['support_features'],g)@model['dual_coef']+model['intercept'])
  out.append(np.concatenate(scores))
 return out
def best_threshold(s,y):
 order=np.argsort(s);a=s[order];v=y[order];c=np.r_[sum(v==1),sum(v==1)+np.cumsum(np.where(v==-1,1,-1))]
 c=np.where(np.r_[True,a[1:]!=a[:-1],True],c,-1);k=int(np.argmax(c));th=np.nextafter(a[0],-np.inf) if k==0 else np.nextafter(a[-1],np.inf) if k==len(a) else float((a[k-1]+a[k])/2)
 return float(th)
def metrics(s,y,th):
 p=np.where(s>th,1,-1);tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[-1,1]).ravel());N=len(y)
 return dict(n=N,correct=tn+tp,accuracy=(tn+tp)/N,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),tn=tn,fp=fp,fn=fn,tp=tp,threshold=th)
specs=[('E0','完整模型',{}),('E1','去掉对比度',{'traditional':'hog'}),('E2','去掉HOG',{'traditional':'contrast'}),('E3','去掉全部传统特征',{'traditional':'none'}),('E4','分支A仅160尺度',{'sourceA':'160'}),('E5','分支A仅96尺度',{'sourceA':'96'}),('E6','仅分支A',{'single':'A'}),('E7','仅分支B',{'single':'B'}),('E8','不使用镜像训练增强',{'augment':False}),('E9','不使用镜像推理',{'inference':'original'}),('E10','两分支等权融合',{'weights':[.5,.5]})]
def identities(opt):
 members=[opt['single']] if 'single' in opt else ['A','B'];out=[]
 for b in members:
  tr=opt.get('traditional','full');src=opt.get('sourceA','multi') if b=='A' else 'r18';aug=opt.get('augment',True)
  name=f'{b}_{tr}_{src}_'+('mirrortrain' if aug else 'originaltrain');out.append((name,b,tr,src,aug))
 return out
def fit_branch(name,b,tr,src,aug):
 mp=O/'models'/(name+'.joblib');sp=O/'scores'/(name+'_validation.npz')
 if mp.exists() and sp.exists():log('RESUME '+name);return
 start=time.perf_counter();F=feature(b,tr,src,'train',fitidx,'original')
 if aug:F=np.concatenate([F,feature(b,tr,src,'train',fitidx,'mirror')])
 yy=np.tile(y[fitidx],2) if aug else y[fitidx];sw=np.full(len(F),.5 if aug else 1.)
 log(f'FIT {name} n={len(F)} d={F.shape[1]} kernel_GB={len(F)**2*8/1e9:.3f}')
 K=np.empty((len(F),len(F)),dtype=np.float64)
 for i in range(0,len(F),256):K[i:i+256]=kernel(F[i:i+256],F,params[b]['gamma'])
 svc=SVC(C=params[b]['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
 svc.fit(K,yy,sample_weight=sw);assert svc.fit_status_==0
 config=dict(name=name,branch=b,traditional=tr,source=src,augment=aug,C=params[b]['C'],gamma=params[b]['gamma'],amplitude=params[b]['amplitude'],dimension=F.shape[1],n_train_views=len(F),n_support=len(svc.support_),fit_status=int(svc.fit_status_),n_iter=svc.n_iter_.tolist(),fit_sample_count=len(fitidx),validation_sample_count=len(validx))
 artifact=dict(svc=svc,support_features=F[svc.support_].copy(),dual_coef=svc.dual_coef_[0].copy(),intercept=float(svc.intercept_[0]),config=config)
 # Ensure compact support-vector scoring reproduces the original fitted SVC API.
 fi=feature(b,tr,src,'train',validx[:32],'original');check=svc.decision_function(kernel(fi,F,params[b]['gamma']));compact=kernel(fi,artifact['support_features'],params[b]['gamma'])@artifact['dual_coef']+artifact['intercept'];assert np.allclose(check,compact,atol=1e-9,rtol=1e-9)
 del K,F;gc.collect();so,sm=predict_artifact(artifact,tr,src,'train',validx)
 config['elapsed_seconds']=time.perf_counter()-start;joblib.dump(artifact,mp,compress=3)
 loaded=joblib.load(mp);re=predict_artifact(loaded,tr,src,'train',validx[:16]);assert np.allclose(re[0],so[:16],atol=1e-9,rtol=1e-9)
 np.savez_compressed(sp,original=so,mirror=sm);dump(O/'models'/(name+'.json'),config)
 log(f'DONE {name} support={config["n_support"]} seconds={config["elapsed_seconds"]:.1f}')

if args.stage=='fit':
 start=time.perf_counter();unique={a[0]:a for _,_,opt in specs for a in identities(opt)}
 plan=dict(specs=specs,params=params,unique_branches=list(unique),seed=20261115,fit_count=len(fitidx),validation_count=len(validx),training_rule='5580 fitting images; 1397 training-internal calibration images. No final refit on calibration samples; no test labels used for calibration.',threshold_objective='Maximum validation ACC; ties choose first ascending-score split.',scale='Validation score population standard deviation ddof=0; no centering.',weights='Fixed historical 0.4/0.6; E10 uses 0.5/0.5. No new weight search.',protocol='All official 6977 train and 24045 test images accounted for; test used for final evaluation only in this run. Historical 420 source samples influenced method development, so never-seen blind-test independence is not claimed.')
 dump(O/'experiment_plan.json',plan);dump(O/'svm_environment.json',dict(executable=sys.executable,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,platform=platform.platform(),threads=4))
 for a in unique.values():fit_branch(*a)
 configs=[]
 for eid,title,opt in specs:
  ident=identities(opt);ss=[]
  for name,*_ in ident:
   z=np.load(O/'scores'/(name+'_validation.npz'));ss.append(z['original'] if opt.get('inference')=='original' else (z['original']+z['mirror'])/2)
  std=[float(s.std()) for s in ss];weights=opt.get('weights',[.4,.6]) if len(ss)==2 else [1.];s=sum(w*a/d for w,a,d in zip(weights,ss,std));th=best_threshold(s,vy)
  row=dict(id=eid,title=title,options=opt,branches=[a[0] for a in ident],score_std=std,weights=weights,threshold=th,validation=metrics(s,vy,th));configs.append(row)
  np.savez_compressed(O/'scores'/(eid+'_validation.npz'),score=s,pred=np.where(s>th,1,-1));log(f'CALIBRATED {eid} correct={row["validation"]["correct"]}/{len(vy)} threshold={th:.8f}')
 dump(O/'frozen_configurations.json',configs);dump(O/'fit_completion.json',dict(elapsed_seconds=time.perf_counter()-start,models=len(unique),configurations=len(configs),test_labels_read=False))
 log('FIT AND CALIBRATION COMPLETE; configurations frozen before test label evaluation')
else:
 start=time.perf_counter();confpath=O/'frozen_configurations.json';before=hashlib.sha256(confpath.read_bytes()).hexdigest();configs=json.loads(confpath.read_text(encoding='utf-8'));ty=np.load(O/'dataset/test_labels.npy');testidx=np.arange(len(ty));branch_scores={}
 for name in dict.fromkeys(n for c in configs for n in c['branches']):
  sp=O/'scores'/(name+'_test.npz')
  if sp.exists():z=np.load(sp);branch_scores[name]=(z['original'],z['mirror']);continue
  a=joblib.load(O/'models'/(name+'.joblib'));cfg=a['config'];so,sm=predict_artifact(a,cfg['traditional'],cfg['source'],'test',testidx)
  np.savez_compressed(sp,original=so,mirror=sm);branch_scores[name]=(so,sm);log('TEST SCORES '+name)
 result=[];preds={};baseline=configs[0]
 for c in configs:
  ss=[branch_scores[n][0] if c['options'].get('inference')=='original' else sum(branch_scores[n])/2 for n in c['branches']]
  s=sum(w*a/d for w,a,d in zip(c['weights'],ss,c['score_std']));m=metrics(s,ty,c['threshold']);r=dict(id=c['id'],title=c['title'],**m,validation=c['validation'],score_std=c['score_std'],weights=c['weights']);r['delta_pp_vs_full']=100*(r['accuracy']-result[0]['accuracy']) if result else 0.
  if c['id'] in ['E9','E10']:
   fixed=sum(w*a/d for w,a,d in zip(c['weights'],ss,baseline['score_std']));r['frozen_full_calibration_metrics']=metrics(fixed,ty,baseline['threshold'])
  pred=np.where(s>c['threshold'],1,-1);preds[c['id']]=pred
  np.savez_compressed(O/'scores'/(c['id']+'_test.npz'),score=s,pred=pred)
  with (O/(c['id']+'_test_predictions.txt')).open('w',encoding='ascii',newline='\n') as f:
   f.writelines(f'{i} {int(v)}\n' for i,v in enumerate(pred,1))
  result.append(r);log(f'RESULT {c["id"]} correct={m["correct"]}/{len(ty)} fp={m["fp"]} fn={m["fn"]} recall={m["recall"]:.6f}')
 dump(O/'results.json',result)
 fields=['id','title','n','correct','accuracy','delta_pp_vs_full','precision','recall','f1','specificity','balanced_accuracy','roc_auc','average_precision','tn','fp','fn','tp','threshold']
 with (O/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result)
 ca,cb,cf=[preds[k]==ty for k in ['E6','E7','E0']];bestid='E6' if ca.sum()>=cb.sum() else 'E7';best=preds[bestid]==ty
 complement=dict(A_correct=int(ca.sum()),B_correct=int(cb.sum()),both_correct=int(sum(ca&cb)),A_only_correct=int(sum(ca&~cb)),B_only_correct=int(sum(~ca&cb)),both_wrong=int(sum(~ca&~cb)),best_single=bestid,fusion_corrects_best_single=int(sum(~best&cf)),fusion_regresses_best_single=int(sum(best&~cf)),fusion_correct_when_both_wrong=int(sum(~ca&~cb&cf)))
 dump(O/'complementarity.json',complement)
 assert before==hashlib.sha256(confpath.read_bytes()).hexdigest()
 dump(O/'evaluation_completion.json',dict(elapsed_seconds=time.perf_counter()-start,frozen_config_sha256=before,configurations_unchanged=True,official_test_n=len(ty),all_nonface_accuracy=float(np.mean(ty==-1)),test_labels_used_for_selection=False))
 log('FINAL EVALUATION COMPLETE '+json.dumps(complement))
