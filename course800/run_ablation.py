"""Fit E0-E10 on 800 labeled training images and predict unlabeled course tests.
Historical calibration constants are replayed; no test labels or local test metrics.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,joblib
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from face_model import unit,contrast,hog,features,flip
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs');args=p.parse_args()
O=args.out;O.mkdir(parents=True,exist_ok=True);threadpool_limits(4)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def log(s):print(s,flush=True)
x=loadmat(args.data/'train(Task_2)/train_data.mat')['train_data'];y=loadmat(args.data/'train(Task_2)/train_label.mat')['train_label'].ravel().astype(int)
t=loadmat(args.data/'test(Task_2)/test_data.mat')['test_data']
assert x.shape==(800,361) and t.shape==(420,361) and set(y)=={-1,1}
assert np.isfinite(x).all() and np.isfinite(t).all()
cfg={'members':[{'C':100.,'gamma':.1,'weight':4.},{'C':1.,'gamma':2.,'weight':.5}],'weights':[.4,.6]}
frozen={x['id']:x for x in json.loads((Path(__file__).parent/'frozen_calibration.json').read_text(encoding='utf-8'))}
keys=['train','train_mirror','test','test_mirror'];raws=dict(zip(keys,[x,flip(x),t,flip(t)]))
trad={name:{k:(features(a,'hog_1.0') if name=='full' else hog(a) if name=='hog' else contrast(a) if name=='contrast' else None) for k,a in raws.items()} for name in ['full','hog','contrast','none']}
sources={}
for name,fn in [('r50_160','resnet50_160'),('r50_96','resnet50_96'),('r18_160','resnet18_160')]:
 with np.load(Path(__file__).parent/'features'/(fn+'.npz')) as z:sources[name]={k:z[k].copy() for k in keys}
sources['r50_multi']={k:unit(np.concatenate([sources['r50_160'][k],sources['r50_96'][k]],axis=1)) for k in keys}
modeldir=O/'models';modeldir.mkdir(exist_ok=True);scoredir=O/'scores';scoredir.mkdir(exist_ok=True)
fitinfo=[];branchcache={}
def fitbranch(branch,traditional='full',source=None,augment=True):
 row=cfg['members'][0 if branch=='A' else 1]
 source=source or ('r50_multi' if branch=='A' else 'r18_160')
 name=f'{branch}_{traditional}_{source}_'+('mirrortrain' if augment else 'originaltrain')
 if name in branchcache:return branchcache[name]
 mp=modeldir/(name+'.joblib');sp=scoredir/(name+'.npz');jp=scoredir/(name+'.json')
 if sp.exists() and jp.exists() and mp.exists():
  z=np.load(sp);info=json.loads(jp.read_text());fitinfo.append(info);branchcache[name]=(z['original'],z['mirror'],info);log('RESUME '+name);return branchcache[name]
 start=time.perf_counter();a=sources[source]
 f={k:(a[k] if traditional=='none' else unit(np.concatenate([trad[traditional][k],row['weight']*a[k]],axis=1))) for k in keys}
 train=np.concatenate([f['train'],f['train_mirror']]) if augment else f['train']
 train_y=np.tile(y,2) if augment else y
 sw=np.full(len(train),.5 if augment else 1.)
 def kernel(z):return np.exp(-row['gamma']*np.maximum(2-2*linear_kernel(z,train),0))
 m=SVC(C=row['C'],kernel='precomputed',tol=1e-4,max_iter=100000,cache_size=512)
 log(f'FIT {name} shape={train.shape}')
 m.fit(kernel(train),train_y,sample_weight=sw);assert m.fit_status_==0
 so=m.decision_function(kernel(f['test']));sm=m.decision_function(kernel(f['test_mirror']))
 info=dict(name=name,branch=branch,traditional=traditional,source=source,augment=augment,n_train_views=len(train),dimension=train.shape[1],C=row['C'],gamma=row['gamma'],transfer_amplitude=row['weight'] if traditional!='none' else None,support_vectors=len(m.support_),n_iter=m.n_iter_.tolist(),fit_status=int(m.fit_status_),elapsed_seconds=time.perf_counter()-start)
 joblib.dump(dict(model=m,config=info,train_feature_storage='Reconstruct using source caches and run_ablation.py'),mp,compress=3)
 np.savez_compressed(sp,original=so,mirror=sm);dump(jp,info);fitinfo.append(info)
 branchcache[name]=(so,sm,info);log(f'DONE {name} seconds={info["elapsed_seconds"]:.2f}');return branchcache[name]

specs=[
 ('E0','完整模型',{}),('E1','去掉对比度',{'traditional':'hog'}),('E2','去掉HOG',{'traditional':'contrast'}),('E3','去掉全部传统特征',{'traditional':'none'}),
 ('E4','分支A仅160尺度',{'sourceA':'r50_160'}),('E5','分支A仅96尺度',{'sourceA':'r50_96'}),('E6','仅分支A',{'single':'A'}),('E7','仅分支B',{'single':'B'}),
 ('E8','不使用镜像训练增强',{'augment':False}),('E9','不使用镜像推理',{'inference':'original'}),('E10','两分支等权融合',{'weights':[.5,.5]})]

for eid,title,opt in specs:
 members=[opt['single']] if 'single' in opt else ['A','B'];ss=[]
 for b in members:
  so,sm,_=fitbranch(b,opt.get('traditional','full'),opt.get('sourceA') if b=='A' else None,opt.get('augment',True))
  ss.append(so if opt.get('inference')=='original' else (so+sm)/2)
 c=frozen[eid];score=sum(w*a/sd for w,a,sd in zip(c['weights'],ss,c['score_std']))
 pred=np.where(score>c['threshold'],1,-1)
 np.savez_compressed(scoredir/(eid+'.npz'),score=score,pred=pred)
 (O/(eid+'_predictions.txt')).write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii'))
 dump(O/(eid+'_configuration.json'),c);log(eid+' predictions written; submit to platform for ACC')
dump(O/'fit_summary.json',fitinfo)
