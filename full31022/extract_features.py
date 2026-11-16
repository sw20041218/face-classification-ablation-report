"""Frozen CNN extraction, resumable; does not read any label arrays."""
import argparse,hashlib,json,time,sys,os
from pathlib import Path
import numpy as np,torch,torchvision
from torch.nn import functional as F
from torchvision.models import resnet18,resnet50
P=argparse.ArgumentParser();P.add_argument('--benchmark',action='store_true');P.add_argument('--batch',type=int,default=16);args=P.parse_args()
O=Path(__file__).parent;C=O/'features';C.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
device='cuda' if torch.cuda.is_available() else 'cpu'
assert device=='cuda','GPU required for the planned full extraction'
ims=[np.load(O/'dataset'/f'{split}_images.npy',mmap_mode='r') for split in ['train','test']];n=sum(len(x) for x in ims)
def getbatch(start,end):
 cut=len(ims[0])
 a=ims[0][start:min(end,cut)] if start<cut else np.empty((0,19,19),dtype=np.uint8)
 b=ims[1][max(0,start-cut):max(0,end-cut)] if end>cut else np.empty((0,19,19),dtype=np.uint8)
 return np.concatenate([a,b],axis=0)
def unit(a):return a/torch.clamp(torch.linalg.vector_norm(a,dim=1,keepdim=True),min=1e-12)
def extract(model,q):
 q=model.maxpool(model.relu(model.bn1(model.conv1(q))));q=model.layer1(q);q=model.layer2(q);a=unit(F.adaptive_avg_pool2d(q,(2,2)).flatten(1));q=model.layer3(q);b=unit(F.adaptive_avg_pool2d(q,(2,2)).flatten(1));q=model.layer4(q);c=unit(F.adaptive_avg_pool2d(q,(1,1)).flatten(1));return unit(torch.cat([a,b,c],dim=1))
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
logs=O/'extraction.log'
def log(s):print(s,flush=True);logs.open('a',encoding='utf-8').write(s+'\n')
env=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,cuda_build=torch.version.cuda,device=device,gpu=torch.cuda.get_device_name(0),batch_original=args.batch,forward_batch_including_mirrors=args.batch*2,precision='float32; TF32 disabled',total_original_images=n)
(O/'feature_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
with torch.inference_mode():
 for arch,size,dim,fn in [('resnet18',160,2048,'resnet18-f37072fd.pth'),('resnet50',96,8192,'resnet50-0676ba61.pth'),('resnet50',160,8192,'resnet50-0676ba61.pth')]:
  name=f'{arch}_{size}';wp=Path.home()/'.cache/torch/hub/checkpoints'/fn
  model=(resnet18(weights=None) if arch=='resnet18' else resnet50(weights=None));model.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True));model=model.eval().to(device)
  progress=C/(name+'_progress.json');done=json.loads(progress.read_text())['done'] if progress.exists() else 0
  if done==n:log('SKIP complete '+name);continue
  arrays=[np.lib.format.open_memmap(C/(name+'_'+key+'.npy'),mode='r+' if done else 'w+',dtype='float32',shape=(n,dim)) for key in ['original','mirror']]
  start=time.perf_counter();start_i=done
  for i in range(done,n,args.batch):
   j=min(i+args.batch,n);a=getbatch(i,j);q=torch.from_numpy(a.copy()).to(device,dtype=torch.float32)[:,None]/255
   q=F.interpolate(q,size=(size,size),mode='bilinear',align_corners=False).repeat(1,3,1,1);q=torch.cat([q,q.flip(-1)],dim=0);q=(q-mean)/std
   f=extract(model,q).cpu().numpy();assert np.isfinite(f).all();arrays[0][i:j]=f[:j-i];arrays[1][i:j]=f[j-i:]
   if args.benchmark and j>=args.batch*4:
    torch.cuda.synchronize();log(json.dumps(dict(benchmark=name,images=j,seconds=time.perf_counter()-start,images_per_second=j/(time.perf_counter()-start),peak_cuda_MB=torch.cuda.max_memory_allocated()/2**20)));break
   if (j//args.batch)%32==0 or j==n:
    for ar in arrays:ar.flush()
    info=dict(done=j,total=n,arch=arch,size=size,dimension=dim,weight_file=str(wp),weight_sha256=hashlib.sha256(wp.read_bytes()).hexdigest(),precision='float32',seconds_current_session=time.perf_counter()-start)
    progress.write_text(json.dumps(info,indent=2),encoding='utf-8')
    rate=(j-start_i)/max(time.perf_counter()-start,1e-9);log(f'{name} {j}/{n} {rate:.1f} images/s remaining={(n-j)/rate/60:.1f} min')
  del model,arrays;torch.cuda.empty_cache()
  if args.benchmark:break
log('BENCHMARK COMPLETE' if args.benchmark else 'ALL FEATURES COMPLETE')
