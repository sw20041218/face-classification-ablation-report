from pathlib import Path
import io,json,zipfile,hashlib,collections
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
O=Path(__file__).parent;dest=O/'dataset';dest.mkdir(exist_ok=True)
z=zipfile.ZipFile(O/'mlcnn-master.zip');root=dest.resolve();manifest=[];arrays={};pixel_hashes={};group_hashes={}
def sha(b):return hashlib.sha256(b).hexdigest()
for split,expected in [('train',6977),('test',24045)]:
 files=sorted(n for n in z.namelist() if f'/mitcbcl/{split}/' in n and n.endswith('.pgm'))
 assert len(files)==expected
 images=np.empty((expected,19,19),dtype=np.uint8);labels=[];hs=[];gh=[]
 for i,n in enumerate(files):
  rel=Path(*Path(n).parts[2:]);p=(dest/rel).resolve();assert root in p.parents
  b=z.read(n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  im=np.asarray(Image.open(io.BytesIO(b)),dtype=np.uint8);assert im.shape==(19,19)
  y=1 if '/face/' in n else -1
  images[i]=im;labels.append(y);h=sha(im.tobytes());hs.append(h);gh.append(min(h,sha(im[:,::-1].tobytes())))
  manifest.append(dict(split=split,row=i+1,path=str(rel).replace('\\','/'),label=y,sha256=sha(b),pixel_sha256=h))
 arrays[split]=(images,np.array(labels,dtype=np.int8));pixel_hashes[split]=hs;group_hashes[split]=gh
 np.save(dest/(split+'_images.npy'),images);np.save(dest/(split+'_labels.npy'),arrays[split][1]);print(split,images.shape,collections.Counter(labels),flush=True)
readme=z.read(next(n for n in z.namelist() if n.endswith('/mitcbcl/README')));(dest/'README').write_bytes(readme)
train_y=arrays['train'][1];groups=np.array(group_hashes['train'])
fit,val=next(StratifiedGroupKFold(5,shuffle=True,random_state=20261115).split(np.zeros(len(train_y)),train_y,groups))
assert not set(groups[fit])&set(groups[val])
np.savez(O/'train_validation_split.npz',fit=fit,validation=val)
audit=dict(train=6977,test=24045,total=31022,train_faces=int(sum(train_y==1)),train_nonfaces=int(sum(train_y==-1)),test_faces=int(sum(arrays['test'][1]==1)),test_nonfaces=int(sum(arrays['test'][1]==-1)),fit_count=len(fit),validation_count=len(val),fit_faces=int(sum(train_y[fit]==1)),validation_faces=int(sum(train_y[val]==1)),split_seed=20261115,split_method='First fold of StratifiedGroupKFold(5), groups are exact original-or-horizontal-mirror pixel hashes',train_duplicate_rows=len(groups)-len(set(pixel_hashes['train'])),test_duplicate_rows=24045-len(set(pixel_hashes['test'])),train_test_exact_overlap=len(set(pixel_hashes['train'])&set(pixel_hashes['test'])),train_test_original_or_mirror_overlap=len(set(groups)&set(group_hashes['test'])),manifest_sha256=None,limitations='Official train/test split retained; no near-duplicate or identity audit. Historical method development used a 420-sample subset of this source, so official test is not claimed as never previously seen.')
(O/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8');audit['manifest_sha256']=sha((O/'dataset_manifest.json').read_bytes())
(O/'dataset_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit),flush=True)
