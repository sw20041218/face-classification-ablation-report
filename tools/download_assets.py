"""Download public source data and official torchvision ImageNet V1 weights."""
import argparse,hashlib,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset',action='store_true');p.add_argument('--weights',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
def fetch(url,path,prefix=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  temp=path.with_suffix('.download');urllib.request.urlretrieve(url,temp);temp.replace(path)
 if prefix:
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if not digest.startswith(prefix):raise RuntimeError('SHA256 mismatch: '+str(path))
 print(path.name,'ready')
if a.dataset:
 commit='a6bfe4e6e8a75a9ca1f4e216ef96be7253efdbb3'
 path=root/'full31022/mlcnn-master.zip'
 fetch('https://codeload.github.com/galeone/mlcnn/zip/'+commit,path)
 with zipfile.ZipFile(path) as z:
  if z.comment.decode()!=commit:raise RuntimeError('Unexpected dataset source commit')
if a.weights:
 dest=Path.home()/'.cache/torch/hub/checkpoints'
 for fn in ['resnet18-f37072fd.pth','resnet50-0676ba61.pth']:
  fetch('https://download.pytorch.org/models/'+fn,dest/fn,fn.split('-')[1].split('.')[0])
