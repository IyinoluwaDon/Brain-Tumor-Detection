import sys, zipfile, hashlib, json, io
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import argparse
parser=argparse.ArgumentParser(description='Train an educational baseline from the documented Kaggle ZIP')
parser.add_argument('archive',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
archive=args.archive
samples={}
with zipfile.ZipFile(archive) as z:
 print(z.namelist()[:15],flush=True)
 for name in z.namelist():
  if not name.lower().endswith(('.jpg','.jpeg','.png')): continue
  parts=name.lower().split('/')
  label=0 if any(p in {'brain tumor','brain_tumor','yes'} for p in parts) else 1 if any(p in {'healthy','no'} for p in parts) else None
  if label is None: continue
  with Image.open(io.BytesIO(z.read(name))) as im:
   a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255
  key=hashlib.sha256(a.tobytes()).hexdigest()
  if key in samples and samples[key][1]!=label: raise ValueError('Conflicting duplicate labels')
  samples[key]=(a.ravel(),label)
print('Unique images',len(samples),flush=True)
if len(samples)<1000: raise ValueError('Unexpected dataset structure')
X=np.stack([s[0] for s in samples.values()]);y=np.array([s[1] for s in samples.values()])
tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
model=LogisticRegression(C=.1,max_iter=2000,class_weight='balanced').fit(X[tr],y[tr])
pred=model.predict(X[te])
report={'source':'https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset','source_license':'GPL-2.0 (dataset metadata; verify image rights before redistribution)','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'unique_resized_images':len(y),'train':len(tr),'test':len(te),'accuracy':accuracy_score(y[te],pred),'balanced_accuracy':balanced_accuracy_score(y[te],pred),'confusion_matrix_labels':[0,1],'confusion_matrix':confusion_matrix(y[te],pred).tolist(),'labels':{'0':'Brain_Tumor','1':'Healthy'},'limitations':['Educational image-level split; patient identifiers unavailable, so patient independence is not established.','Exact duplicate resized pixels are grouped before splitting; near duplicates may remain.','This is a newly trained 32x32 grayscale logistic baseline, not the missing original CNN.','No clinical validation or external test set.'],'preprocessing':'Pillow grayscale, bilinear resize 32x32, float32 /255','seed':42,'C':.1}
(root/'models').mkdir(exist_ok=True)
np.savez_compressed(root/'models/baseline.npz',coef=model.coef_,intercept=model.intercept_)
(root/'models/evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
