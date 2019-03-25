# Running pytorch on GPUs

## Get Data

Download data to './data/' directory from bash:
```bash
wget -O zip.train.gz https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz
wget -O zip.test.gz https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz
```

Unzip downloaded files:

```bash
unzip zip.train.gz
unzip zip.test.gz
```

## Setup environment

Activate preinstalled pytorch environment on aws deeplearning gpu instance 
```bash
  source activate pytorch_p36 
```

In project dir setup project (add package to environment)
```bash
 python setup.py install --user
```