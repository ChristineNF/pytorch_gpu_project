# Running pytorch on GPUs

## Get Data

Download data to './data/' directory from bash:
```bash
wget -O zip.train.gz https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz
wget -O zip.test.gz https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz
```

Unzip downloaded files:

```bash
gunzip zip.train.gz
gunzip zip.test.gz
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


## Sync repo
Clone project to local machine and sync it to aws instance. Use 'rsync' to synconize code with machine. 


```bash
rsync -avL --progress -e "ssh -i /path/to/mykeypair.pem" \
       ~/path/to/projects/pytorch_gpu_project/* \ 
       root@ec2-XX-XXX-XXX-XXX.eu-central-1.compute.amazonaws.com:pytorch_gpu_project/
```
