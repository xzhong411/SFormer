# SFormer
The pytorch code for  [Dual Branched SFormer for Effective Weakly Supervised Semantic Segmentation].

#### 1. Download the augmented annotations

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `DataSet/VOC2012`. The directory sctructure should thus be 

``` bash
DataSet/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```


## 2. Create and activate conda environment
- Ubuntu 18.04, with Python 3.6 and the following python dependencies.
```
conda create --name py36 python=3.6
conda activate py36
pip install -r requirements.txt
```

### 3. Clone this repo

```bash
git clone https://github.com/xzhong411/SFormer.git
cd SFormer
```

### Generating CAMs
```bash
bash run.sh
```

### Generating PGT

Subsequent code to be released soon ....


