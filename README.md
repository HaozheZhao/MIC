# vl-blip2
 tuning the bilp2 adapt to multiple images and few-shot settings

## Reproduce Tips
Since experiments reported in our paper are all conducted on NVIDIA DGX-A40 servers (which might be difficult to acquire), 
we reimplement vl-blip2 with:

* Ubuntu servers with 6* NVIDIA GeForce A40 (46G) GPUs
* cuda 11.3
* packages with certain versions (provided below)

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for vl-blip2:

```shell
conda create -n vlprompt python=3.8.5
conda activate vlprompt
```

After we setup basic conda environment, install pytorch related packages via:

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Data
Using multiple data source such as: VCR, VQA, GQA, COCO, NLVR2, OKVQK,FILCKR, STVQA
We tranform it into few shot style and stored it into jsonl files:
It forms the all data in to multi instruction style with zero to few-shot form data.
runing the preprocessed script and change the data into raw arrow file for further training:
```shell
python data_preprocess.py
```
### Training
Run training scripts in [run_script](run_script) :

```shell
bash run_script/flickr/run.sh
```
