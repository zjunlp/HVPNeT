# HVPNet

Code for the NAACL2022 (Findings) paper "[Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction](https://arxiv.org/pdf/2205.03521.pdf)".

Model Architecture
==========
<div align=center>
<img src="resource/model.png" width="80%" height="80%" />
</div>
The overall architecture of our hierarchical modality fusion network.


Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

Data Preprocess
==========
To extract visual object images, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Taking the twitter2015 dataset as an example, the extracted objects are stored in `twitter2015_aux_images`. The images of the object obey the following naming format: `imgname_pred_yolo_crop_num.png`, where `imgname` is the name of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit. (Note that in `train/val/test.txt`, text and raw image have a one-to-one relationship, so the `imgname` can be used as a unique identifier for the raw images)
3. Establishing the correspondence between the raw images and the objects. We construct a dictionary to record the correspondence between the raw images and the objects. Taking `twitter2015/twitter2015_train_dict.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['imgname_pred_yolo_crop_num0.png', 'imgname_pred_yolo_crop_num1.png', ...] }`, where key is the name of raw images, value is a List of the objects.


The detected objects and the dictionary of the correspondence between the raw images and the objects are available in our data links.

The expected structure of files is:


```
HMNeT
 |-- data	# conll2003, mit-movie, mit-restaurant and atis
 |    |-- NER_data
 |    |    |-- twitter2015  # text data
 |    |    |    |-- train.txt
 |    |    |    |-- valid.txt
 |    |    |    |-- test.txt
 |    |    |    |-- twitter2015_train_dict.pth  # {imgname: [object-image]}
 |    |    |    |-- ...
 |    |    |-- twitter2015_images       # raw image data
 |    |    |-- twitter2015_aux_images   # object image data
 |    |    |-- twitter2017
 |    |    |-- twitter2017_images
 |    |-- RE_data
 |    |    |-- img_org          # raw image data
 |    |    |-- img_vg           # object image data
 |    |    |-- txt              # text data
 |    |    |-- ours_rel2id.json # relation data
 |-- models	# models
 |    |-- bert_model.py
 |    |-- modeling_bert.py
 |-- modules
 |    |-- metrics.py    # metric
 |    |-- train.py  # trainer
 |-- processor
 |    |-- dataset.py    # processor, dataset
 |-- logs     # code logs
 |-- run.py   # main 
 |-- run_ner_task.sh
 |-- run_re_task.sh
```

Train
==========

## NER Task

The data path and GPU related configuration are in the `run.py`. To train ner model, run this script.

```shell
bash run_twitter15.sh
bash run_twitter17.sh
```

checkpoints can be download via [Twitter15_ckpt](https://drive.google.com/file/d/1E6ed_V2aGAPLExYAF3C3G8j8Td0v-7BM/view?usp=sharing), [Twitter17_ckpt](https://drive.google.com/file/d/1sgsjx_JVMYfu-_95e_3hB8tTR-NUiD27/view?usp=sharing).

## RE Task

To train re model, run this script.

```shell
bash run_re_task.sh
```
checkpoints can be download via [re_ckpt](https://drive.google.com/file/d/1x-yPYy8pjhsDzhhLLzLzEjyVFeQ063HM/view?usp=sharing)

Test
==========
## NER Task

To test ner model, you can download the model chekpoints we provide via [Twitter15_ckpt](https://drive.google.com/file/d/1E6ed_V2aGAPLExYAF3C3G8j8Td0v-7BM/view?usp=sharing), [Twitter17_ckpt](https://drive.google.com/file/d/1sgsjx_JVMYfu-_95e_3hB8tTR-NUiD27/view?usp=sharing) or use your own tained model and set `load_path` to the model path, then run following script:

```shell
python -u run.py \
      --dataset_name="twitter15/twitter17" \
      --bert_name="bert-base-uncased" \
      --seed=1234 \
      --only_test \
      --max_seq=80 \
      --use_prompt \
      --prompt_len=4 \
      --sample_ratio=1.0 \
      --load_path='your_ner_ckpt_path'

```

## RE Task

To test re model, you can download the model chekpoints we provide via [re_ckpt](https://drive.google.com/file/d/1x-yPYy8pjhsDzhhLLzLzEjyVFeQ063HM/view?usp=sharing) or use your own tained model and set `load_path` to the model path, then run following script:

```shell
python -u run.py \
      --dataset_name="MRE" \
      --bert_name="bert-base-uncased" \
      --seed=1234 \
      --only_test \
      --max_seq=80 \
      --use_prompt \
      --prompt_len=4 \
      --sample_ratio=1.0 \
      --load_path='your_re_ckpt_path'

```

Acknowledgement
==========

The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.

The acquisition of MNRE data for multimodal relation extraction task refer to the code from [MEGA](https://github.com/thecharm/Mega), many thanks.

Papers for the Project & How to Cite
==========


If you use or extend our work, please cite the paper as follows:

```bibtex
@article{DBLP:journals/corr/abs-2205-03521,
  author    = {Xiang Chen and
               Ningyu Zhang and
               Lei Li and
               Yunzhi Yao and
               Shumin Deng and
               Chuanqi Tan and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  title     = {Good Visual Guidance Makes {A} Better Extractor: Hierarchical Visual
               Prefix for Multimodal Entity and Relation Extraction},
  journal   = {CoRR},
  volume    = {abs/2205.03521},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.03521},
  doi       = {10.48550/arXiv.2205.03521},
  eprinttype = {arXiv},
  eprint    = {2205.03521},
  timestamp = {Wed, 11 May 2022 17:29:40 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-03521.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
