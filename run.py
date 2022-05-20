import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import HMNeTREModel, HMNeTNERModel
from processor.dataset import MMREProcessor, MMPNERProcessor, MMREDataset, MMPNERDataset
from modules.train import RETrainer, NERTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'MRE': HMNeTREModel,
    'twitter15': HMNeTNERModel,
    'twitter17': HMNeTNERModel
}

TRAINER_CLASSES = {
    'MRE': RETrainer,
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}
DATA_PROCESS = {
    'MRE': (MMREProcessor, MMREDataset),
    'twitter15': (MMPNERProcessor, MMPNERDataset), 
    'twitter17': (MMPNERProcessor, MMPNERDataset)
}

DATA_PATH = {
    'MRE': {
            # text data
            'train': 'data/RE_data/txt/ours_train.txt',    
            'dev': 'data/RE_data/txt/ours_val.txt',
            'test': 'data/RE_data/txt/ours_test.txt',
            # {data_id : object_crop_img_path}
            'train_auximgs': 'data/RE_data/txt/mre_train_dict.pth',
            'dev_auximgs': 'data/RE_data/txt/mre_dev_dict.pth',
            'test_auximgs': 'data/RE_data/txt/mre_test_dict.pth',
            # relation json data
            're_path': 'data/RE_data/ours_rel2id.json'
            },
    
    'twitter15': {
                # text data
                'train': 'data/NER_data/twitter2015/train.txt',
                'dev': 'data/NER_data/twitter2015/valid.txt',
                'test': 'data/NER_data/twitter2015/test.txt',
                # {data_id : object_crop_img_path}
                'train_auximgs': 'data/NER_data/twitter2015/twitter2015_train_dict.pth',
                'dev_auximgs': 'data/NER_data/twitter2015/twitter2015_val_dict.pth',
                'test_auximgs': 'data/NER_data/twitter2015/twitter2015_test_dict.pth'
            },

    'twitter17': {
                # text data
                'train': 'data/NER_data/twitter2017/train.txt',
                'dev': 'data/NER_data/twitter2017/valid.txt',
                'test': 'data/NER_data/twitter2017/test.txt',
                # {data_id : object_crop_img_path}
                'train_auximgs': 'data/NER_data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': 'data/NER_data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': 'data/NER_data/twitter2017/twitter2017_test_dict.pth'
            },
        
}

# image data
IMG_PATH = {
    'MRE': {'train': 'data/RE_data/img_org/train/',
            'dev': 'data/RE_data/img_org/val/',
            'test': 'data/RE_data/img_org/test'},
    'twitter15': 'data/NER_data/twitter2015_images',
    'twitter17': 'data/NER_data/twitter2017_images',
}

# auxiliary images
AUX_PATH = {
    'MRE':{
            'train': 'data/RE_data/img_vg/train/crops',
            'dev': 'data/RE_data/img_vg/val/crops',
            'test': 'data/RE_data/img_vg/test/crops'
    },
    'twitter15': {
                'train': 'data/NER_data/twitter2015_aux_images/train/crops',
                'dev': 'data/NER_data/twitter2015_aux_images/val/crops',
                'test': 'data/NER_data/twitter2015_aux_images/test/crops',
            },

    'twitter17': {
                'train': 'data/NER_data/twitter2017_aux_images/train/crops',
                'dev': 'data/NER_data/twitter2017_aux_images/val/crops',
                'test': 'data/NER_data/twitter2017_aux_images/test/crops',
            }
}

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    writer=None

    if not args.use_prompt:
        img_path, aux_path = None, None
        
    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.dataset_name == 'MRE':  # RE task
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer
        model = HMNeTREModel(num_labels, tokenizer, args=args)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, processor=processor, args=args, logger=logger, writer=writer)
    else:   # NER task
        label_mapping = processor.get_label_mapping()
        label_list = list(label_mapping.keys())
        model = HMNeTNERModel(label_list, args)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()
    

if __name__ == "__main__":
    main()