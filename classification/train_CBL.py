import argparse
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
import config as CFG
from dataset_utils import preprocess, train_val_test_split
from modules import CBL, RobertaCBL, GPT2CBL, BERTCBL
from utils import cos_sim_cubed, get_labels, eos_pooling
import time
from dataset_utils import *

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--backbone", type=str, default="roberta", help="roberta or gpt2 or bert")
parser.add_argument('--tune_cbl_only', action=argparse.BooleanOptionalAction)
parser.add_argument('--automatic_concept_correction', action=argparse.BooleanOptionalAction)
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--cbl_only_batch_size", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encode_roberta, s):
        self.encode_roberta = encode_roberta
        self.s = s

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_roberta.items()}
        y = torch.FloatTensor(self.s[idx])

        return t, y

    def __len__(self):
        return len(self.encode_roberta['input_ids'])

def build_loaders(encode_roberta, s, mode):
    dataset = ClassificationDataset(encode_roberta, s)
    if args.tune_cbl_only:
        batch_size = args.cbl_only_batch_size
    else:
        batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print(args)

    print("loading data...")
    train_dataset, val_dataset, test_dataset = train_val_test_split(args.dataset, CFG.dataset_config[args.dataset]["label_column"], ratio=0.2, has_val=False)
    print("tokenizing...")

    if args.labeling == 'llm':
        d_list = []
        for i in range(CFG.class_num[args.dataset]):
            d_list.append(
                train_dataset.filter(lambda e: e['label'] == i).select(range(1000 // CFG.class_num[args.dataset])))
        train_dataset = concatenate_datasets(d_list)
        if args.dataset == 'SetFit/sst2':
            d_list = []
            for i in range(CFG.class_num[args.dataset]):
                d_list.append(
                    val_dataset.filter(lambda e: e['label'] == i).select(range(80 // CFG.class_num[args.dataset])))
            val_dataset = concatenate_datasets(d_list)

        print("training labeled data len: ", len(train_dataset))
        if args.dataset == 'SetFit/sst2':
            print("val labeled data len: ", len(val_dataset))

    if args.backbone == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif args.backbone == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.backbone == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise Exception("backbone should be roberta or gpt2")

    print(list(set(train_dataset[CFG.dataset_config[args.dataset]["label_column"]])))

    train_dataset = preprocess(train_dataset, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])
    val_dataset = preprocess(val_dataset, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])

    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.dataset_config[args.dataset]["text_column"]], padding=True, truncation=True, max_length=args.max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_val_dataset = val_dataset.map(
       lambda e: tokenizer(e[CFG.dataset_config[args.dataset]["text_column"]], padding=True, truncation=True, max_length=args.max_length), batched=True,
       batch_size=len(val_dataset))
    
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.dataset_config[args.dataset]["text_column"]])
    encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.dataset_config[args.dataset]["text_column"]])

    # HuggingFace map operations are lazy so we access the entire dataset to ensure the operations are applied
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]
    encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

    concept_set = CFG.concept_set[args.dataset]
    print("concept len: ", len(concept_set))

    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    if args.labeling == 'mpnet':
        prefix += "mpnet_acs"
    elif args.labeling == 'simcse':
        prefix += "simcse_acs"
    elif args.labeling == 'angle':
        prefix += "angle_acs"
    elif args.labeling == 'llm':
        prefix += "llm_labeling"

    prefix += "/"
    prefix += d_name
    prefix += "/"
    train_similarity = np.load(prefix + "/concept_labels_train.npy")
    val_similarity = np.load(prefix + "/concept_labels_val.npy")


    if args.automatic_concept_correction:
        start = time.time()
        print("training intervention...")
        # Convert label list to NumPy array
        train_labels = np.array(encoded_train_dataset[CFG.dataset_config[args.dataset]["label_column"]])
        
        # Get all concept labels
        concept_labels = np.array([get_labels(j, args.dataset) for j in range(len(concept_set))])  # shape: (num_concepts,)
        
        # Create a (num_samples, num_concepts) label match matrix
        label_matches = train_labels[:, None] == concept_labels[None, :]  # shape: (num_samples, num_concepts)
        
        # Zero out all mismatched labels
        train_similarity[~label_matches] = 0.0
        
        # Clip negative similarities to 0
        np.maximum(train_similarity, 0.0, out=train_similarity)


        val_labels = np.array(encoded_val_dataset[CFG.dataset_config[args.dataset]["label_column"]])
        
        # Get all concept labels
        concept_labels = np.array([get_labels(j, args.dataset) for j in range(len(concept_set))])  # shape: (num_concepts,)
        
        # Create a (num_samples, num_concepts) label match matrix
        label_matches = val_labels[:, None] == concept_labels[None, :]  # shape: (num_samples, num_concepts)
        
        # Zero out all mismatched labels
        val_similarity[~label_matches] = 0.0
        
        # Clip negative similarities to 0
        np.maximum(val_similarity, 0.0, out=val_similarity)
        
        end = time.time()
        print("time of training intervention:", (end - start) / 3600, "hours")

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    val_loader = build_loaders(encoded_val_dataset, val_similarity, mode="valid")

    if args.backbone == 'roberta':
        if args.tune_cbl_only:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            preLM = AutoModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)
    elif args.backbone == 'gpt2':
        if args.tune_cbl_only:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            preLM = AutoModel.from_pretrained('gpt2').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)
    elif args.backbone == 'bert':
        if args.tune_cbl_only:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            preLM = AutoModel.from_pretrained('bert-base-uncased').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
        else:
            print("preparing backbone(bert)+CBL...")
            backbone_cbl = BERTCBL(len(concept_set), args.dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("start training...")
    best_loss = float('inf')

    if args.backbone == 'roberta':
        prefix += 'roberta_cbm'
    elif args.backbone == 'gpt2':
        prefix += 'gpt2_cbm'
    elif args.backbone == 'bert':
        prefix += 'bert_cbm'
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "cbl"
    if args.tune_cbl_only:
        model_name += "_no_backbone"
    if args.automatic_concept_correction:
        model_name += "_acc"

    start = time.time()
    if args.labeling == 'llm':
        epochs = 10
    else:
        epochs = CFG.cbl_epochs[args.dataset]
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        if args.tune_cbl_only:
            cbl.train()
        else:
            backbone_cbl.train()
        training_loss = []
        for i, batch in enumerate(train_loader):
            batch_text, batch_sim = batch[0], batch[1]
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_sim = batch_sim.to(device)

            if args.tune_cbl_only:
                with torch.no_grad():
                    LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta' or args.backbone == 'bert':
                        LM_features = LM_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                cbl_features = cbl(LM_features)
            else:
                cbl_features = backbone_cbl(batch_text)
            optimizer.zero_grad()
            loss = -cos_sim_cubed(cbl_features, batch_sim)
            loss.backward()
            optimizer.step()
            print("batch ", str(i), " loss: ", loss.detach().cpu().numpy(), end="\r")
            training_loss.append(loss.detach().cpu().numpy())
        avg_training_loss = sum(training_loss)/len(training_loss)
        print("training loss: ", avg_training_loss)
        if args.tune_cbl_only:
            cbl.eval()
        else:
            backbone_cbl.eval()
        val_loss = []
        for batch in val_loader:
            batch_text, batch_sim = batch[0], batch[1]
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_sim = batch_sim.to(device)
            with torch.no_grad():
                if args.tune_cbl_only:
                    LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta' or args.backbone == 'bert':
                        LM_features = LM_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                    cbl_features = cbl(LM_features)
                else:
                    cbl_features = backbone_cbl(batch_text)
                loss = -cos_sim_cubed(cbl_features, batch_sim)
                val_loss.append(loss.detach().cpu().numpy())
        avg_val_loss = sum(val_loss)/len(val_loss)
        print("val loss: ", avg_val_loss)
        if avg_val_loss < best_loss:
            print("save model")
            best_loss = avg_val_loss
            if args.tune_cbl_only:
                torch.save(cbl.state_dict(), prefix + model_name + ".pt")
            else:
                torch.save(backbone_cbl.state_dict(), prefix + model_name + ".pt")

        # ==== DIAGNOSTICS: đo S_W và S_B ====
    print("\nRunning diagnostic on train set for SW/SB...")
    model_path = prefix + model_name + ".pt"

    # Load model tốt nhất
    if args.tune_cbl_only:
        cbl.load_state_dict(torch.load(model_path))
        cbl.eval()
    else:
        backbone_cbl.load_state_dict(torch.load(model_path))
        backbone_cbl.eval()

    # Lấy toàn bộ embedding concept z_pred trên train
    zs, ys = [], []
    with torch.no_grad():
        for batch in train_loader:
            batch_text, batch_sim = batch
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            labels = np.array(batch_sim.cpu())  # chỉ để đồng bộ chiều

            if args.tune_cbl_only:
                LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                if args.backbone in ["roberta", "bert"]:
                    LM_features = LM_features[:, 0, :]
                elif args.backbone == "gpt2":
                    LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                z = cbl(LM_features)
            else:
                z = backbone_cbl(batch_text)

            zs.append(z.detach().cpu().numpy())
            ys.append(encoded_train_dataset[CFG.dataset_config[args.dataset]["label_column"]][:z.size(0)])

    z_all = np.concatenate(zs)
    y_all = np.concatenate(ys)
    print("z_all shape:", z_all.shape, "y_all shape:", y_all.shape)

    # ==== Tính S_W và S_B ====
    classes = np.unique(y_all)
    mus, ncs = [], []
    S_W, S_B = 0.0, 0.0
    for c in classes:
        zc = z_all[y_all == c]
        mu_c = zc.mean(axis=0)
        mus.append(mu_c)
        ncs.append(len(zc))
        S_W += ((zc - mu_c) ** 2).sum()
    mu_all = z_all.mean(axis=0)
    for mu_c, n_c in zip(mus, ncs):
        S_B += n_c * ((mu_c - mu_all) ** 2).sum()
    print(f"S_W = {S_W:.4f}, S_B = {S_B:.4f}, ratio S_W/S_B = {S_W/S_B:.4f}")

    end = time.time()
    print("time of training CBL:", (end - start) / 3600, "hours")