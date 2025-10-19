# backbone_probe.py
import argparse, os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, AutoModel, GPT2Model, GPT2TokenizerFast, RobertaModel, RobertaTokenizerFast
from datasets import load_dataset
import evaluate

import config as CFG
from dataset_utils import train_val_test_split, preprocess
from utils import eos_pooling, normalize
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class EncodedTextDataset(Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

def build_loader(enc, batch_size, num_workers, shuffle):
    return DataLoader(EncodedTextDataset(enc), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def encode_all(tokenizer, dataset, text_col, max_len, batch_size):
    # map theo batch lớn để nhanh
    enc = dataset.map(lambda e: tokenizer(e[text_col], padding=True, truncation=True, max_length=max_len),
                      batched=True, batch_size=min(len(dataset), batch_size))
    enc = enc.remove_columns([text_col])
    return enc[:len(enc)]

def extract_features(backbone, dl, device, bb_type):
    feats = []
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting backbone features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            if bb_type in ["roberta", "bert"]:
                last = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                x = last[:, 0, :]  # [CLS]
            elif bb_type == "gpt2":
                last = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                x = eos_pooling(last, batch["attention_mask"])
            else:
                raise ValueError("bb_type must be roberta|bert|gpt2")
            feats.append(x.detach().cpu())
    return torch.cat(feats, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SetFit/sst2")
    parser.add_argument("--backbone", type=str, default="roberta", choices=["roberta","bert","gpt2"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--saga_epoch", type=int, default=500)
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--use_relu", action="store_true", help="ReLU trước FL (tắt mặc định)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Split + preprocess
    train_ds, val_ds, test_ds = train_val_test_split(args.dataset, CFG.dataset_config[args.dataset]["label_column"], ratio=0.2, has_val=False)
    text_col = CFG.dataset_config[args.dataset]["text_column"]; label_col = CFG.dataset_config[args.dataset]["label_column"]
    train_ds = preprocess(train_ds, args.dataset, text_col, label_col)
    val_ds   = preprocess(val_ds,   args.dataset, text_col, label_col)
    test_ds  = preprocess(test_ds,  args.dataset, text_col, label_col)

    # 2) Tokenizer + backbone
    if args.backbone == "roberta":
        tok = RobertaTokenizerFast.from_pretrained("roberta-base")
        bb  = RobertaModel.from_pretrained("roberta-base").to(device)
    elif args.backbone == "bert":
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        bb  = AutoModel.from_pretrained("bert-base-uncased").to(device)
    else:
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        bb  = GPT2Model.from_pretrained("gpt2").to(device)

    # 3) Encode text
    map_bs = 1024
    enc_tr = encode_all(tok, train_ds, text_col, args.max_length, map_bs)
    enc_va = encode_all(tok, val_ds,   text_col, args.max_length, map_bs)
    enc_te = encode_all(tok, test_ds,  text_col, args.max_length, map_bs)

    # 4) Dataloaders (extract: không shuffle để giữ hàng lối)
    dl_tr = build_loader(enc_tr, args.batch_size, args.num_workers, shuffle=False)
    dl_va = build_loader(enc_va, args.batch_size, args.num_workers, shuffle=False)
    dl_te = build_loader(enc_te, args.batch_size, args.num_workers, shuffle=False)

    # 5) Backbone → features
    tr_X = extract_features(bb, dl_tr, device, args.backbone)
    va_X = extract_features(bb, dl_va, device, args.backbone)
    te_X = extract_features(bb, dl_te, device, args.backbone)

    # 6) Normalize (+ optional ReLU)
    tr_X, mu, sigma = normalize(tr_X, d=0)
    va_X, _,  _     = normalize(va_X, d=0, mean=mu, std=sigma)
    te_X, _,  _     = normalize(te_X, d=0, mean=mu, std=sigma)
    if args.use_relu:
        tr_X = F.relu(tr_X); va_X = F.relu(va_X); te_X = F.relu(te_X)

    # 7) Labels
    tr_y = torch.LongTensor(train_ds[label_col])
    va_y = torch.LongTensor(val_ds[label_col])
    te_y = torch.LongTensor(test_ds[label_col])

    # 8) Train FL (GLM-SAGA logistic regression)
    n_classes = CFG.class_num[args.dataset]
    linear = torch.nn.Linear(tr_X.shape[1], n_classes)
    linear.weight.data.zero_(); linear.bias.data.zero_()

    train_indexed = IndexedTensorDataset(tr_X, tr_y)
    va_tensor = TensorDataset(va_X, va_y)
    te_tensor = TensorDataset(te_X, te_y)

    train_loader = DataLoader(train_indexed, batch_size=args.saga_batch_size, shuffle=True)
    val_loader   = DataLoader(va_tensor,     batch_size=args.saga_batch_size, shuffle=False)
    test_loader  = DataLoader(te_tensor,     batch_size=args.saga_batch_size, shuffle=False)

    STEP_SIZE = 0.05; ALPHA = 0.99
    out = glm_saga(linear, train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                   val_loader=val_loader, test_loader=test_loader, do_zero=True,
                   n_classes=n_classes)

    print("[Dense] best test acc:", out['path'][-1]['metrics']['acc_test'])
    W_g = out['path'][-1]['weight']; b_g = out['path'][-1]['bias']

    # Sparse path (tùy chọn, giữ giống train_FL.py)
    meta = {'max_reg': {'nongrouped': 0.0007}}
    out_sp = glm_saga(linear, train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, test_loader=test_loader, do_zero=False,
                      n_classes=n_classes, metadata=meta, n_ex=tr_X.shape[0])
    print("[Sparse] first-path test acc:", out_sp['path'][0]['metrics']['acc_test'])
    W_g_sp = out_sp['path'][0]['weight']; b_g_sp = out_sp['path'][0]['bias']

    # 9) Save artifacts
    prefix = f"./backbone_probe/{args.dataset.replace('/','_')}/{args.backbone}/"
    os.makedirs(prefix, exist_ok=True)
    torch.save(mu,       prefix + "train_mean.pt")
    torch.save(sigma,    prefix + "train_std.pt")
    torch.save(W_g,      prefix + "W_g.pt")
    torch.save(b_g,      prefix + "b_g.pt")
    torch.save(W_g_sp,   prefix + "W_g_sparse.pt")
    torch.save(b_g_sp,   prefix + "b_g_sparse.pt")
    print("Artifacts saved to:", prefix)

if __name__ == "__main__":
    main()