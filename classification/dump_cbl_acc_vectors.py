# dump_cbl_acc_vectors.py
import os, argparse, random, torch, numpy as np, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast, GPT2TokenizerFast
from datasets import concatenate_datasets
import config as CFG
from dataset_utils import train_val_test_split, preprocess
from modules import CBL, RobertaCBL, GPT2CBL, BERTCBL
from utils import eos_pooling, get_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(backbone):
    if backbone == 'roberta': return RobertaTokenizerFast.from_pretrained('roberta-base')
    if backbone == 'gpt2':
        tok = GPT2TokenizerFast.from_pretrained('gpt2'); tok.pad_token = tok.eos_token; return tok
    if backbone == 'bert':   return AutoTokenizer.from_pretrained('bert-base-uncased')
    raise ValueError

def build_prefix(labeling, dataset):
    d_name = dataset.replace('/', '_')
    base = {'mpnet':'mpnet_acs','simcse':'simcse_acs','angle':'angle_acs','llm':'llm_labeling'}[labeling]
    return f"./{base}/{d_name}"

def apply_acc(similarity, labels, concept_set, dataset):
    # zero-out concepts whose class != sample class; clip negatives to 0
    concept_labels = np.array([get_labels(j, dataset) for j in range(len(concept_set))])
    matches = labels[:, None] == concept_labels[None, :]
    similarity = similarity.copy()
    similarity[~matches] = 0.0
    np.maximum(similarity, 0.0, out=similarity)
    return similarity

def load_cbl(cbl_path, backbone, n_concepts, dropout=0.1):
    name = os.path.basename(cbl_path)
    nobb = 'no_backbone' in name
    if nobb:
        cbl = CBL(n_concepts, dropout).to(device)
        cbl.load_state_dict(torch.load(cbl_path, map_location=device)); cbl.eval()
        if backbone=='roberta':
            preLM = AutoModel.from_pretrained('roberta-base').to(device)
        elif backbone=='gpt2':
            preLM = AutoModel.from_pretrained('gpt2').to(device)
        elif backbone=='bert':
            preLM = AutoModel.from_pretrained('bert-base-uncased').to(device)
        preLM.eval()
        return ('cbl_only', cbl, preLM)
    else:
        if backbone=='roberta':
            model = RobertaCBL(n_concepts, dropout).to(device)
        elif backbone=='gpt2':
            model = GPT2CBL(n_concepts, dropout).to(device)
        elif backbone=='bert':
            model = BERTCBL(n_concepts, dropout).to(device)
        model.load_state_dict(torch.load(cbl_path, map_location=device)); model.eval()
        return ('with_backbone', model, None)

def forward_cbl(batch, mode, model, preLM, backbone):
    with torch.no_grad():
        if mode=='cbl_only':
            hs = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            if backbone in ['roberta','bert']: hs = hs[:,0,:]
            elif backbone=='gpt2':             hs = eos_pooling(hs, batch["attention_mask"])
            z = model(hs)
        else:
            z = model(batch)
    return z

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SetFit/sst2")
    ap.add_argument("--backbone", default="roberta", choices=["roberta","gpt2","bert"])
    ap.add_argument("--labeling", default="mpnet", choices=["mpnet","simcse","angle","llm"])
    ap.add_argument("--cbl_path", required=True)
    ap.add_argument("--acc_like_train", action="store_true",
                    help="áp ACC lên z_label giống lúc train_CBL")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    # 1) data
    train_ds, val_ds, _ = train_val_test_split(args.dataset, CFG.dataset_config[args.dataset]["label_column"], ratio=0.2, has_val=False)
    tokenizer = load_tokenizer(args.backbone)
    val_ds = preprocess(val_ds, args.dataset, CFG.dataset_config[args.dataset]["text_column"], CFG.dataset_config[args.dataset]["label_column"])
    enc_val = val_ds.map(lambda e: tokenizer(e[CFG.dataset_config[args.dataset]["text_column"]],
                                             padding=True, truncation=True, max_length=args.max_length),
                         batched=True, batch_size=len(val_ds))
    enc_val = enc_val.remove_columns([CFG.dataset_config[args.dataset]["text_column"]])
    enc_val = enc_val[:len(enc_val)]
    labels_val = np.array(enc_val[CFG.dataset_config[args.dataset]["label_column"]])

    # 2) load labels (ACS) and apply ACC if requested
    prefix = build_prefix(args.labeling, args.dataset)
    z_label_val = np.load(f"{prefix}/concept_labels_val.npy")
    concept_set = CFG.concept_set[args.dataset]
    if args.acc_like_train:
        z_label_val = apply_acc(z_label_val, labels_val, concept_set, args.dataset)

    # 3) load CBL
    mode, model, preLM = load_cbl(args.cbl_path, args.backbone, len(concept_set))

    # 4) pick K random indices
    idxs = random.sample(range(len(enc_val['input_ids'])), k=min(args.k, len(enc_val['input_ids'])))
    batch = {k: torch.tensor(np.array(v)[idxs]).to(device) for k, v in enc_val.items() if k in ["input_ids","attention_mask"]}

    # 5) compute z_pred
    z_pred = forward_cbl(batch, mode, model, preLM, args.backbone).detach().cpu().numpy()
    z_label = z_label_val[idxs]

    # also normalized z_pred for cosine comparison
    z_pred_norm = z_pred / (np.linalg.norm(z_pred, axis=1, keepdims=True) + 1e-12)

    # 6) print
    print(f"\n=== Dump {len(idxs)} samples ===")
    for i, gi in enumerate(idxs):
        cos = float((z_pred_norm[i] * z_label[i]).sum())
        top_p = z_pred[i].argsort()[-5:][::-1]
        top_l = z_label[i].argsort()[-5:][::-1]
        print(f"\n[#{i} | global_id={gi} | y={labels_val[gi]} | cosine(z_pred_norm, z_label)={cos:.4f}]")
        print("  CBL raw top-5:", [(j, float(z_pred[i, j])) for j in top_p])
        print("  ACC/ACS top-5:", [(j, float(z_label[i, j])) for j in top_l])

    # 7) optional save
    np.save("dump_z_pred.npy", z_pred)
    np.save("dump_z_label.npy", z_label)
    print("\nSaved: dump_z_pred.npy, dump_z_label.npy")