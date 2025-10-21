# dump_cbl_acc_vectors.py
import os, argparse, random, torch, numpy as np, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast, GPT2TokenizerFast
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
    raise ValueError(f"Unknown backbone: {backbone}")

def build_prefix(labeling, dataset):
    d_name = dataset.replace('/', '_')
    base = {'mpnet':'mpnet_acs','simcse':'simcse_acs','angle':'angle_acs','llm':'llm_labeling'}[labeling]
    return f"./{base}/{d_name}"

def apply_acc(similarity, labels, concept_set, dataset):
    # zero-out concepts whose class != sample class; clip negatives to 0
    concept_labels = np.array([get_labels(j, dataset) for j in range(len(concept_set))])
    matches = labels[:, None] == concept_labels[None, :]
    similarity = similarity.copy()
    # safety mask shape
    if matches.shape != similarity.shape:
        min_r = min(matches.shape[0], similarity.shape[0])
        min_c = min(matches.shape[1], similarity.shape[1])
        matches = matches[:min_r, :min_c]
        similarity = similarity[:min_r, :min_c]
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
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        preLM.eval()
        return ('cbl_only', cbl, preLM)
    else:
        if backbone=='roberta':
            model = RobertaCBL(n_concepts, dropout).to(device)
        elif backbone=='gpt2':
            model = GPT2CBL(n_concepts, dropout).to(device)
        elif backbone=='bert':
            model = BERTCBL(n_concepts, dropout).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
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

def load_split(dataset, split, backbone, max_length):
    # always build consistent split via our helper
    has_val = True  # giữ nguyên như train_CBL thường dùng
    tr, va, te = train_val_test_split(dataset, CFG.dataset_config[dataset]["label_column"],
                                      ratio=0.2, has_val=has_val)
    if split == "train": ds = tr
    elif split == "val": ds = va
    elif split == "test": ds = te
    else: raise ValueError("--split must be one of {train,val,test}")

    tok = load_tokenizer(backbone)
    ds = preprocess(ds, dataset,
                    CFG.dataset_config[dataset]["text_column"],
                    CFG.dataset_config[dataset]["label_column"])
    enc = ds.map(lambda e: tok(e[CFG.dataset_config[dataset]["text_column"]],
                               padding=True, truncation=True, max_length=max_length),
                 batched=True, batch_size=len(ds))
    enc = enc.remove_columns([CFG.dataset_config[dataset]["text_column"]])
    return enc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SetFit/sst2")
    ap.add_argument("--backbone", default="roberta", choices=["roberta","gpt2","bert"])
    ap.add_argument("--labeling", default="mpnet", choices=["mpnet","simcse","angle","llm"])
    ap.add_argument("--cbl_path", required=True)
    ap.add_argument("--acc_like_train", action="store_true",
                    help="Áp ACC lên z_label giống lúc train_CBL (zero-out khác lớp, clip âm→0).")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5, help="Top-k concept để in")
    args = ap.parse_args()

    dataset = args.dataset
    concept_set = CFG.concept_set[dataset]
    concept_names = list(concept_set)  # đảm bảo index -> tên concept

    # 1) data
    enc = load_split(dataset, args.split, args.backbone, args.max_length)
    labels = np.array(enc[ CFG.dataset_config[dataset]["label_column"] ])

    # 2) load labels (ACS) và áp ACC nếu yêu cầu
    prefix = build_prefix(args.labeling, dataset)
    z_label_all = np.load(f"{prefix}/concept_labels_{args.split}.npy")

    # đồng bộ số hàng giữa enc và z_label_all (phòng lệch split/seed)
    min_len = min(len(enc["input_ids"]), z_label_all.shape[0])
    if min_len == 0:
        raise RuntimeError("Empty split or labels; check your files.")
    # cắt dữ liệu về min_len
    for k in list(enc.keys()):
        enc[k] = enc[k][:min_len]
    labels = labels[:min_len]
    z_label_all = z_label_all[:min_len]

    if args.acc_like_train:
        z_label_all = apply_acc(z_label_all, labels, concept_set, dataset)

    # 3) load CBL
    mode, model, preLM = load_cbl(args.cbl_path, args.backbone, len(concept_set))

    # 4) pick K random indices
    idxs = random.sample(range(min_len), k=min(args.k, min_len))
    batch = {
        "input_ids": torch.tensor(np.array(enc["input_ids"])[idxs]).to(device),
        "attention_mask": torch.tensor(np.array(enc["attention_mask"])[idxs]).to(device)
    }

    # 5) compute z_pred & gather z_label
    z_pred = forward_cbl(batch, mode, model, preLM, args.backbone).detach().cpu().numpy()
    z_label = z_label_all[idxs]
    y_batch = labels[idxs]

    # normalize both for true cosine
    z_pred_norm = z_pred / (np.linalg.norm(z_pred, axis=1, keepdims=True) + 1e-12)
    z_label_norm = z_label / (np.linalg.norm(z_label, axis=1, keepdims=True) + 1e-12)

    # 6) print
    print(f"\n=== Dump {len(idxs)} samples (split={args.split}) ===")
    for i, gi in enumerate(idxs):
        true_cos = float((z_pred_norm[i] * z_label_norm[i]).sum())
        top_p_idx = z_pred[i].argsort()[-args.topk:][::-1]
        top_l_idx = z_label[i].argsort()[-args.topk:][::-1]

        top_p_named = [(concept_names[j], float(z_pred[i, j])) for j in top_p_idx]
        top_l_named = [(concept_names[j], float(z_label[i, j])) for j in top_l_idx]

        # optional: top-k overlap rate (tên concept)
        overlap = len(set(top_p_idx) & set(top_l_idx)) / float(args.topk)

        print(f"\n[#{i} | global_id={gi} | y={y_batch[i]} | cosine(z_pred, z_label)={true_cos:.4f} | top{args.topk}_overlap={overlap:.2f}]")
        print("  CBL raw top-{}:".format(args.topk), top_p_named)
        print("  ACC/ACS top-{}:".format(args.topk), top_l_named)

    # 7) optional save
    np.save("dump_z_pred.npy", z_pred)
    np.save("dump_z_label.npy", z_label)
    print("\nSaved: dump_z_pred.npy, dump_z_label.npy")