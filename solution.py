# made by - Karthik
import sys, os, re, math, time, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

START_TIME = time.time()
TIME_BUDGET_SEC = float(os.environ.get("ERIS_TIME_BUDGET_SEC", 3300))


def time_left():
    return TIME_BUDGET_SEC - (time.time() - START_TIME)


EMB_DIM = int(os.environ.get("ERIS_EMB_DIM", 128))
HIDDEN = int(os.environ.get("ERIS_HIDDEN", 128))
BATCH_SIZE = int(os.environ.get("ERIS_BATCH_SIZE", 64))
MAX_EPOCHS = int(os.environ.get("ERIS_MAX_EPOCHS", 45))
PATIENCE = int(os.environ.get("ERIS_PATIENCE", 8))
LR = float(os.environ.get("ERIS_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("ERIS_WEIGHT_DECAY", 1e-3))
N_FOLDS = int(os.environ.get("ERIS_N_FOLDS", 5))
N_REPEATS = int(os.environ.get("ERIS_N_REPEATS", 2))
LABEL_SMOOTH = float(os.environ.get("ERIS_LABEL_SMOOTH", 0.2))
MAXLEN_PASSAGE = int(os.environ.get("ERIS_MAXLEN_PASSAGE", 40))
MAXLEN_ITEM = int(os.environ.get("ERIS_MAXLEN_ITEM", 90))
MAX_TAGS = int(os.environ.get("ERIS_MAX_TAGS", 8))
MIN_FREQ = int(os.environ.get("ERIS_MIN_FREQ", 3))
MAX_VOCAB = int(os.environ.get("ERIS_MAX_VOCAB", 15000))
EMB_DROPOUT = float(os.environ.get("ERIS_EMB_DROPOUT", 0.5))
GRAD_CLIP = float(os.environ.get("ERIS_GRAD_CLIP", 5.0))
SEED = int(os.environ.get("ERIS_SEED", 13))

TIERS = ["t1", "t2", "t3", "t4", "t5"]

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def find_data_root(argv_path):
    candidates = []
    if argv_path:
        candidates.append(Path(argv_path))
    here = Path(__file__).resolve().parent
    candidates += [
        Path("."),
        here,
        here / "dataset" / "public",
        here.parent / "dataset" / "public",
        Path("./dataset/public"),
        Path("./data"),
    ]
    for c in candidates:
        try:
            if (c / "train.csv").exists() and (c / "items.csv").exists():
                return c
        except Exception:
            continue
    for root in [Path("."), here]:
        try:
            for p in root.rglob("train.csv"):
                if (p.parent / "items.csv").exists():
                    return p.parent
        except Exception:
            continue
    raise FileNotFoundError("could not locate dataset root with train.csv/items.csv")


def per_task_score(p, true_idx):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / s
    pt = p[true_idx]
    g = int((p > pt).sum())
    k = int((p == pt).sum())
    rr = float(np.mean([1.0 / r for r in range(g + 1, g + k + 1)]))
    cal = 1.0 + math.log(max(pt, 1e-9)) / math.log(20.0)
    cal = min(1.0, cal)
    return 0.5 * rr + 0.5 * cal


def tier_balanced_score(probs_matrix, true_idx_arr, tiers_arr):
    tier_scores = {}
    for t in TIERS:
        idx = np.where(tiers_arr == t)[0]
        if len(idx) == 0:
            continue
        scores = [per_task_score(probs_matrix[i], true_idx_arr[i]) for i in idx]
        tier_scores[t] = float(np.mean(scores))
    if not tier_scores:
        return 0.02, {}
    overall = float(np.mean(list(tier_scores.values())))
    overall = max(0.02, min(1.0, overall))
    return overall, tier_scores


def tune_temperature(probs_matrix, true_idx_arr, tiers_arr):
    grid = np.geomspace(0.15, 6.0, 26)
    temp_by_tier = {}
    for t in TIERS:
        idx = np.where(tiers_arr == t)[0]
        if len(idx) == 0:
            temp_by_tier[t] = 1.0
            continue
        best_T, best_s = 1.0, -1e18
        for T in grid:
            scores = []
            for i in idx:
                p = np.clip(probs_matrix[i], 1e-9, None)
                p = p ** (1.0 / T)
                p = p / p.sum()
                scores.append(per_task_score(p, true_idx_arr[i]))
            s = float(np.mean(scores))
            if s > best_s:
                best_s, best_T = s, T
        temp_by_tier[t] = best_T
    return temp_by_tier


def apply_temperature(probs_matrix, tiers_arr, temp_by_tier):
    out = np.array(probs_matrix, dtype=np.float64, copy=True)
    for t in TIERS:
        idx = np.where(tiers_arr == t)[0]
        if len(idx) == 0:
            continue
        T = temp_by_tier.get(t, 1.0)
        p = np.clip(out[idx], 1e-9, None)
        p = p ** (1.0 / T)
        p = p / p.sum(axis=1, keepdims=True)
        out[idx] = p
    return out


def build_word_vocab(texts, min_freq, max_vocab):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    ordered = [w for w, c in cnt.most_common() if c >= min_freq][:max_vocab]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in ordered:
        vocab[w] = len(vocab)
    return vocab


def encode_tokens(text, vocab, maxlen):
    toks = tokenize(text)[:maxlen]
    ids = [vocab.get(t, 1) for t in toks]
    L = len(ids)
    if L < maxlen:
        ids = ids + [0] * (maxlen - L)
    mask = [1.0] * L + [0.0] * (maxlen - L)
    if L == 0:
        mask[0] = 1.0
    return ids, mask


def build_cat_vocab(values):
    vocab = {"<unk>": 0}
    for v in sorted(set(str(x) for x in values)):
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def parse_traits(s):
    d = {}
    if isinstance(s, str) and s:
        for kv in s.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k] = v
    return d


def trait_cat(v):
    if v is None:
        return 0
    vs = str(v)
    if vs == "False":
        return 1
    if vs == "True":
        return 2
    return 3


class AttnEncoder(nn.Module):
    def __init__(self, emb_dim, hidden, out_dim, dropout=0.3, n_heads=1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, tok_emb, mask):
        key_padding_mask = mask < 0.5
        tok_emb = self.drop(tok_emb)
        attn_out, _ = self.self_attn(tok_emb, tok_emb, tok_emb,
                                      key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(tok_emb + attn_out)
        B = h.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = self.ln2(pooled.squeeze(1))
        vec = self.proj(pooled)
        vec = nn.functional.normalize(vec, dim=-1)
        return vec


class GroundingModel(nn.Module):
    def __init__(self, vocab_size, tag_vocab_size, locale_size, region_size, tier_size,
                 n_trait_keys, n_difficulty, n_extra, emb_dim=128, hidden=128, cat_hidden=32,
                 dropout=0.5):
        super().__init__()
        self.hidden = hidden
        self.n_raw_signals = 1 + n_extra
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.passage_enc = AttnEncoder(emb_dim, hidden, hidden, dropout=dropout)
        self.item_text_enc = AttnEncoder(emb_dim, hidden, hidden, dropout=dropout)

        self.tag_emb = nn.Embedding(tag_vocab_size, 16, padding_idx=0)
        self.locale_emb = nn.Embedding(locale_size, 6)
        self.region_emb = nn.Embedding(region_size, 6)
        self.tier_emb = nn.Embedding(tier_size, 6)
        self.trait_emb = nn.Embedding(4, 6)
        self.trait_proj = nn.Sequential(nn.Linear(6 * n_trait_keys, 16), nn.ReLU())

        cat_in = 16 + 6 + 6 + 6 + 16
        self.cat_fuse = nn.Sequential(
            nn.Linear(cat_in, cat_hidden), nn.ReLU(), nn.Dropout(dropout),
        )

        self.diff_emb = nn.Embedding(n_difficulty, 6)

        self.tier_scale = nn.Embedding(n_difficulty, self.n_raw_signals)
        self.tier_bias = nn.Embedding(n_difficulty, self.n_raw_signals)
        nn.init.ones_(self.tier_scale.weight)
        nn.init.zeros_(self.tier_bias.weight)

        inter_dim = self.n_raw_signals * 2 + cat_hidden + 6
        self.scorer = nn.Sequential(
            nn.Linear(inter_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode_passage(self, passage_tok, passage_mask):
        emb = self.word_emb(passage_tok)
        return self.passage_enc(emb, passage_mask)

    def encode_items(self, item_tok, item_mask, tag_ids, tag_mask, locale_id, region_id,
                      tier_id, trait_ids):
        emb = self.word_emb(item_tok)
        text_vec = self.item_text_enc(emb, item_mask)

        tag_e = self.tag_emb(tag_ids)
        tag_mask_f = tag_mask.unsqueeze(-1)
        tag_vec = (tag_e * tag_mask_f).sum(1) / tag_mask_f.sum(1).clamp(min=1.0)

        loc_vec = self.locale_emb(locale_id)
        reg_vec = self.region_emb(region_id)
        tier_vec = self.tier_emb(tier_id)

        trait_e = self.trait_emb(trait_ids)
        B = trait_e.shape[0]
        trait_vec = self.trait_proj(trait_e.reshape(B, -1))

        cat_in = torch.cat([tag_vec, loc_vec, reg_vec, tier_vec, trait_vec], dim=-1)
        cat_vec = self.cat_fuse(cat_in)

        return torch.cat([text_vec, cat_vec], dim=-1)

    def score(self, passage_vec, item_vec, extra_feats, diff_id):
        B, K, D = item_vec.shape
        text_vec = item_vec[..., :self.hidden]
        cat_vec = item_vec[..., self.hidden:]
        p_exp = passage_vec.unsqueeze(1).expand(B, K, self.hidden)
        cos_sim = (p_exp * text_vec).sum(-1, keepdim=True)
        raw_signals = torch.cat([cos_sim, extra_feats], dim=-1)

        scale = self.tier_scale(diff_id).unsqueeze(1)
        bias = self.tier_bias(diff_id).unsqueeze(1)
        interacted = raw_signals * scale + bias

        d_vec = self.diff_emb(diff_id).unsqueeze(1).expand(B, K, self.diff_emb.embedding_dim)
        inter = torch.cat([raw_signals, interacted, cat_vec, d_vec], dim=-1)
        logits = self.scorer(inter).squeeze(-1)
        return logits


def main():
    argv_public = sys.argv[1] if len(sys.argv) > 1 else None
    argv_out = sys.argv[2] if len(sys.argv) > 2 else None

    data_root = find_data_root(argv_public)
    print("data_root:", data_root)

    submission_out = Path(argv_out) if argv_out else Path("working/submission.csv")
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    if submission_out.exists():
        for v in range(1, 1000):
            cand = submission_out.parent / f"{submission_out.stem}_v{v}{submission_out.suffix}"
            if not cand.exists():
                try:
                    submission_out.replace(cand)
                except Exception:
                    pass
                break

    items = pd.read_csv(data_root / "items.csv")
    train = pd.read_csv(data_root / "train.csv")
    test = pd.read_csv(data_root / "test.csv")
    sample_sub = pd.read_csv(data_root / "sample_submission.csv")

    test_task_ids = test["task_id"].tolist()
    n_test = len(test_task_ids)

    prior_rows = []
    for i, tid in enumerate(test_task_ids):
        prior_rows.append([tid] + [1.0 / 20.0] * 20)
    prior_cols = ["task_id"] + [f"p{i}" for i in range(1, 21)]
    pd.DataFrame(prior_rows, columns=prior_cols).to_csv(submission_out, index=False)
    print("wrote uniform prior submission")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
    print("device:", device)

    set_seed(SEED)

    item_id_to_row = {iid: i for i, iid in enumerate(items["item_id"].tolist())}
    N_items = len(items)

    tag_lists = [[t.strip() for t in str(tags).split(",") if t.strip()] for tags in items["tags"]]
    all_tags_flat = [t for lst in tag_lists for t in lst]
    tag_cnt = Counter(all_tags_flat)
    tag_vocab = {"<pad>": 0, "<unk>": 1}
    for t, _ in tag_cnt.most_common():
        if t not in tag_vocab:
            tag_vocab[t] = len(tag_vocab)

    locale_vocab = build_cat_vocab(items["locale"].tolist())
    region_vocab = build_cat_vocab(items["region"].tolist())
    tier_vocab = build_cat_vocab(items["tier"].tolist())

    trait_dicts = [parse_traits(s) for s in items["traits"].tolist()]
    trait_keys = sorted({k for d in trait_dicts for k in d.keys()})
    if not trait_keys:
        trait_keys = ["_dummy"]
    n_trait_keys = len(trait_keys)

    print("building word vocab...")
    all_text_for_vocab = list(items["content_summary"].astype(str)) + \
        list(train["passage"].astype(str)) + list(test["passage"].astype(str))
    word_vocab = build_word_vocab(all_text_for_vocab, MIN_FREQ, MAX_VOCAB)
    print("word vocab size:", len(word_vocab))

    print("encoding item features...")
    item_tok_ids = np.zeros((N_items, MAXLEN_ITEM), dtype=np.int64)
    item_tok_mask = np.zeros((N_items, MAXLEN_ITEM), dtype=np.float32)
    item_tag_ids = np.zeros((N_items, MAX_TAGS), dtype=np.int64)
    item_tag_mask = np.zeros((N_items, MAX_TAGS), dtype=np.float32)
    item_locale_id = np.zeros((N_items,), dtype=np.int64)
    item_region_id = np.zeros((N_items,), dtype=np.int64)
    item_tier_id = np.zeros((N_items,), dtype=np.int64)
    item_trait_ids = np.zeros((N_items, n_trait_keys), dtype=np.int64)

    for i in range(N_items):
        ids, mask = encode_tokens(str(items["content_summary"].iat[i]), word_vocab, MAXLEN_ITEM)
        item_tok_ids[i] = ids
        item_tok_mask[i] = mask

        tlist = tag_lists[i][:MAX_TAGS]
        for j, t in enumerate(tlist):
            item_tag_ids[i, j] = tag_vocab.get(t, 1)
            item_tag_mask[i, j] = 1.0
        if len(tlist) == 0:
            item_tag_mask[i, 0] = 1.0

        item_locale_id[i] = locale_vocab.get(str(items["locale"].iat[i]), 0)
        item_region_id[i] = region_vocab.get(str(items["region"].iat[i]), 0)
        item_tier_id[i] = tier_vocab.get(str(items["tier"].iat[i]), 0)

        d = trait_dicts[i]
        for k_idx, k in enumerate(trait_keys):
            item_trait_ids[i, k_idx] = trait_cat(d.get(k))

    def build_task_arrays(df):
        n = len(df)
        passage_tok = np.zeros((n, MAXLEN_PASSAGE), dtype=np.int64)
        passage_mask = np.zeros((n, MAXLEN_PASSAGE), dtype=np.float32)
        slate_idx = np.zeros((n, 20), dtype=np.int64)
        diff_ids = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            ids, mask = encode_tokens(str(df["passage"].iat[i]), word_vocab, MAXLEN_PASSAGE)
            passage_tok[i] = ids
            passage_mask[i] = mask
            slate = df["slate"].iat[i].split(";")
            for j, iid in enumerate(slate):
                slate_idx[i, j] = item_id_to_row.get(iid, 0)
            diff_ids[i] = TIERS.index(df["difficulty"].iat[i]) if df["difficulty"].iat[i] in TIERS else 0
        return passage_tok, passage_mask, slate_idx, diff_ids

    print("encoding train/test task arrays...")
    tr_passage_tok, tr_passage_mask, tr_slate_idx, tr_diff_ids = build_task_arrays(train)
    te_passage_tok, te_passage_mask, te_slate_idx, te_diff_ids = build_task_arrays(test)

    tr_chosen_idx = np.zeros((len(train),), dtype=np.int64)
    for i in range(len(train)):
        slate = train["slate"].iat[i].split(";")
        chosen = train["chosen_item_id"].iat[i]
        tr_chosen_idx[i] = slate.index(chosen) if chosen in slate else 0

    tr_tiers = train["difficulty"].values
    te_tiers = test["difficulty"].values

    print("computing tfidf / jaccard / length features...")
    corpus = list(items["content_summary"].astype(str)) + \
        list(train["passage"].astype(str)) + list(test["passage"].astype(str))
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    tfidf.fit(corpus)

    item_tfidf = tfidf.transform(items["content_summary"].astype(str))
    tr_passage_tfidf = tfidf.transform(train["passage"].astype(str))
    te_passage_tfidf = tfidf.transform(test["passage"].astype(str))

    from sklearn.preprocessing import normalize as sk_normalize
    item_tfidf_n = sk_normalize(item_tfidf)
    tr_passage_tfidf_n = sk_normalize(tr_passage_tfidf)
    te_passage_tfidf_n = sk_normalize(te_passage_tfidf)

    def compute_tfidf_sims(passage_tfidf_n, slate_idx):
        n = slate_idx.shape[0]
        flat_item_idx = slate_idx.reshape(-1)
        item_rows = item_tfidf_n[flat_item_idx]
        rep_idx = np.repeat(np.arange(n), 20)
        passage_rows = passage_tfidf_n[rep_idx]
        prod = item_rows.multiply(passage_rows)
        sims = np.asarray(prod.sum(axis=1)).ravel()
        return sims.reshape(n, 20).astype(np.float32)

    tr_tfidf_sim = compute_tfidf_sims(tr_passage_tfidf_n, tr_slate_idx)
    te_tfidf_sim = compute_tfidf_sims(te_passage_tfidf_n, te_slate_idx)

    item_token_sets = [set(tokenize(str(s))) for s in items["content_summary"]]
    item_lens = np.array([max(1, len(ts)) for ts in item_token_sets], dtype=np.float32)

    def compute_jaccard_len(passage_series, slate_idx):
        n = slate_idx.shape[0]
        jac = np.zeros((n, 20), dtype=np.float32)
        lenr = np.zeros((n, 20), dtype=np.float32)
        for i in range(n):
            ptoks = set(tokenize(str(passage_series.iat[i])))
            plen = max(1, len(ptoks))
            for j in range(20):
                ridx = slate_idx[i, j]
                its = item_token_sets[ridx]
                inter = len(ptoks & its)
                union = len(ptoks | its)
                jac[i, j] = inter / union if union > 0 else 0.0
                lenr[i, j] = min(plen, item_lens[ridx]) / max(plen, item_lens[ridx])
        return jac, lenr

    tr_jac, tr_lenr = compute_jaccard_len(train["passage"], tr_slate_idx)
    te_jac, te_lenr = compute_jaccard_len(test["passage"], te_slate_idx)

    print("computing trait-keyword match features...")
    KEYWORD_KEYS = ["OutdoorSeating", "RestaurantsDelivery", "GoodForKids", "RestaurantsTakeOut"]
    KEYWORD_PATTERNS = [
        re.compile(r"\b(outdoor|patio|terrace)\b"),
        re.compile(r"\bdeliver(y|ies)?\b"),
        re.compile(r"\b(kids?|family|families)\b"),
        re.compile(r"\b(takeout|take-out|take out|carryout|carry-out|carry out)\b"),
    ]
    item_trait_sign = np.zeros((N_items, len(KEYWORD_KEYS)), dtype=np.float32)
    for i in range(N_items):
        d = trait_dicts[i]
        for k_idx, key in enumerate(KEYWORD_KEYS):
            v = d.get(key)
            if v == "True":
                item_trait_sign[i, k_idx] = 1.0
            elif v == "False":
                item_trait_sign[i, k_idx] = -1.0

    def compute_trait_match(passage_series, slate_idx):
        n = slate_idx.shape[0]
        flags = np.zeros((n, len(KEYWORD_KEYS)), dtype=np.float32)
        for i in range(n):
            text = str(passage_series.iat[i]).lower()
            for k_idx, pat in enumerate(KEYWORD_PATTERNS):
                if pat.search(text):
                    flags[i, k_idx] = 1.0
        match = np.zeros((n, 20, len(KEYWORD_KEYS)), dtype=np.float32)
        for j in range(20):
            item_rows = slate_idx[:, j]
            match[:, j, :] = flags * item_trait_sign[item_rows]
        return match

    tr_trait_match = compute_trait_match(train["passage"], tr_slate_idx)
    te_trait_match = compute_trait_match(test["passage"], te_slate_idx)

    tr_extra = np.concatenate(
        [np.stack([tr_tfidf_sim, tr_jac, tr_lenr], axis=-1), tr_trait_match], axis=-1)
    te_extra = np.concatenate(
        [np.stack([te_tfidf_sim, te_jac, te_lenr], axis=-1), te_trait_match], axis=-1)
    N_EXTRA = tr_extra.shape[-1]

    item_tok_ids_t = torch.tensor(item_tok_ids, device=device)
    item_tok_mask_t = torch.tensor(item_tok_mask, device=device)
    item_tag_ids_t = torch.tensor(item_tag_ids, device=device)
    item_tag_mask_t = torch.tensor(item_tag_mask, device=device)
    item_locale_id_t = torch.tensor(item_locale_id, device=device)
    item_region_id_t = torch.tensor(item_region_id, device=device)
    item_tier_id_t = torch.tensor(item_tier_id, device=device)
    item_trait_ids_t = torch.tensor(item_trait_ids, device=device)

    tr_passage_tok_t = torch.tensor(tr_passage_tok, device=device)
    tr_passage_mask_t = torch.tensor(tr_passage_mask, device=device)
    tr_slate_idx_t = torch.tensor(tr_slate_idx, device=device)
    tr_diff_ids_t = torch.tensor(tr_diff_ids, device=device)
    tr_extra_t = torch.tensor(tr_extra, device=device)
    tr_chosen_idx_t = torch.tensor(tr_chosen_idx, device=device)

    te_passage_tok_t = torch.tensor(te_passage_tok, device=device)
    te_passage_mask_t = torch.tensor(te_passage_mask, device=device)
    te_slate_idx_t = torch.tensor(te_slate_idx, device=device)
    te_diff_ids_t = torch.tensor(te_diff_ids, device=device)
    te_extra_t = torch.tensor(te_extra, device=device)

    def gather_item_batch(slate_idx_batch):
        flat = slate_idx_batch.reshape(-1)
        return (item_tok_ids_t[flat], item_tok_mask_t[flat], item_tag_ids_t[flat],
                item_tag_mask_t[flat], item_locale_id_t[flat], item_region_id_t[flat],
                item_tier_id_t[flat], item_trait_ids_t[flat])

    n_train = len(train)
    oof_prob_sum = np.zeros((n_train, 20), dtype=np.float64)
    oof_count = np.zeros((n_train,), dtype=np.int64)
    test_prob_sum = np.zeros((n_test, 20), dtype=np.float64)
    test_model_count = 0

    def run_inference(model, passage_tok, passage_mask, slate_idx, diff_ids, extra, batch_size=128):
        model.eval()
        n = passage_tok.shape[0]
        out = np.zeros((n, 20), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                p_tok = passage_tok[start:end]
                p_mask = passage_mask[start:end]
                s_idx = slate_idx[start:end]
                d_id = diff_ids[start:end]
                ex = extra[start:end]
                B = p_tok.shape[0]
                passage_vec = model.encode_passage(p_tok, p_mask)
                it_tok, it_mask, tg_ids, tg_mask, loc_id, reg_id, tier_id, trait_ids = gather_item_batch(s_idx)
                item_vec = model.encode_items(it_tok, it_mask, tg_ids, tg_mask, loc_id, reg_id, tier_id, trait_ids)
                item_vec = item_vec.reshape(B, 20, -1)
                logits = model.score(passage_vec, item_vec, ex, d_id)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out[start:end] = probs
        return out

    def write_submission(probs_matrix, tiers_arr, temp_by_tier, task_ids, path):
        final_probs = apply_temperature(probs_matrix, tiers_arr, temp_by_tier)
        rows = []
        for i, tid in enumerate(task_ids):
            p = final_probs[i]
            s = p.sum()
            if s <= 0 or not np.isfinite(s):
                p = np.full(20, 1.0 / 20.0)
            else:
                p = p / s
            rows.append([tid] + list(p))
        cols = ["task_id"] + [f"p{i}" for i in range(1, 21)]
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)

    skf_cache = {}
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + r * 1000)
        skf_cache[r] = list(skf.split(np.zeros(n_train), tr_tiers))

    model = None
    for r in range(N_REPEATS):
        for f in range(N_FOLDS):
            if time_left() < 180:
                print("time budget nearly exhausted, stopping fold loop")
                break
            train_idx, val_idx = skf_cache[r][f]
            fold_seed = SEED + r * 100 + f
            set_seed(fold_seed)

            model = GroundingModel(
                vocab_size=len(word_vocab), tag_vocab_size=len(tag_vocab),
                locale_size=len(locale_vocab), region_size=len(region_vocab),
                tier_size=len(tier_vocab), n_trait_keys=n_trait_keys,
                n_difficulty=len(TIERS), n_extra=N_EXTRA, emb_dim=EMB_DIM, hidden=HIDDEN,
                dropout=EMB_DROPOUT,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

            best_val_score = -1e18
            best_state = None
            epochs_since_improve = 0

            train_idx_arr = np.array(train_idx)
            for epoch in range(MAX_EPOCHS):
                if time_left() < 120:
                    print("time budget nearly exhausted mid-fold, stopping epochs")
                    break
                model.train()
                np.random.shuffle(train_idx_arr)
                total_loss = 0.0
                n_batches = 0
                for start in range(0, len(train_idx_arr), BATCH_SIZE):
                    batch_idx = train_idx_arr[start:start + BATCH_SIZE]
                    batch_idx_t = torch.tensor(batch_idx, device=device)
                    p_tok = tr_passage_tok_t[batch_idx_t]
                    p_mask = tr_passage_mask_t[batch_idx_t]
                    s_idx = tr_slate_idx_t[batch_idx_t]
                    d_id = tr_diff_ids_t[batch_idx_t]
                    ex = tr_extra_t[batch_idx_t]
                    target = tr_chosen_idx_t[batch_idx_t]

                    B = p_tok.shape[0]
                    passage_vec = model.encode_passage(p_tok, p_mask)
                    it_tok, it_mask, tg_ids, tg_mask, loc_id, reg_id, tier_id, trait_ids = gather_item_batch(s_idx)
                    item_vec = model.encode_items(it_tok, it_mask, tg_ids, tg_mask, loc_id, reg_id, tier_id, trait_ids)
                    item_vec = item_vec.reshape(B, 20, -1)
                    logits = model.score(passage_vec, item_vec, ex, d_id)

                    loss = crit(logits, target)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    total_loss += float(loss.item())
                    n_batches += 1

                val_probs = run_inference(model, tr_passage_tok_t[val_idx], tr_passage_mask_t[val_idx],
                                           tr_slate_idx_t[val_idx], tr_diff_ids_t[val_idx], tr_extra_t[val_idx])
                val_score, _ = tier_balanced_score(val_probs, tr_chosen_idx[val_idx], tr_tiers[val_idx])
                avg_loss = total_loss / max(1, n_batches)
                print(f"repeat {r} fold {f} epoch {epoch} loss {avg_loss:.4f} val_score {val_score:.4f}")

                if val_score > best_val_score + 1e-5:
                    best_val_score = val_score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= PATIENCE:
                        print("early stopping")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            val_probs = run_inference(model, tr_passage_tok_t[val_idx], tr_passage_mask_t[val_idx],
                                       tr_slate_idx_t[val_idx], tr_diff_ids_t[val_idx], tr_extra_t[val_idx])
            oof_prob_sum[val_idx] += val_probs
            oof_count[val_idx] += 1

            test_probs = run_inference(model, te_passage_tok_t, te_passage_mask_t, te_slate_idx_t,
                                        te_diff_ids_t, te_extra_t)
            test_prob_sum += test_probs
            test_model_count += 1

            valid_oof = oof_count > 0
            if valid_oof.sum() > 0:
                oof_avg = np.zeros_like(oof_prob_sum)
                oof_avg[valid_oof] = oof_prob_sum[valid_oof] / oof_count[valid_oof][:, None]
                score_now, tier_now = tier_balanced_score(oof_avg[valid_oof], tr_chosen_idx[valid_oof],
                                                            tr_tiers[valid_oof])
                print("interim ensemble OOF score:", score_now, tier_now)

                temp_now = tune_temperature(oof_avg[valid_oof], tr_chosen_idx[valid_oof], tr_tiers[valid_oof])
                test_avg = test_prob_sum / test_model_count
                write_submission(test_avg, te_tiers, temp_now, test_task_ids, submission_out)
                print("wrote interim submission")
        if time_left() < 180:
            break

    valid_oof = oof_count > 0
    oof_avg = np.zeros_like(oof_prob_sum)
    oof_avg[valid_oof] = oof_prob_sum[valid_oof] / oof_count[valid_oof][:, None]
    final_oof_score, final_tier_scores = tier_balanced_score(oof_avg[valid_oof], tr_chosen_idx[valid_oof],
                                                               tr_tiers[valid_oof])
    print("FINAL ensemble-averaged OOF score:", final_oof_score)
    print("FINAL per-tier OOF scores:", final_tier_scores)

    temp_by_tier = tune_temperature(oof_avg[valid_oof], tr_chosen_idx[valid_oof], tr_tiers[valid_oof])
    print("tuned per-tier temperature:", temp_by_tier)

    calibrated_oof = apply_temperature(oof_avg[valid_oof], tr_tiers[valid_oof], temp_by_tier)
    calibrated_score, calibrated_tier_scores = tier_balanced_score(calibrated_oof, tr_chosen_idx[valid_oof],
                                                                     tr_tiers[valid_oof])
    print("FINAL calibrated OOF score:", calibrated_score)
    print("FINAL calibrated per-tier OOF scores:", calibrated_tier_scores)

    test_avg = test_prob_sum / max(1, test_model_count)
    write_submission(test_avg, te_tiers, temp_by_tier, test_task_ids, submission_out)
    print("wrote final submission to", submission_out)


if __name__ == "__main__":
    main()
