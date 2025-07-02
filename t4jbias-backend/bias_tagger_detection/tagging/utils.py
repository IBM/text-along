from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
from bias_tagger_detection.shared.args import ARGS
from bias_tagger_detection.shared.constants import CUDA


def build_optimizer(model, num_train_steps, learning_rate):
    global ARGS

    if ARGS.tagger_from_debiaser:
        parameters = list(model.cls_classifier.parameters()) + list(
            model.tok_classifier.parameters())
        parameters = list(filter(lambda p: p.requires_grad, parameters))
        return optim.Adam(parameters, lr=ARGS.learning_rate)
    else:
        param_optimizer = list(model.named_parameters())
        param_optimizer = list(
            filter(lambda name_param: name_param[1].requires_grad, param_optimizer))
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        return BertAdam(optimizer_grouped_parameters,
                        lr=learning_rate,
                        warmup=0.1,
                        t_total=num_train_steps)


def build_loss_fn(debias_weight=None):
    global ARGS

    if debias_weight is None:
        debias_weight = ARGS.debias_weight

    weight_mask = torch.ones(ARGS.num_tok_labels)
    weight_mask[-1] = 0

    if CUDA:
        weight_mask = weight_mask.cuda()
        criterion = CrossEntropyLoss(weight=weight_mask).cuda()
        per_tok_criterion = CrossEntropyLoss(
            weight=weight_mask, reduction='none').cuda()
    else:
        criterion = CrossEntropyLoss(weight=weight_mask)
        per_tok_criterion = CrossEntropyLoss(
            weight=weight_mask, reduction='none')

    def cross_entropy_loss(logits, labels, apply_mask=None):
        return criterion(
            logits.contiguous().view(-1, ARGS.num_tok_labels),
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

    def weighted_cross_entropy_loss(logits, labels, apply_mask=None):
        # weight mask = where to apply weight (post_tok_label_id from the batch)
        weights = apply_mask.contiguous().view(-1)
        weights = ((debias_weight - 1) * weights) + 1.0

        per_tok_losses = per_tok_criterion(
            logits.contiguous().view(-1, ARGS.num_tok_labels),
            labels.contiguous().view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))
        per_tok_losses = per_tok_losses * weights

        loss = torch.mean(
            per_tok_losses[torch.nonzero(per_tok_losses)].squeeze())

        return loss

    if debias_weight == 1.0:
        loss_fn = cross_entropy_loss
    else:
        loss_fn = weighted_cross_entropy_loss

    return loss_fn

# generalisation of logistic function to multiple dimensions
# x has 3 dimensions


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def run_inference(model, eval_dataloader, loss_fn, tokenizer):
    global ARGS
    # # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
    out = {
        'input_toks': [],
        'post_toks': [],
        'tok_loss': [],
        'tok_logits': [],
        'tok_probs': [],
        'labeling_hits': [],
        'results': []
    }

    for step, batch in enumerate(tqdm(eval_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)

        (
            pre_id, pre_mask, pre_len,
            post_in_id, post_out_id,
            tok_label_id, _,
            rel_ids, pos_ids, categories, tmi_ids
        ) = batch

        with torch.no_grad():
            _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                                  rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                                  pre_len=pre_len, tmi_ids=tmi_ids)
            tok_loss = loss_fn(tok_logits, tok_label_id,
                               apply_mask=tok_label_id)
        out['input_toks'] += [tokenizer.convert_ids_to_tokens(
            seq) for seq in pre_id.cpu().numpy()]
        out['post_toks'] += [tokenizer.convert_ids_to_tokens(
            seq) for seq in post_in_id.cpu().numpy()]
        out['tok_loss'].append(float(tok_loss.cpu().numpy()))
        logits = tok_logits.detach().cpu().numpy()
        labels = tok_label_id.detach().cpu().numpy()
        out['tok_logits'] += logits.tolist()
        out['tok_probs'] += to_probs(logits, pre_len)
        result = tag_hits(logits, labels, pre_id.cpu().numpy())
        out['labeling_hits'] += result[-1]["hits"]
        out['results'] += result
    return out


def train_for_epoch(model, train_dataloader, loss_fn, optimizer):
    global ARGS

    losses = []

    for step, batch in enumerate(tqdm(train_dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len,
            post_in_id, post_out_id,
            tok_label_id, _,
            rel_ids, pos_ids, categories, tmi_ids
        ) = batch
        _, tok_logits = model(pre_id, attention_mask=1.0-pre_mask,
                              rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                              pre_len=pre_len, tmi_ids=tmi_ids)
        loss = loss_fn(tok_logits, tok_label_id, apply_mask=tok_label_id)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        losses.append(loss.detach().cpu().numpy())

    return losses


def to_probs(logits, lens):
    per_tok_probs = softmax(np.array(logits)[:, :, :2], axis=2)
    # per_tok_probs is 2 dimensional with the probability in the 2nd dimension more important
    pos_scores = per_tok_probs[:, :, -1]
    out = []
    for score_seq, l in zip(pos_scores, lens):
        out.append(score_seq[:l].tolist())
    return out


def is_ranking_hit(probs, labels, ids, top=1):
    global ARGS
    # get rid of padding idx
    [probs, labels, tok_ids] = list(zip(
        *[(p, l, ids) for p, l, ids in zip(probs, labels, ids[0]) if l != ARGS.num_tok_labels - 1]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    top_ranked_values, top_indices = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    top_ids = [tok_ids[i] for i in top_indices]
    return {"top_tok_ids": top_ids, "probability": top_ranked_values, "bias_status": [True if probs_indices[i][0] >= 0.5 else False for i in top_indices], "hits": 1 if sum([labels[i] for i in top_indices]) > 0 else 0}

def tag_hits(logits, tok_labels, ids, top=1):
    global ARGS
    top_ids, top_ranked_values, bias_status, hits = [],[],[],[]
    # probs is 2 dimensional, last dimension is for bias probability
    probs = softmax(np.array(logits)[:, :, : ARGS.num_tok_labels - 1], axis=2)
    results = [is_ranking_hit(prob_dist, tok_label, ids, top=top) for prob_dist, tok_label in zip(probs, tok_labels)]
    for item in results:
        top_ids.append(item["top_tok_ids"][0])
        top_ranked_values.append(item["probability"][0])
        bias_status.append(item["bias_status"][0])
        hits.append(item["hits"])
    return [{"top_tok_ids": top_ids, "probability": top_ranked_values, "bias_status": bias_status, "hits": hits}]
