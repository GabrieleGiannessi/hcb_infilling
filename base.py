# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: nlp_project
#     language: python
#     name: python3
# ---

# +
from transformers import BertTokenizer, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, DistilBertTokenizer, DistilBertForMaskedLM
from functools import partial
import torch
import numpy as np
import nltk
nltk.download('brown')
print("Downloaded corpus.")

from nltk.corpus import brown
from torch import nn
import re
import time

from bert_score import BERTScorer
from collections import defaultdict

model_name = 'bert-base-uncased'
method = 'topk'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertForMaskedLM.from_pretrained(model_name).to(device)
if "roberta" in model_name:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
elif "distilbert" in model_name:
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForMaskedLM.from_pretrained(model_name).to(device)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

sm = nn.Softmax(dim=-1)

np.random.seed(42)

beam_size = 5
total_examples = len(brown.words())  # Use entire corpus.
context_length = 512
report_period = 10
num_masks = 2
num_examples = 16
num_experiments = 32

def get_masked_positions(inputs, mask_id = tokenizer.mask_token_id):
  masked_positions = [(inputs[i] == mask_id).nonzero().squeeze() for i in range(len(inputs))]
  num_masked_per_input = [len(x) for x in masked_positions]
  # Currently require that all inputs have the same number of masked positions.
  # Could probably relax this in the future if need be.
  assert len(np.unique(num_masked_per_input)) == 1

  return torch.cat(masked_positions, dim=-1).reshape(len(inputs), -1)

def get_best_masked_positions(log_probs, remaining_masked_positions):
  remaining_masked_positions = remaining_masked_positions.to(log_probs.device)
  # First take max over vocab dimension, giving us top probs per position
  # in each input.
  max_per_pos, _ = log_probs.max(dim=-1)
  # Subset to just the remaining masked positions and get the position
  # with the highest prob option.
  max_pos_idx = torch.gather(max_per_pos, 1, remaining_masked_positions).argmax(dim=-1)
  # Map back to what the actual masked position is.
  best_mask_positions = torch.gather(remaining_masked_positions, 1, max_pos_idx.unsqueeze(1)).squeeze().detach().cpu()

  # Remove the selected masked positions from the tensor of remaining masked
  # positions.
  to_remove = torch.ones_like(remaining_masked_positions).scatter_(1, max_pos_idx.unsqueeze(1), 0)
  remaining_masked_positions = (remaining_masked_positions
                                [to_remove.bool()]
                                .reshape(remaining_masked_positions.shape[0], -1)
                                .detach()
                                .cpu())

  return best_mask_positions, remaining_masked_positions


from torch import nn

sm = nn.Softmax(dim=-1)

def mask_tokens(input_ids, indices):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    true_ids = input_ids[0, indices].tolist()
    masked_inputs = input_ids.clone()
    masked_inputs[0, indices] = tokenizer.mask_token_id
    return masked_inputs, true_ids

def mask_tokens_batch(input_ids, indices, pad_token_id = 0):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    true_ids = input_ids[:,indices]

    num_pads_masked = (true_ids == pad_token_id).sum().item()
    if num_pads_masked != 0:
      print(f"WARN: {num_pads_masked} / {true_ids.numel()} masked positions are pad tokens")

    masked_input_ids = input_ids.clone()
    masked_input_ids[:, indices] = tokenizer.mask_token_id
    return masked_input_ids, true_ids.tolist()

def display_suggestions(input_ids, suggestions):
    input_ids_copy = torch.clone(input_ids)
    mask_token_ids = (input_ids_copy.squeeze() == tokenizer.mask_token_id).nonzero().squeeze().tolist()
    for suggestion in suggestions:
        for token_idx, token_id in enumerate(suggestion[1:]):
              input_ids_copy[0, mask_token_ids[token_idx]] = token_id
        print(f"{suggestion[0]:.2%} - {tokenizer.decode(input_ids_copy.squeeze()[1:-1])}") #join(tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))

def display_suggestions_batch(input_ids, suggestions):
    input_ids_copy = torch.clone(input_ids)
    for i in range(input_ids.shape[0]):
      mask_token_ids = (input_ids_copy[i] == tokenizer.mask_token_id).nonzero().squeeze().tolist()
      for suggestion in suggestions[i]:
          for token_idx, token_id in enumerate(suggestion[1:]):
                input_ids_copy[i, mask_token_ids[token_idx]] = token_id
          print(f"{suggestion[0]:.2%} - {tokenizer.decode(input_ids_copy[i, 1:-1])}") #join(tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))

def sent_too_long(sent, max_toks):
  # split up a list until no more than 500 tokens anywhere
  words = sent.split(' ')
  # tokenize each word, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for word in words:
    toks = tokenizer(word)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + word)
      cur_toks += len(toks)
    else:
      final_ls.append(cur_str)
      cur_str = word
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
    return final_ls

# returns a list of sentences which we can write to a file, separated by newlines
def sep_sents(text, ind, max_toks=500):

  # add special separator after end punctuation
  text = re.sub(r'\·', '· [SEP] ', text)
  text = re.sub(r'\.', '. [SEP] ', text)
  text = re.sub(r'\;', '; [SEP] ', text)
  text = re.sub(r'\!', '! [SEP] ', text)
  text = re.sub(r'\?', '? [SEP] ', text)

  sents = text.split('[SEP]')
  # tokenize each sentence, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for sent in sents:
    toks = tokenizer(sent)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + sent)
      cur_toks += len(toks)
    elif cur_toks == 0:
      final_ls += sent_too_long(sent, max_toks)
      cur_toks = 0
      cur_str = ''
    else:
      final_ls.append(cur_str)
      cur_str = sent
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
  if ' ' not in cur_str:
    print('HERE: ' + str(ind))
  ret_ls = []
  for el in final_ls:
    if len(tokenizer(el)['input_ids']) > max_toks:
      ret_ls += sep_sents(el, -1 * ind)
    else:
      ret_ls.append(el)
  return ret_ls

def pseudo_ll_update(
    token_log_probs,
    curr_log_likelihoods,
    probabilities,
    pivots,
    mask_idxs,
):
  return token_log_probs + curr_log_likelihoods[:, None]

def ccr_update(
    token_log_probs,
    curr_log_likelihoods,
    log_probs,
    pivots,
    mask_idxs,
):
  # Let N = num_inputs * num_beams
  # pivots is (N, padded_input_length) and each value
  # contains the desired token to use as a pivot for that (input, position)
  # pair.

  # token_probs is (N, vocab_size)

  # pivot_idxs is a vector of len N where each value is the desired mask index
  # for that candidate.
  N = pivots.shape[0]
  pivot_idxs = pivots[torch.arange(N), mask_idxs]
  pivot_vals = log_probs[torch.arange(N), pivot_idxs]
  token_log_probs -= pivot_vals[:, None]
  return token_log_probs + curr_log_likelihoods[:, None]

def beam_search(
    num_inputs,
    candidates,
    candidate_log_likelihoods,
    remaining_masked_positions,
    token_log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots = None,
    initial = False,
):
  if initial:
    top_log_probs, top_indices = token_log_probs.topk(beam_size, dim=-1)

    mask_ids_repeated = torch.repeat_interleave(mask_ids, repeats=beam_size, dim=0)
    candidates = torch.repeat_interleave(candidates, repeats=beam_size, dim=0)
    candidates[torch.arange(len(candidates)), mask_ids_repeated] = top_indices.ravel()

    candidate_log_likelihoods = probs_update_fxn(
          top_log_probs,
          candidate_log_likelihoods,
          token_log_probs,
          pivots,
          mask_ids,
    )
    candidate_log_likelihoods = candidate_log_likelihoods.ravel()
    return candidates, candidate_log_likelihoods, remaining_masked_positions
  else:
    top_log_probs, top_indices = token_log_probs.topk(beam_size, dim=-1)
    top_log_probs = probs_update_fxn(
        top_log_probs,
        candidate_log_likelihoods,
        token_log_probs,
        pivots,
        mask_ids,
    )
    # Update each input's candidates in place.
    for input_idx in range(num_inputs):
      # Indices of this input's candidates in the overall candidate tensor.
      start = input_idx * beam_size
      end = start + beam_size

      # Collect token probabilities and ids across all candidates for this input.
      values_for_input = top_log_probs[start:end]
      tokens_for_input = top_indices[start:end]

      # Get top-k across all candidates.
      top_candidate_likelihoods, top_candidate_idxs = values_for_input.ravel().topk(beam_size)

      # Map back to which original candidate for this input we're extending.
      orig_candidate_idxs = (top_candidate_idxs // beam_size) + start

      # Update in place.
      candidates[start:end] = candidates[orig_candidate_idxs]
      remaining_masked_positions[start:end] = remaining_masked_positions[orig_candidate_idxs]
      mask_token_ids = mask_ids[orig_candidate_idxs]

      candidates[torch.arange(start, end), mask_token_ids] = tokens_for_input.ravel()[top_candidate_idxs]
      candidate_log_likelihoods[start:end] = top_candidate_likelihoods

    return candidates, candidate_log_likelihoods, remaining_masked_positions

def token_sampling(
    num_inputs,
    candidates,
    candidate_log_likelihoods,
    remaining_masked_positions,
    token_log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots = None,
    initial = False,
    nucleus_prob = None,
    rng = None,
):
  token_probs = token_log_probs.exp()
  if nucleus_prob is not None:
    sorted_probs, sorted_indices = token_probs.sort(dim=1, descending=True)
    mask = sorted_probs.cumsum(dim=1) > nucleus_prob

    flattened_idxs = (sorted_indices + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None])
    zero_indices = flattened_idxs[mask.bool()]

    sample_probs = token_probs.clone()
    sample_probs.ravel()[zero_indices] = 0
    # Hacky way to work around the case where one element has prob greater than nucleus_prob
    sample_probs.ravel()[flattened_idxs[:, 0].flatten()] = token_probs.ravel()[flattened_idxs[:, 0].flatten()]
  else:
    sample_probs = token_probs

  if initial:
    sampled_tokens = torch.multinomial(sample_probs, beam_size, generator=rng)
    flattened_idxs = (sampled_tokens + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None]).ravel()
    sampled_probs = token_probs.ravel()[flattened_idxs].reshape(num_inputs, beam_size)

    mask_ids_repeated = torch.repeat_interleave(mask_ids, repeats=beam_size, dim=0)
    candidates = torch.repeat_interleave(candidates, repeats=beam_size, dim=0)
    candidates[torch.arange(len(candidates)), mask_ids_repeated] = sampled_tokens.ravel()

    candidate_log_likelihoods = probs_update_fxn(
          sampled_probs.log(),
          candidate_log_likelihoods,
          token_probs,
          pivots,
          mask_ids,
    )
    candidate_log_likelihoods = candidate_log_likelihoods.ravel()

    return candidates, candidate_log_likelihoods, remaining_masked_positions
  else:
    sampled_tokens = torch.multinomial(sample_probs, 1, generator=rng)
    flattened_idxs = (sampled_tokens + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None]).ravel()
    sampled_probs = token_probs.ravel()[flattened_idxs].unsqueeze(1)

    candidates[torch.arange(len(candidates)), mask_ids] = sampled_tokens.ravel()

    candidate_log_likelihoods = probs_update_fxn(
        sampled_probs.log(),
        candidate_log_likelihoods,
        token_probs,
        pivots,
        mask_ids,
    ).squeeze()

    return candidates, candidate_log_likelihoods, remaining_masked_positions

@torch.no_grad()
def decode_base(
    input_ids,
    attention_mask,
    beam_size,
    probs_update_fxn,
    probs_to_token_fxn,
    pivots = None,
    best_to_worst = False,
    model = model,
    mask_id = tokenizer.mask_token_id
):
  masked_positions = get_masked_positions(input_ids, mask_id=mask_id)
  remaining_masked_positions = masked_positions.clone()
  num_masked_positions = masked_positions.shape[1]
  num_inputs = len(input_ids)

  log_softmax = nn.LogSoftmax(dim=-1)

  # Get initial pool of candidates by considering first masked position separately.
  initial_logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask).logits
  log_probs = log_softmax(initial_logits)
  if best_to_worst:
    mask_ids, remaining_masked_positions = get_best_masked_positions(
      log_probs,
      remaining_masked_positions,
    )
  else:
    mask_ids = masked_positions[:, 0]
    remaining_masked_positions = remaining_masked_positions[:, 1:]

  log_probs = log_probs[torch.arange(len(log_probs)), mask_ids, :].detach().cpu()
  candidates, candidate_log_likelihoods, remaining_masked_positions = probs_to_token_fxn(
    num_inputs,
    input_ids,
    torch.zeros(len(input_ids)),
    remaining_masked_positions,
    log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots,
    initial=True,
  )

  # Create repeated versions of the various tensors that have num_rows =  num_inputs * beam_size,
  # instead of just num_rows = num_inputs, where each input is repeated beam_size times.
  # Rows (i, i+beam_size) are the current candidates for the i'th input
  if pivots is not None:
    pivots_repeated = torch.repeat_interleave(pivots, repeats=beam_size, dim=0)
  else:
    pivots_repeated = None
  attn_mask_beam_search = torch.repeat_interleave(attention_mask, repeats=beam_size, dim=0)
  masked_positions_repeated = torch.repeat_interleave(masked_positions, repeats=beam_size, dim=0)
  remaining_mask_ids_repeated = torch.repeat_interleave(remaining_masked_positions, repeats=beam_size, dim=0)

  # Do the rest of the beam search given the initial candidates.
  for _  in range(num_masked_positions - 1):
    # Token probabilities for all candidates for all inputs.
    logits = model(input_ids=candidates.to(device), attention_mask=attn_mask_beam_search).logits
    log_probs = log_softmax(logits)
    if best_to_worst:
      mask_ids, remaining_mask_ids_repeated = get_best_masked_positions(
        log_probs,
        remaining_mask_ids_repeated,
      )
    else:
      mask_ids = remaining_mask_ids_repeated[:, 0]
      remaining_mask_ids_repeated = remaining_mask_ids_repeated[:, 1:]

    log_probs = log_probs[torch.arange(len(log_probs)), mask_ids, :].detach().cpu()

    candidates, candidate_log_likelihoods, remaining_mask_ids_repeated = probs_to_token_fxn(
      num_inputs,
      candidates,
      candidate_log_likelihoods,
      remaining_mask_ids_repeated,
      log_probs,
      mask_ids,
      probs_update_fxn,
      beam_size,
      pivots_repeated,
      initial=False,
    )

  probs_by_input = candidate_log_likelihoods.reshape(num_inputs, beam_size)
  completions_by_input = (torch.gather(candidates, 1, masked_positions_repeated)
                          .reshape(num_inputs, beam_size, num_masked_positions))

  # Repackage into same output format as previous functions.
  output = []
  for input_idx in range(num_inputs):
    probs = probs_by_input[input_idx].tolist()
    completions = completions_by_input[input_idx].tolist()
    output.append([[probs[i]] + completions[i] for i in range(beam_size)])
  return output


def decode_sample_LeftToRight_vectorized(
    input_ids,
    attention_mask,
    num_suggestions,
    seed=42,
):
  rng = torch.Generator()
  rng.manual_seed(seed)

  sampling_fxn = partial(token_sampling, rng=rng, nucleus_prob=None)

  return decode_base(
    input_ids,
    attention_mask,
    beam_size=num_suggestions,
    pivots=None,
    probs_update_fxn=pseudo_ll_update,
    probs_to_token_fxn=sampling_fxn,
    best_to_worst=False,
)

def decode_nucleus_LeftToRight_vectorized(
    input_ids,
    attention_mask,
    num_suggestions,
    p=0.9,
    seed=42,
):
  rng = torch.Generator()
  rng.manual_seed(seed)

  sampling_fxn = partial(token_sampling, rng=rng, nucleus_prob=p)

  return decode_base(
    input_ids,
    attention_mask,
    beam_size=num_suggestions,
    pivots=None,
    probs_update_fxn=pseudo_ll_update,
    probs_to_token_fxn=sampling_fxn,
    best_to_worst=False,
)

decode_beam_search = partial(decode_base, probs_to_token_fxn=beam_search)

decode_standard_LeftToRight_vectorized = partial(
  decode_beam_search,
  pivots=None,
  probs_update_fxn=pseudo_ll_update,
  best_to_worst=False,
)

decode_standard_BestToWorst_vectorized = partial(
  decode_beam_search,
  pivots=None,
  probs_update_fxn=pseudo_ll_update,
  best_to_worst=True,
)

def decode_modified_LeftToRight_vectorized(
  input_ids,
  attention_mask,
  beam_size,
):
  N = input_ids.shape[0]
  pivots = torch.full((N, input_ids.shape[1]), tokenizer.mask_token_id)

  return decode_beam_search(
    input_ids,
    attention_mask,
    beam_size,
    pivots=pivots,
    probs_update_fxn=ccr_update,
    best_to_worst=False
  )

def decode_modified_BestToWorst_vectorized(
  input_ids,
  attention_mask,
  beam_size,
):
  N = input_ids.shape[0]
  pivots = torch.full((N, input_ids.shape[1]), tokenizer.mask_token_id)

  return decode_beam_search(
    input_ids,
    attention_mask,
    beam_size,
    pivots=pivots,
    probs_update_fxn=ccr_update,
    best_to_worst=True
  )

decode_pivot_LeftToRight_vectorized = partial(
  decode_beam_search,
  probs_update_fxn=ccr_update,
  best_to_worst=False,
)

decode_pivot_BestToWorst_vectorized = partial(
  decode_beam_search,
  probs_update_fxn=ccr_update,
  best_to_worst=True,
)

def getBLEU(reference, hypothesis):
  return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1,0,0,0))

def score_batch(suggestions_batch, true_ids_batch, method='topk'):
  count = 0 # Track total number of non-padding ids
  num_correct = np.zeros(len(suggestions_batch[0])) # Track top-k accuracy for each k < batchsize
  for suggestions, true_ids in zip(suggestions_batch, true_ids_batch):
    if tokenizer.pad_token_id in true_ids: continue # This is just an instance of padding
    else:
      sorted_suggestions = [s[1:] for s in sorted(suggestions, key=lambda x: x[0], reverse=True)]
      if method=='topk':
        if true_ids in sorted_suggestions:
          num_correct[sorted_suggestions.index(true_ids)] += 1
      elif method=='BLEU':
        num_correct[0] += getBLEU(true_ids, sorted_suggestions[0])
      count += 1

  return count, num_correct

def reconstruct_masked_sentence(masked_ids, replacements):
  replacement_tensor = replacements.flatten()
  reconstructed_tensor = masked_ids.clone()
  indices = torch.nonzero(masked_ids == tokenizer.mask_token_id)
  for i, index in enumerate(indices):
    reconstructed_tensor[tuple(index.tolist())] = replacement_tensor[i]
  return reconstructed_tensor

def score_BLEU_batch(suggestions_batch, true_ids_batch, masked_ids):
  suggestion_ids_batch = [
      sorted(suggestions, key=lambda x: x[0], reverse=True)[0][1:] for suggestions in suggestions_batch
  ]

  true_sentence_ids = reconstruct_masked_sentence(masked_ids, torch.tensor(true_ids_batch))
  sugg_sentence_ids = reconstruct_masked_sentence(masked_ids, torch.tensor(suggestion_ids_batch))

  true_sentences = [tokenizer.convert_ids_to_tokens(true_sentence_id) for true_sentence_id in true_sentence_ids]
  sugg_sentences = [tokenizer.convert_ids_to_tokens(sugg_sentence_id) for sugg_sentence_id in sugg_sentence_ids]


  count = 0
  num_correct = np.zeros(len(suggestions_batch[0])) # Track top-k accuracy for each k < batchsize
  for true_sentence, sugg_sentence in zip(true_sentences, sugg_sentences):
    num_correct[0] += getBLEU(true_sentence, sugg_sentence)
    count += 1

  return count, num_correct

default_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
def compute_bertscore(
    suggestions,
    true_ids_batch,
    masked_inputs_batch,
    masked_positions,
    scorer=default_scorer,
):
    # Currently just looking at top prediction (i.e. top-k = 1)
    preds_for_decode_top_only = [suggestions[i][0][1:]  for i in range(len(suggestions))]
    full_preds = masked_inputs_batch.clone()
    full_preds[:, masked_positions] = torch.tensor(preds_for_decode_top_only)
    preds_text = tokenizer.batch_decode(full_preds, skip_special_tokens=True)

    full_true = masked_inputs_batch.clone()
    full_true[:, masked_positions] = torch.tensor(true_ids_batch)
    true_text = tokenizer.batch_decode(full_true, skip_special_tokens=True)

    # Scorer returns a tuple of 1-D tensors giving precision, recall, and F1
    scores = scorer.score(preds_text, true_text)
    return {
        "precision": scores[0].mean().item(),
        "recall": scores[1].mean().item(),
        "f1": scores[2].mean().item(),
    }

def update_metrics(metrics, key, got):
    if key not in metrics:
        metrics[key] = got
    else:
        for (k, v) in got.items():
            metrics[key][k] += v

# Construct data

data = sep_sents(' '.join(brown.words()), 0, max_toks = context_length)
print("Separated data.")

data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
print("Tokenized data.")


num_total = 0
num_modified_LeftToRight_correct = np.zeros(beam_size) # Track top-k accuracy
num_modified_BestToWorst_correct = np.zeros(beam_size) # Track top-k accuracy
num_standard_LeftToRight_correct = np.zeros(beam_size) # Track top-k accuracy
num_standard_BestToWorst_correct = np.zeros(beam_size) # Track top-k accuracy
bertscore_metrics = defaultdict(dict)

rng = np.random.default_rng(seed=42)

for batch_num in range(num_experiments):
  if batch_num % report_period == 0:
    print("Starting batch", batch_num+1)
    print()
  example_nums = rng.choice(np.arange(len(data.input_ids)), size=num_examples, replace=False)

  input_ids = data.input_ids[example_nums]
  attention_mask = data.attention_mask[example_nums].to(device)
  total = input_ids.shape[1]

  if batch_num % report_period == 0:
    print("Example text:")
    print(' '.join(tokenizer.convert_ids_to_tokens(input_ids[0][:10])))

  mask_start_ind = rng.choice(np.arange(1, total-num_masks))
  if batch_num % report_period == 0:
    print(f"Index: {mask_start_ind} / {total-num_masks}")
  masked_positions = list(range(mask_start_ind,mask_start_ind+num_masks))
  masked_inputs_batch, true_ids_batch = mask_tokens_batch(input_ids, masked_positions)

  suggestions_batch = decode_modified_LeftToRight_vectorized(masked_inputs_batch, attention_mask, beam_size)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, method=method)
  num_modified_LeftToRight_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "modified_left_to_right",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
    )
  )

  suggestions_batch = decode_modified_BestToWorst_vectorized(masked_inputs_batch, attention_mask, beam_size)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, method=method)
  num_modified_BestToWorst_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "modified_best_to_worst",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
    )
  )

  suggestions_batch = decode_standard_LeftToRight_vectorized(masked_inputs_batch, attention_mask, beam_size)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, method=method)
  num_standard_LeftToRight_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "standard_left_to_right",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
    )
  )

  suggestions_batch = decode_standard_BestToWorst_vectorized(masked_inputs_batch, attention_mask, beam_size)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, method=method)
  num_standard_BestToWorst_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "standard_best_to_worst",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
    )
  )

  # Update total count only once:
  num_total += count_batch

  if batch_num % report_period == 0:
    print()
    print(f"Modified Left-to-Right Correct: {num_modified_LeftToRight_correct}/{num_total}")
    print(f"Modified Best-to-Worst Correct: {num_modified_BestToWorst_correct}/{num_total}")
    print(f"Standard Left-To-Right Correct: {num_standard_LeftToRight_correct}/{num_total}")
    print(f"Standard Best-to-Worst Correct: {num_standard_BestToWorst_correct}/{num_total}")
    print()

    # Printing top-k accuracies:
    for k in range(beam_size):
      print(f"Modified Left-to-Right Top-{k+1}: {sum(num_modified_LeftToRight_correct[:k+1])/num_total:.2%}")
      print(f"Modified Best-to-Worst Top-{k+1}: {sum(num_modified_BestToWorst_correct[:k+1])/num_total:.2%}")
      print(f"Standard Left-To-Right Top-{k+1}: {sum(num_standard_LeftToRight_correct[:k+1])/num_total:.2%}")
      print(f"Standard Best-to-Worst Top-{k+1}: {sum(num_standard_BestToWorst_correct[:k+1])/num_total:.2%}")
      print()
    print(f"Total: {num_total}")

for exp_name, results in bertscore_metrics.items():
  for metric, value in bertscore_metrics[exp_name].items():
      bertscore_metrics[exp_name][metric] = value / num_experiments
print(bertscore_metrics)
# -


