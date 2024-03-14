from functools import partial

import torch

from .core import decode_base
from .probs_update import ccr_update, pseudo_ll_update
from .token_samplers import beam_search, token_sampling


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
  mask_id
):
  N = input_ids.shape[0]
  pivots = torch.full((N, input_ids.shape[1]), mask_id)

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
  mask_id,
):
  N = input_ids.shape[0]
  pivots = torch.full((N, input_ids.shape[1]), mask_id)

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
