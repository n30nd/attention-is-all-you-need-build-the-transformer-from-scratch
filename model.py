"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    vocabs = {}
    curr_len = len(specials)
    specials_list = list(specials)
    for i,special in enumerate(specials):
        vocabs[special] = i

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in specials or word in vocabs:
                continue
            else:
                vocabs[word] = curr_len
                curr_len = curr_len + 1
    return vocabs

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {}
    for key, value in token_to_id.items():
        id_to_token[value] = key
    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    words = sentence.split()
    ids = []
    for word in words:
        if word in token_to_id:
            ids.append(token_to_id[word])
        else:
            ids.append(token_to_id[unk_token])

    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    tokens = []
    for id in ids:
        tokens.append(id_to_token[id])
    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    padded_ids = ids[:max_len]
    while len(padded_ids) < max_len:
        padded_ids.append(pad_id)
    return padded_ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)
    # return torch.mul(embeddings, math.sqrt(d_model))

# Step 8 - compute_positional_div_term
import torch
import math
def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    tensor = torch.exp(torch.arange(0, d_model, 2) * -1 * math.log(10000) / d_model)
    return tensor

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len, dtype=torch.float32).reshape(max_len, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe = pe.clone()
    pe_half_sin = torch.sin(position * div_term)
    d_model = pe.size(1)
    # for j in range(d_model//2):
    #     pe[:,2*j] = pe_half_sin[:,j]
    pe[:,0::2] = pe_half_sin
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe = pe.clone()
    pe_half_cos = torch.cos(position * div_term)

    pe[:,1::2] = pe_half_cos
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch
import math

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix

    # sinusoidal_positional_encoding(max_len, d_model) 
    # -> slice matrix to sin, cos part
    # -> make tensor (max_len, d_model/2) that calculate pos/10000^(2*i/d_model)

    # 1. make tensor (max_len, d_model/2) that calculate pos/10000^(2*i/d_model)
    # Naive: - make tensor 1: (max_len, d_model/2) full pos 
    #        - make tensor 2: (max_len, d_model/2) full 10000^(2*i/d_model)
    #       But you can see the value is duplicat throguh dim, so we can reduce tensor,
    #       Then, pytorch will broadcast it
    #       - tensor 1 (max_len, 1)
    #       - tensor 2 (1, d_model/2)
    position_index_column = torch.arange(max_len, dtype=torch.float32).reshape(max_len, 1)
    positional_div_term = torch.exp(torch.arange(0, d_model, 2) * -1 * math.log(10000) / d_model)
    
    tensor1 = position_index_column * positional_div_term

    # Fit into pe
    pe = torch.zeros(max_len, d_model)
    
    # Fit sin
    pe[:, 0::2] = torch.sin(tensor1)
    
    # Fit cos
    pe[:, 1::2] = torch.cos(tensor1)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    L = embedded_batch.size(1)
    return embedded_batch + positional_encoding[:L,:]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    padding_mask = token_ids != pad_id
    B = token_ids.size(0)
    L = token_ids.size(1)
    return padding_mask.reshape(B, 1, 1, L)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    # A = torch.arange(seq_len).reshape(seq_len, 1)
    # B = torch.arange(seq_len).reshape(1, seq_len)

    # casual_mask = A >= B
    casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    return casual_mask.reshape(1, 1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    combined_mask = padding_mask & causal_mask
    return combined_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return torch.einsum('...md, ...nd -> ...mn', query, key)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(mask == False, -float('inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights = F.softmax(masked_scores, dim = -1)
    all_masked = (~torch.isfinite(masked_scores)).all(dim=-1, keepdim=True)
    weights = weights.masked_fill(all_masked == True, 0.0)

    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    # return torch.einsum('...mn, ...nk -> ...mk', attention_weights, value)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, -float('inf'))
    
    weights = F.softmax(scores, dim=-1)
    all_masked = (~torch.isfinite(scores)).all(dim=-1, keepdim=True)
    weights = weights.masked_fill(all_masked == True, 0.0)
    return (weights @ value, weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.size()
    d_k = d_model // num_heads
    return tensor.reshape(B, L, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    B, num_heads, L, d_k = multi_head_tensor.shape
    # Step 1: move L back in front of the head axis -> (B, L, num_heads, d_k)
    x = multi_head_tensor.transpose(1, 2)
    # Step 2: make contiguous, then merge the last two axes -> (B, L, num_heads * d_k)
    return x.reshape(B, L, num_heads * d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is not None:
        return x @ weight.transpose(-1, -2) + bias
    else:
        return x @ weight.transpose(-1, -2)

# Step 27 - project_to_query_key_value
import torch

def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    y = x @ weight.transpose(-1, -2)
    if bias is not None:
        y += bias
    return y
    
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.size()
    d_h = d_model // num_heads
    return tensor.reshape(B, L, num_heads, d_h)

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.permute(0, 2, 1, 3)


def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    out_q = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    out_v = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    out_k = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))

    return out_q, out_k, out_v

# Step 29 - multi_head_scaled_dot_product_attention
import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, -float('inf'))
    
    weights = F.softmax(scores, dim=-1)
    all_masked = (~torch.isfinite(scores)).all(dim=-1, keepdim=True)
    weights = weights.masked_fill(all_masked == True, 0.0)
    return (weights @ value, weights)

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
def merge_heads_back_to_model_dim(multi_head_tensor):
    B, num_heads, L, d_k = multi_head_tensor.shape
    # Step 1: move L back in front of the head axis -> (B, L, num_heads, d_k)
    x = multi_head_tensor.transpose(1, 2)
    # Step 2: make contiguous, then merge the last two axes -> (B, L, num_heads * d_k)
    # return x.contiguous().reshape(B, L, num_heads * d_k)
    return x.reshape(B, L, num_heads * d_k)

def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    y = x @ weight.transpose(-1, -2)
    if bias is not None:
        y += bias
    return y

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merge_head_tensor = merge_heads_back_to_model_dim(context)
    output_tensor = apply_linear_projection(merge_head_tensor, w_o, b_o)
    return output_tensor

# Step 31 - assemble_multi_head_attention_forward
import math
import torch

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = query @ w_q # (B, Tq, D) -> (B, Tq, D)
    k = key @ w_k
    v = value @ w_v

    # (B, Tq, D) -> (B, Tq, H, d_k),d_k = D/H -> (B, H, T, d_k)
    B, T, D = q.size()
    q = q.reshape(B, T, num_heads, D // num_heads).transpose(1,2)
    B, T, D = k.size()
    k = k.reshape(B, T, num_heads, D // num_heads).transpose(1,2)
    B, T, D = v.size()
    v = v.reshape(B, T, num_heads, D // num_heads).transpose(1,2)

    # Attention
    d_k = q.size(-1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_k) # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask == False, -float('inf'))
    
    weights = torch.softmax(scores, dim=-1) # (B, H, Tq, Tk)
    all_masked = (~scores.isfinite()).all(dim=-1, keepdim=True)
    weights = weights.masked_fill(all_masked == True, 0.0)


    head_output = weights @ v # (B, H, Tq, d_k)
    B, H, Tq, d_k = head_output.size()
    merged_head = head_output.transpose(1,2).reshape(B, Tq, H*d_k)

    context = merged_head @ w_o

    return context

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(x@w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    first_linear = apply_ffn_first_linear_and_relu(x, w1, b1)
    second_linear = apply_ffn_second_linear(first_linear, w2, b2)
    return second_linear

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mu = torch.mean(x, dim=-1, keepdim=True)
    x_mu_2 = torch.square(x-mu)
    variance = torch.mean(x_mu_2, dim=-1, keepdim=True)
    return mu, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mu, var = compute_layer_norm_mean_and_variance(x)
    x_hat = (x-mu) / torch.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    x_mask = x.masked_fill_(keep_mask == False, 0.0)
    x_scale = x / keep_prob
    return x_scale

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    out_mha = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    out_norm = apply_residual_add_and_norm(x, out_mha, gamma, beta)
    return out_norm

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    out_ffn = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    out_norm = apply_residual_add_and_norm(x, out_ffn, gamma, beta)
    return out_norm

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    h = encoder_layer_self_attention_sublayer(x, layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], layer_params['attn_gamma'], layer_params['attn_beta'], num_heads, src_mask) 
    y = encoder_layer_feed_forward_sublayer(h, layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta'])
    return y

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    num_layers = len(encoder_layer_params_list)
    h = x
    for i in range(num_layers):
        h = assemble_encoder_layer(h, encoder_layer_params_list[i], num_heads, src_mask)
    return h

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    mha = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    out_norm = apply_residual_add_and_norm(y, mha, gamma, beta)
    return out_norm

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    # src_mask is (B, T_src); reshape to (B, 1, 1, T_src) so it broadcasts over heads and query positions
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]
    mha = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    out_norm = apply_residual_add_and_norm(y, mha, gamma, beta)
    return out_norm

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    out_ffn = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    out_norm = apply_residual_add_and_norm(y, out_ffn, gamma, beta)
    return out_norm

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # 1) Masked self-attention: Q/K/V all come from y, uses tgt_mask (causal + pad).
    h = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params['w_q_self'], layer_params['w_k_self'],
        layer_params['w_v_self'], layer_params['w_o_self'],
        layer_params['self_gamma'], layer_params['self_beta'],
        num_heads, tgt_mask,
    )

    # 2) Cross-attention: Q from h, K/V from encoder_output, uses src_mask.
    c = decoder_layer_cross_attention_sublayer(
        h, encoder_output,
        layer_params['w_q_cross'], layer_params['w_k_cross'],
        layer_params['w_v_cross'], layer_params['w_o_cross'],
        layer_params['cross_gamma'], layer_params['cross_beta'],
        num_heads, src_mask,
    )

    # 3) Position-wise feed-forward, no mask.
    out = decoder_layer_feed_forward_sublayer(
        c,
        layer_params['w1'], layer_params['b1'],
        layer_params['w2'], layer_params['b2'],
        layer_params['ffn_gamma'], layer_params['ffn_beta'],
    )
    return out

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    num_layers = len(decoder_layer_params_list)
    h = y
    for layer in range(num_layers):
        h = assemble_decoder_layer(h, encoder_output, decoder_layer_params_list[layer], num_heads, src_mask, tgt_mask)
    return h

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V). weight (vocab_size, d_model)
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.nn.functional.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
import torch.nn as nn


def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    # 1.1 Token embedding
    token_embedding = model_params['token_embedding'] # (V, D)
    V, D = token_embedding.shape
    # embedding = nn.Embedding(V, D, _weight=token_embedding)

    # x = embedding(src_ids) # (B, S, D)
    # y = embedding(tgt_ids) # (B, T, D)
    x = token_embedding[src_ids]
    y = token_embedding[tgt_ids]

    # 1.2 Scale
    x = scale_embeddings_by_sqrt_d_model(x, D)
    y = scale_embeddings_by_sqrt_d_model(y, D)

    # 1.3 PE
    pe = build_sinusoidal_positional_encoding(max(x.size(1), y.size(1)), D)
    x = add_positional_encoding_to_embeddings(x, pe)
    y = add_positional_encoding_to_embeddings(y, pe)


    # Build masks
    src_mask = build_padding_mask(src_ids, pad_id) # (B, 1, 1, S)

    T = tgt_ids.size(-1)
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id) # (B, 1, 1, T)
    tgt_casual_mask = build_causal_mask(T) # (1, 1, T, T)
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, tgt_casual_mask)

    # Encode and Decode
    encoder_output = stack_encoder_layers(x, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(y, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask) # (B, T, D)

    # Project output
    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    log_probabilities = apply_log_softmax_over_vocab(logits)

    return log_probabilities

# Step 52 - init_encoder_layer_parameters
import torch
import math

def xavier_normal(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return (torch.randn(fan_in, fan_out, dtype=torch.float32) * std).requires_grad_(True)


def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    d = {}


    # Self-attention projections
    d["w_q"] = xavier_normal(d_model, d_model)
    d["w_k"] = xavier_normal(d_model, d_model)
    d["w_v"] = xavier_normal(d_model, d_model)
    d["w_o"] = xavier_normal(d_model, d_model)

    # Feed-forward network
    d["w1"] = xavier_normal(d_model, d_ff)
    d["b1"] = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)

    d["w2"] = xavier_normal(d_ff, d_model)
    d["b2"] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    # LayerNorm before/around attention block
    d["attn_gamma"] = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    d["attn_beta"] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    # LayerNorm before/around FFN block
    d["ffn_gamma"] = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    d["ffn_beta"] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return d

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    d = {}
    d['w_q_self'] = xavier_normal(d_model, d_model)
    d['w_k_self'] = xavier_normal(d_model, d_model)
    d['w_v_self'] = xavier_normal(d_model, d_model)
    d['w_o_self'] = xavier_normal(d_model, d_model)

    d['w_q_cross'] = xavier_normal(d_model, d_model)
    d['w_k_cross'] = xavier_normal(d_model, d_model)
    d['w_v_cross'] = xavier_normal(d_model, d_model)
    d['w_o_cross'] = xavier_normal(d_model, d_model)

    d['w1'] = xavier_normal(d_model, d_ff)
    d['b1'] = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    d['w2'] = xavier_normal(d_ff, d_model)
    d['b2'] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    d['self_gamma'] = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    d['self_beta'] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    d['cross_gamma'] = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    d['cross_beta'] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    d['ffn_gamma'] = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    d['ffn_beta'] = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return d

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    d = {}
    d['src_embedding'] = xavier_normal(vocab_size, d_model)
    d['tgt_embedding'] = xavier_normal(vocab_size, d_model)
    if tie_weights:
        d['output_projection'] = d['tgt_embedding']
    else:
        d['output_projection'] = xavier_normal(vocab_size, d_model)
    return d

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    seen = set()
    params = []
    for layer in encoder_layer_params:
        for value in layer.values():
            if id(value) not in seen:
                seen.add(id(value))
                params.append(value)
    
    for layer in decoder_layer_params:
        for value in layer.values():
            if id(value) not in seen:
                seen.add(id(value))
                params.append(value)

    for value in embedding_params.values():
        if id(value) not in seen:
            seen.add(id(value))
            params.append(value)

    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    out = target_ids.clone()
    out[:, 0] = start_token_id
    out[:, 1:] = target_ids[:,:-1]

    return out

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    rate1 = step ** (-1/2)
    rate2 = step * warmup_steps ** (-3/2)
    return d_model ** (-1/2) * min(rate1, rate2)

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.full(shape, epsilon / (vocab_size - 2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution: torch.Tensor, gold_token_ids: torch.Tensor, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    distribution = smoothed_distribution.clone()

    # Naive
    B, T = gold_token_ids.size()
    # for b in range(B):
    #     for t in range(T):
    #         gold_token_id = gold_token_ids[b,t]
    #         distribution[b, t, gold_token_id] = confidence
    # return distribution

    distribution.scatter_(2, gold_token_ids.unsqueeze(-1), confidence)

    return distribution

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
   smoothed_distribution[:,:, pad_id] = 0

   # B, T = gold_token_ids.size()
   # for b in range(B):
   #    for t in range(T):
   #       if gold_token_ids[b, t] == pad_id:
   #          smoothed_distribution[b, t, :] = 0

   mask = gold_token_ids == pad_id
   smoothed_distribution.masked_fill_(mask.unsqueeze(-1), 0)
   
   return smoothed_distribution

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    q_logp = smoothed_distribution * log_probabilities
    loss = (-1 * q_logp).sum()
    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    count_normal_token = (gold_token_ids != pad_id).sum()
    return total_loss / max(count_normal_token, 1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    y_hat = log_probabilities.argmax(dim=-1)
    div_term = (gold_token_ids != pad_id).sum()
    acc = torch.tensor(0.0) if div_term == 0 else ((y_hat == gold_token_ids) * (gold_token_ids != pad_id)).sum() / div_term
    return acc

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    state = {}
    state['m'] = []
    state['v'] = []
    state['t'] = 0

    for param in parameter_list:
        state['m'].append(torch.zeros_like(param))
        state['v'].append(torch.zeros_like(param))

    return state

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    m_curr = beta1 * m_prev + (1 - beta1)*grad.detach()
    return m_curr

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    v_t = beta2 * v_prev + (1-beta2)*grad.detach()**2
    return v_t

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat_t = m_t / (1 - beta1 ** step)
    v_hat_t= v_t / (1 - beta2 ** step)
    return m_hat_t, v_hat_t

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # 1. increment the shared step counter once
    optimizer_state['t'] += 1
    t = optimizer_state['t']

    for i in range(len(parameter_list)):
        param = parameter_list[i]
        if param.grad is None:
            continue

        # 2. update this parameter's moment buffers
        optimizer_state['m'][i] = update_adam_first_moment(optimizer_state['m'][i], param.grad, beta1)
        optimizer_state['v'][i] = update_adam_second_moment(optimizer_state['v'][i], param.grad, beta2)

        # 3. bias correction
        m_hat, v_hat = apply_adam_bias_correction(
            optimizer_state['m'][i], optimizer_state['v'][i], beta1, beta2, t
        )

        # 4. write back without touching autograd
        with torch.no_grad():
            param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
def run_transformer_forward_v2(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    # 1.1 Token embedding
    # token_embedding = model_params['token_embedding'] # (V, D)
    src_embedding = model_params['src_embedding']
    tgt_embedding = model_params['tgt_embedding']
    V, D = src_embedding.shape

    # x = embedding(src_ids) # (B, S, D)
    # y = embedding(tgt_ids) # (B, T, D)
    x = src_embedding[src_ids]
    y = tgt_embedding[tgt_ids]

    # 1.2 Scale
    x = scale_embeddings_by_sqrt_d_model(x, D)
    y = scale_embeddings_by_sqrt_d_model(y, D)

    # 1.3 PE
    pe = build_sinusoidal_positional_encoding(max(x.size(1), y.size(1)), D)
    x = add_positional_encoding_to_embeddings(x, pe)
    y = add_positional_encoding_to_embeddings(y, pe)


    # Build masks
    src_mask = build_padding_mask(src_ids, pad_id) # (B, 1, 1, S)

    T = tgt_ids.size(-1)
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id) # (B, 1, 1, T)
    tgt_casual_mask = build_causal_mask(T) # (1, 1, T, T)
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, tgt_casual_mask)

    # Encode and Decode
    encoder_output = stack_encoder_layers(x, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(y, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask) # (B, T, D)

    # Project output
    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    log_probabilities = apply_log_softmax_over_vocab(logits)

    return log_probabilities


def build_smoothed_distribution(gold_token_ids: torch.Tensor, config):
    B, T = gold_token_ids.size()
    
    # Init distribution
    uniform_distribution = build_uniform_smoothing_distribution((B, T, config['vocab_size']), config['vocab_size'], config['smoothing'])
    
    # Set confidence
    distribution = set_confidence_on_gold_tokens(uniform_distribution, gold_token_ids, 1 - config['smoothing'])

    # Zero pad
    distribution = zero_pad_column_and_pad_token_rows(distribution, gold_token_ids, config['pad_id'])

    return distribution

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    shift_tgt_batch = shift_targets_right_with_start_token(tgt_batch, config['start_id'])
    log_probabilities = run_transformer_forward_v2(src_batch, shift_tgt_batch, model_params, config['num_heads'], config['pad_id'])

    smoothed_distribution = build_smoothed_distribution(tgt_batch, config)

    loss = compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution)
    return loss

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # TODO: zero grads, compute loss, backward, look up Noam LR, apply Adam step
    zero_all_parameter_gradients(parameter_list)

    loss = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)

    loss.backward()

    lr = compute_noam_learning_rate(step_number, config['d_model'], config['warmup_steps'])

    optimizer_state = apply_adam_step_to_all_parameters(parameter_list, optimizer_state, lr)

    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    # TODO: iterate for num_steps steps, calling run_training_step_with_backprop each time
    l = []
    for t in range(1, num_steps+1):
        src, tgt = batches[(t-1) % len(batches)]
        loss = run_training_step_with_backprop(src, tgt, parameter_list, model_params, optimizer_state, t, config)
        l.append(loss)
    return l

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    return torch.argmax(final_step_logits, dim=-1, keepdim=False)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5 + sequence_length) / 6) ** alpha

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    return beam_scores.unsqueeze(-1) + next_token_log_probs

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    V = candidate_scores.size(-1)
    flat = candidate_scores.reshape(-1)
    scores, flat_idx = torch.topk(flat, k)
    beam_indices = flat_idx // V
    token_ids = flat_idx % V

    return {
        'beam_indices': beam_indices,
        'token_ids': token_ids,
        'scores': scores
    }

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    parents = beam_sequences[beam_indices]
    return torch.cat([parents, token_ids.unsqueeze(-1)], dim=-1)

# Step 79 - mark_finished_beams
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    # TODO: return updated boolean finished flags for each beam given the new token ids
    mask = token_ids == end_token_id
    return mask | finished_flags

# Step 80 - select_best_finished_beam
def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    # TODO: return the finished beam with the highest length-penalized score
    len_seqs = torch.tensor([finished_sequence.size(-1) for finished_sequence in finished_sequences])

    penalties = compute_length_penalty(len_seqs, alpha)
    norm_score = torch.tensor(finished_scores) / penalties

    # max_score = max(norm_score)
    # idx_max_score = norm_score.index(max_score)
    idx_max_score = torch.argmax(norm_score, dim=-1)

    return {
        'sequence': finished_sequences[idx_max_score],
        'score': norm_score[idx_max_score]
    }

