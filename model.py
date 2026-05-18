import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaTokenizer


AGGREGATION_METHODS = ("attention", "top1", "mean", "max")


class SampLayer(nn.Module):

    def __init__(
        self,
        a_emb,
        n_emb,
        hidden_size,
        device,
        attn_type="softmax",
        aggregation_method="attention"
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.n_emb = n_emb
        self.attn_type = attn_type
        self.aggregation_method = self._normalize_aggregation_method(aggregation_method)

        # Q, K, V projections
        # d_q = d_k = n_emb, d_v = d = hidden_size
        self.query_lin = nn.Linear(hidden_size, n_emb)
        self.key_lin = nn.Linear(hidden_size, n_emb)
        self.value_lin = nn.Linear(hidden_size, hidden_size)

        # Relevance scoring:
        # e_i = w^T tanh(W_u u_i)
        # W_u in R^{d' x d}, w in R^{d'}, with d' = a_emb = 256 by default.
        self.W_u = nn.Linear(hidden_size, a_emb, bias=True)
        self.w = nn.Linear(a_emb, 1, bias=False)

    @staticmethod
    def _normalize_aggregation_method(aggregation_method):
        aliases = {
            "attentive": "attention",
            "attentive_aggregation": "attention",
            "top-1": "top1",
            "top_1": "top1",
            "top1_neighbor": "top1",
            "mean_pooling": "mean",
            "max_pooling": "max",
        }
        method = str(aggregation_method).strip().lower()
        method = aliases.get(method, method)
        if method not in AGGREGATION_METHODS:
            raise ValueError(
                "aggregation_method must be one of "
                f"{AGGREGATION_METHODS}, got {aggregation_method}"
            )
        return method

    def _to_long_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        return x.to(self.device)

    def _encode_single_neighbor_response(self, center_fea, token_n, mask_n, neighbert):

        token_n = self._to_long_tensor(token_n).unsqueeze(0)  # [1, L]
        mask_n = self._to_long_tensor(mask_n).unsqueeze(0)    # [1, L]

        with torch.no_grad():
            emb_n = neighbert(
                input_ids=token_n,
                attention_mask=mask_n
            ).last_hidden_state  # [1, L, hidden]

        Q_r = self.query_lin(center_fea)  # [1, span_len, d_q]
        K_n = self.key_lin(emb_n)         # [1, L, d_k]
        V_n = self.value_lin(emb_n)       # [1, L, d_v=hidden]

        # Keep center-to-neighbor attention from attending to padded neighbor tokens.
        # For scaled_dot_product_attention, True means the position is allowed.
        key_mask = mask_n.bool().unsqueeze(1)  # [1, 1, L], broadcast over center span
        Z_i = F.scaled_dot_product_attention(
            query=Q_r,
            key=K_n,
            value=V_n,
            attn_mask=key_mask
        )  # [1, span_len, hidden]

        return Z_i

    def _compute_neighbor_weights(self, responses):

        if len(responses) == 0:
            return None

        # u_i = mean over token dimension of Z_i
        u_list = [z.mean(dim=1) for z in responses]  # each [1, hidden]
        U = torch.cat(u_list, dim=0)                 # [num_neighbors, hidden]

        # e_i = w^T tanh(W_u u_i)
        E = self.w(torch.tanh(self.W_u(U)))          

        if self.attn_type != "softmax":
            raise ValueError(
                f"SANI paper uses softmax for neighbor weighting, got attn_type={self.attn_type}"
            )

        beta = F.softmax(E, dim=0)                   
        return beta.view(-1, 1, 1)                  

    def _aggregate_neighbor_summary(self, center_fea, neigh_ids_list, neigh_mask_list, neighbert):

        responses = []

        for token_n, mask_n in zip(neigh_ids_list, neigh_mask_list):
            Z_i = self._encode_single_neighbor_response(
                center_fea=center_fea,
                token_n=token_n,
                mask_n=mask_n,
                neighbert=neighbert
            )
            responses.append(Z_i)

        if len(responses) == 0:
            return torch.zeros_like(center_fea)

        if self.aggregation_method == "top1":
            return responses[0]

        stacked = torch.cat(responses, dim=0)  # [num_neighbors, span_len, hidden]

        if self.aggregation_method == "mean":
            C_r = stacked.mean(dim=0, keepdim=True)
        elif self.aggregation_method == "max":
            C_r = stacked.max(dim=0, keepdim=True).values
        else:
            beta = self._compute_neighbor_weights(responses)  # [num_neighbors, 1, 1]
            C_r = (beta * stacked).sum(dim=0, keepdim=True)   # [1, span_len, hidden]

        return C_r

    @staticmethod
    def _replace_span_without_inplace(x, start, end, replacement):
        return torch.cat(
            [
                x[:start, :],
                replacement,
                x[end:, :]
            ],
            dim=0
        )

    def inject_neighbors(self, x_n, b_s, xs, neighbert, entity_pos_list):

        updated_xs = []

        for b in range(b_s):
            start, end = entity_pos_list[b]
            x = xs[b]  # [seq_len, hidden]

            center_fea = x[start:end, :].unsqueeze(0)  # [1, span_len, hidden]

            neigh_ids_list = x_n[b].get("neighbors_input_ids", [])
            neigh_mask_list = x_n[b].get("neighbors_attention_mask", [])

            C_r = self._aggregate_neighbor_summary(
                center_fea=center_fea,
                neigh_ids_list=neigh_ids_list,
                neigh_mask_list=neigh_mask_list,
                neighbert=neighbert
            )  # [1, span_len, hidden]

            # Residual fusion: H_tilde_r = H_r + C_r
            updated_span = x[start:end, :] + C_r.squeeze(0)
            updated_xs.append(
                self._replace_span_without_inplace(x, start, end, updated_span)
            )

        return torch.stack(updated_xs, dim=0)

    def inject_pair_neighbors(self, x_n, b_s, xs, neighbert, entity_pos_list):

        updated_xs = []

        for b in range(b_s):
            (e1_start, e1_end), (e2_start, e2_end) = entity_pos_list[b]
            x = xs[b]  # [seq_len, hidden]

            # Entity 1
            e1_fea = x[e1_start:e1_end, :].unsqueeze(0)  # [1, span1_len, hidden]
            neigh1_ids_list = x_n[b].get("neigh1_input_ids", [])
            neigh1_mask_list = x_n[b].get("neigh1_attention_mask", [])

            C_r1 = self._aggregate_neighbor_summary(
                center_fea=e1_fea,
                neigh_ids_list=neigh1_ids_list,
                neigh_mask_list=neigh1_mask_list,
                neighbert=neighbert
            )  # [1, span1_len, hidden]

            # Entity 2
            e2_fea = x[e2_start:e2_end, :].unsqueeze(0)  # [1, span2_len, hidden]
            neigh2_ids_list = x_n[b].get("neigh2_input_ids", [])
            neigh2_mask_list = x_n[b].get("neigh2_attention_mask", [])

            C_r2 = self._aggregate_neighbor_summary(
                center_fea=e2_fea,
                neigh_ids_list=neigh2_ids_list,
                neigh_mask_list=neigh2_mask_list,
                neighbert=neighbert
            )  # [1, span2_len, hidden]

            # Residual fusion
            e1_updated = x[e1_start:e1_end, :] + C_r1.squeeze(0)
            e2_updated = x[e2_start:e2_end, :] + C_r2.squeeze(0)

            updated_xs.append(
                torch.cat(
                    [
                        x[:e1_start, :],
                        e1_updated,
                        x[e1_end:e2_start, :],
                        e2_updated,
                        x[e2_end:, :]
                    ],
                    dim=0
                )
            )

        return torch.stack(updated_xs, dim=0)


class ScholarSampForPretrain(nn.Module):
    def __init__(
        self,
        device,
        num_fields,
        n_emb=256,
        a_emb=256,
        dropout=0.1,
        attn_type="softmax",
        pretrained_name="xlm-roberta-base",
        tam_class_weights=None,
        mfp_pos_weight=None,
        use_sani=True,
        aggregation_method="attention"
    ):
        super().__init__()

        self.device = device
        self.num_fields = num_fields
        self.use_sani = bool(use_sani)

        # Shared encoder, aligned with the paper description
        self.language_model = XLMRobertaModel.from_pretrained(pretrained_name)
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            self.language_model.gradient_checkpointing_enable()
        self.neighbert = self.language_model
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_name)

        self.hidden_size = self.language_model.config.hidden_size

        self.samp_layer = SampLayer(
            a_emb=a_emb,
            n_emb=n_emb,
            hidden_size=self.hidden_size,
            device=device,
            attn_type=attn_type,
            aggregation_method=aggregation_method
        )
        self.post_neighbor_self_attn = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.language_model.config.num_attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

        # MFP head
        self.mfp_entity_proj = nn.Linear(self.hidden_size, a_emb)
        self.mfp_field_proj = nn.Linear(self.hidden_size, a_emb)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        if mfp_pos_weight is None:
            self.loss_fct_mfp = nn.BCEWithLogitsLoss()
        else:
            mfp_pos_weight = torch.tensor(
                mfp_pos_weight,
                dtype=torch.float,
                device=device
            )
            self.loss_fct_mfp = nn.BCEWithLogitsLoss(pos_weight=mfp_pos_weight)

        # HPC head: Homonym Pair Classification
        self.hpc_linear1 = nn.Linear(self.hidden_size, a_emb)
        self.hpc_linear2 = nn.Linear(a_emb, 2)
        self.loss_fct_hpc = nn.CrossEntropyLoss()

        # TAM head
        self.tam_linear1 = nn.Linear(self.hidden_size, a_emb)
        self.tam_linear2 = nn.Linear(a_emb, 2)
        if tam_class_weights is None:
            self.loss_fct_tam = nn.CrossEntropyLoss()
        else:
            tam_class_weights = torch.tensor(
                tam_class_weights,
                dtype=torch.float,
                device=device
            )
            self.loss_fct_tam = nn.CrossEntropyLoss(weight=tam_class_weights)

        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def apply_post_neighbor_self_attention(self, hidden_states, attention_mask):
        key_padding_mask = attention_mask == 0
        return self.post_neighbor_self_attn(
            hidden_states,
            src_key_padding_mask=key_padding_mask
        )

    def get_mfp_position_repr(self, hidden_states, mfp_field_spans=None):
        reprs = []
        for b in range(hidden_states.shape[0]):
            spans = [] if mfp_field_spans is None else mfp_field_spans[b]
            span_reprs = []
            for start, end in spans:
                start = max(int(start), 0)
                end = min(int(end), hidden_states.shape[1])
                if end > start:
                    span_reprs.append(hidden_states[b, start:end, :].mean(dim=0))
            if not span_reprs:
                reprs.append(hidden_states[b, 0, :])
            else:
                reprs.append(torch.stack(span_reprs, dim=0).mean(dim=0))

        return torch.stack(reprs, dim=0)

    def encode_candidate_fields(self, field_input_ids, field_attention_mask):
        batch_size, num_fields, field_len = field_input_ids.shape
        flat_ids = field_input_ids.view(batch_size * num_fields, field_len)
        flat_mask = field_attention_mask.view(batch_size * num_fields, field_len)

        output = self.language_model(
            input_ids=flat_ids,
            attention_mask=flat_mask
        ).last_hidden_state

        field_repr = output[:, 0, :]
        field_repr = self.mfp_field_proj(self.drop(self.gelu(field_repr)))
        field_repr = F.normalize(field_repr, p=2, dim=-1)
        return field_repr.view(batch_size, num_fields, -1)

    def forward_mfp(
        self,
        x,
        x_n,
        att_mask,
        entity_pos_list,
        mfp_labels=None,
        mfp_field_spans=None,
        field_input_ids=None,
        field_attention_mask=None
    ):
        output = self.language_model(
            input_ids=x,
            attention_mask=att_mask
        ).last_hidden_state  # [B, L, H]

        if self.use_sani:
            output = self.samp_layer.inject_neighbors(
                x_n=x_n,
                b_s=x.shape[0],
                xs=output,
                neighbert=self.neighbert,
                entity_pos_list=entity_pos_list
            )  # [B, L, H]
            output = self.apply_post_neighbor_self_attention(output, att_mask)

        mask_repr = self.get_mfp_position_repr(output, mfp_field_spans)  # [B, H]

        entity_embeds = self.mfp_entity_proj(self.drop(self.gelu(mask_repr)))  # [B, a_emb]
        entity_embeds = F.normalize(entity_embeds, p=2, dim=-1)

        if field_input_ids is None or field_attention_mask is None:
            raise ValueError("MFP requires field_input_ids and field_attention_mask for candidate field embeddings.")

        field_embeds = self.encode_candidate_fields(field_input_ids, field_attention_mask)  # [B, M, a_emb]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.bmm(field_embeds, entity_embeds.unsqueeze(-1)).squeeze(-1)

        if mfp_labels is not None:
            mfp_labels = mfp_labels.float()
            loss = self.loss_fct_mfp(logits, mfp_labels)
            return loss, logits

        return logits

    def encode_pair_with_neighbors(self, x, x_n, att_mask, entity_pos_list):
        output = self.language_model(
            input_ids=x,
            attention_mask=att_mask
        ).last_hidden_state  # [B, L, H]

        if self.use_sani:
            output = self.samp_layer.inject_pair_neighbors(
                x_n=x_n,
                b_s=x.shape[0],
                xs=output,
                neighbert=self.neighbert,
                entity_pos_list=entity_pos_list
            )  # [B, L, H]
            output = self.apply_post_neighbor_self_attention(output, att_mask)

        return output[:, 0, :]  # [B, H]

    def forward_hpc(self, x, x_n, att_mask, entity_pos_list, pair_labels=None):
        pair_repr = self.encode_pair_with_neighbors(
            x=x,
            x_n=x_n,
            att_mask=att_mask,
            entity_pos_list=entity_pos_list
        )

        logits = self.hpc_linear2(
            self.drop(
                self.gelu(
                    self.hpc_linear1(pair_repr)
                )
            )
        )  # [B, 2]

        if pair_labels is not None:
            loss = self.loss_fct_hpc(logits, pair_labels.long())
            return loss, logits

        return logits

    def forward_tam(self, x, x_n, att_mask, entity_pos_list, pair_labels=None):
        output = self.language_model(
            input_ids=x,
            attention_mask=att_mask
        ).last_hidden_state  # [B, L, H]

        if self.use_sani:
            output = self.samp_layer.inject_pair_neighbors(
                x_n=x_n,
                b_s=x.shape[0],
                xs=output,
                neighbert=self.neighbert,
                entity_pos_list=entity_pos_list
            )  # [B, L, H]
            output = self.apply_post_neighbor_self_attention(output, att_mask)

        pair_repr = output[:, 0, :]  # [B, H]

        logits = self.tam_linear2(
            self.drop(
                self.gelu(
                    self.tam_linear1(pair_repr)
                )
            )
        )  # [B, 2]

        if pair_labels is not None:
            loss = self.loss_fct_tam(logits, pair_labels.long())
            return loss, logits

        return logits

    def forward(
        self,
        task_type,
        x,
        x_n,
        att_mask,
        entity_pos_list,
        mfp_labels=None,
        pair_labels=None,
        mfp_field_spans=None,
        field_input_ids=None,
        field_attention_mask=None
    ):
        if task_type == "mfp":
            return self.forward_mfp(
                x=x,
                x_n=x_n,
                att_mask=att_mask,
                entity_pos_list=entity_pos_list,
                mfp_labels=mfp_labels,
                mfp_field_spans=mfp_field_spans,
                field_input_ids=field_input_ids,
                field_attention_mask=field_attention_mask
            )
        if task_type == "hpc":
            return self.forward_hpc(
                x=x,
                x_n=x_n,
                att_mask=att_mask,
                entity_pos_list=entity_pos_list,
                pair_labels=pair_labels
            )
        if task_type == "tam":
            return self.forward_tam(
                x=x,
                x_n=x_n,
                att_mask=att_mask,
                entity_pos_list=entity_pos_list,
                pair_labels=pair_labels
            )

        raise ValueError(f"Unsupported task_type: {task_type}")
