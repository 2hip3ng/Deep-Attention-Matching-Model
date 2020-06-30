import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from utils import load_embedding

LayerNorm = torch.nn.LayerNorm


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(args.embedding_dropout_prob)

        embedding = load_embedding(args)
        self.word_embeddings.weight.requires_grad = args.fix_embedding    # False
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding))

    def forward(self, input_ids):
        word_embeddings = self.word_embeddings(input_ids)
        word_embeddings = self.dropout(word_embeddings)
        return word_embeddings


class SelfAttLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(args.attention_dropout_prob)

    def transpose_for_scores(self, x):
        # x : batch_size * max_seq * dim
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # new_x_shape: batch_size * max_seq * attention_heads * head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # return shape: batch_size * attention_heads * max_seq * head_size

    def forward(self, hidden_states, attention_mask):
        # hidden_states_a: batch_size * max_seq_a * embedding_dim
        # hidden_states_b: batch_size * max_seq_b * embedding_dim

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        # Self-Attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # batch_size * attention_heads * max_seq_a * max_seq_b
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.dense_2 = nn.Linear(args.intermediate_size, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.norm = LayerNorm(args.hidden_size, eps=args.norm_eps)

    def forward(self, hidden_states):
        output = self.dense_1(hidden_states)
        output = gelu(output)
        output = self.dropout(output)
        output = self.dense_2(output)
        output = self.norm(hidden_states + output)

        return output


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.selfattlayer = SelfAttLayer(args)
        self.feedforward = FeedForward(args)
        self.norm = LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        output = self.selfattlayer(hidden_states, attention_mask)
        output = self.dropout(output)
        output = self.norm(output + hidden_states)
        output = self.feedforward(output)

        return output


class CrossAttLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(args.attention_dropout_prob)
        self.dense = nn.Linear(args.hidden_size * 4, args.hidden_size)  ### 87.9
        self.norm = LayerNorm(args.hidden_size, eps=args.norm_eps)

    def transpose_for_scores(self, x):
        # x : batch_size * max_seq * dim
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # new_x_shape: batch_size * max_seq * attention_heads * head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # return shape: batch_size * attention_heads * max_seq * head_size

    def forward(self, hidden_states_a, hidden_states_b, attention_mask_a, attention_mask_b):
        # hidden_states_a: batch_size * max_seq_a * embedding_dim
        # hidden_states_b: batch_size * max_seq_b * embedding_dim

        mixed_query_layer_a = hidden_states_a
        mixed_key_layer_a = hidden_states_a
        mixed_value_layer_a = hidden_states_a

        # mixed_query_layer_a = self.dense_1(hidden_states_a)
        # mixed_key_layer_a = self.dense_1(hidden_states_a)
        # mixed_value_layer_a = self.dense_1(hidden_states_a)

        query_layer_a = self.transpose_for_scores(mixed_query_layer_a)
        key_layer_a = self.transpose_for_scores(mixed_key_layer_a)
        value_layer_a = self.transpose_for_scores(mixed_value_layer_a)

        mixed_query_layer_b = hidden_states_b
        mixed_key_layer_b = hidden_states_b
        mixed_value_layer_b = hidden_states_b

        # mixed_query_layer_b = self.dense_1(hidden_states_b)
        # mixed_key_layer_b = self.dense_1(hidden_states_b)
        # mixed_value_layer_b = self.dense_1(hidden_states_b)

        query_layer_b = self.transpose_for_scores(mixed_query_layer_b)
        key_layer_b = self.transpose_for_scores(mixed_key_layer_b)
        value_layer_b = self.transpose_for_scores(mixed_value_layer_b)

        extended_attention_mask_a = attention_mask_a[:, None, None, :]
        extended_attention_mask_a = (1.0 - extended_attention_mask_a) * -10000.0
        attention_mask_a = extended_attention_mask_a

        extended_attention_mask_b = attention_mask_b[:, None, None, :]
        extended_attention_mask_b = (1.0 - extended_attention_mask_b) * -10000.0
        attention_mask_b = extended_attention_mask_b

        attention_scores_a2b = torch.matmul(query_layer_a, key_layer_b.transpose(-1, -2))
        # batch_size * attention_heads * max_seq_a * max_seq_b
        attention_scores_a2b = attention_scores_a2b / math.sqrt(self.attention_head_size)
        attention_scores_a2b = attention_scores_a2b + attention_mask_b
        attention_probs_a2b = nn.Softmax(dim=-1)(attention_scores_a2b)
        attention_probs_a2b = self.dropout(attention_probs_a2b)
        context_layer_a2b = torch.matmul(attention_probs_a2b, value_layer_b)
        context_layer_a2b = context_layer_a2b.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_a2b = context_layer_a2b.size()[:-2] + (self.all_head_size,)
        context_layer_a2b = context_layer_a2b.view(*new_context_layer_shape_a2b)

        attention_scores_b2a = torch.matmul(query_layer_b, key_layer_a.transpose(-1, -2))
        # batch_size * attention_heads * max_seq_b * max_seq_a
        attention_scores_b2a = attention_scores_b2a / math.sqrt(self.attention_head_size)
        attention_scores_b2a = attention_scores_b2a + attention_mask_a
        attention_probs_b2a = nn.Softmax(dim=-1)(attention_scores_b2a)
        attention_probs_b2a = self.dropout(attention_probs_b2a)
        context_layer_b2a = torch.matmul(attention_probs_b2a, value_layer_a)
        context_layer_b2a = context_layer_b2a.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_b2a = context_layer_b2a.size()[:-2] + (self.all_head_size,)
        context_layer_b2a = context_layer_b2a.view(*new_context_layer_shape_b2a)

        context_layer_a = torch.cat([hidden_states_a, context_layer_a2b,
                                     hidden_states_a - context_layer_a2b, hidden_states_a * context_layer_a2b],
                                    -1)

        context_layer_b = torch.cat([hidden_states_b, context_layer_b2a,
                                     hidden_states_b - context_layer_b2a, hidden_states_b * context_layer_b2a],
                                    -1)

        context_layer_a = self.dropout(context_layer_a)
        context_layer_b = self.dropout(context_layer_b)

        context_layer_a = self.dense(context_layer_a)
        context_layer_a = gelu(context_layer_a)

        context_layer_b = self.dense(context_layer_b)
        context_layer_b = gelu(context_layer_b)

        context_layer_a = self.dropout(context_layer_a)
        context_layer_b = self.dropout(context_layer_b)

        context_layer_a = self.norm(hidden_states_a + context_layer_a)
        context_layer_b = self.norm(hidden_states_b + context_layer_b)

        outputs = (context_layer_a, context_layer_b)

        return outputs


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(args) for i in range(args.num_encoder_layers)])
        self.crossattlayer = CrossAttLayer(args)

    def forward(self, hidden_states_a, hidden_states_b, attention_mask_a, attention_mask_b):
        out_a = hidden_states_a
        out_b = hidden_states_b
        for i, encoder in enumerate(self.encoders):
            out_a = encoder(out_a, attention_mask_a)
            out_b = encoder(out_b, attention_mask_b)
        out_a, out_b = self.crossattlayer(out_a, out_b, attention_mask_a, attention_mask_b)
        return out_a, out_b


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        num_filters = args.cnn_num_filters  # 200
        filter_sizes = args.cnn_filter_sizes  # (1, 2, 3)
        hidden_size = args.hidden_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, hidden_size)) for k in filter_sizes])

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def conv_and_pool(self, hidden_states, conv):
        hidden_states = F.relu(conv(hidden_states)).squeeze(3)
        hidden_states = F.max_pool1d(hidden_states, hidden_states.size(2)).squeeze(2)
        return hidden_states

    def forward(self, hidden_states):
        out = hidden_states.unsqueeze(1)  # b * 1 * s * e
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        return out


class Pooling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cnn = CNN(args)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states, mask):
        cnn_output = self.cnn(hidden_states)

        extended_mask = mask[:, :, None]
        extended_mask = (1.0 - extended_mask) * (-100000)
        mask = extended_mask
        max_output = hidden_states + mask
        max_output = max_output.max(dim=1)[0]

        out = torch.cat([cnn_output, max_output], dim=-1)
        out = self.dropout(out)

        return out


class Prediction(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dense_1 = nn.Linear(args.cnn_num_filters * len(args.cnn_filter_sizes) * 5 + args.hidden_size * 5, args.hidden_size * 2)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.dense_2 = nn.Linear(args.hidden_size * 2, len(args.labels))

    def forward(self, a, b):
        outputs = torch.cat([a, b, a - b, torch.abs(a - b), a * b], dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.dense_1(outputs)
        outputs = gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense_2(outputs)
        return outputs


class MatchModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbeddingLayer(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.num_hidden_layers)])
        self.pooling = Pooling(args)
        self.prediction = Prediction(args)
        self.encs = nn.ModuleList([Encoder(args) for _ in range(args.num_last_selfatt_layers)])
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids_a, input_ids_b, attention_mask_a, attention_mask_b, labels):
        hidden_states_a = self.embedding(input_ids_a)
        hidden_states_b = self.embedding(input_ids_b)
        for i, layer in enumerate(self.blocks):
            hidden_states_a, hidden_states_b = layer(hidden_states_a, hidden_states_b, attention_mask_a, attention_mask_b)

        outputs_a = hidden_states_a
        outputs_b = hidden_states_b
        for i, layer in enumerate(self.encs):
            outputs_a = layer(outputs_a, attention_mask_a)
            outputs_b = layer(outputs_b, attention_mask_b)

        outputs_a = self.pooling(outputs_a, attention_mask_a)
        outputs_b = self.pooling(outputs_b, attention_mask_b)

        outputs = self.prediction(outputs_a, outputs_b)
        loss = self.loss_fct(outputs, labels)
        outputs = (loss, outputs)

        return outputs

