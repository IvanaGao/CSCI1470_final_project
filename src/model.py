import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class CrossATT(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8):

        super(CrossATT, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drug2adr_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True,)
        self.adr2drug_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True,)

        # Layer normalization and feed-forward networks are applied to stabilize and refine feature representations
        self.norm1_drug = nn.LayerNorm(embed_dim)
        self.norm1_adr = nn.LayerNorm(embed_dim)

        self.ffn_drug = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_adr = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm2_drug = nn.LayerNorm(embed_dim)
        self.norm2_adr = nn.LayerNorm(embed_dim)

    def forward(self, drug_vec, adr_seq):

        drug_query = drug_vec.unsqueeze(1)  # -> [B, 1, D]
        # Cross-attention computation
        attn_output_drug, _ = self.drug2adr_att(query=drug_query, key=adr_seq, value=adr_seq)  # -> [B, 1, D]
        # Residual connection and layer normalization
        updated_drug = self.norm1_drug(drug_vec + attn_output_drug.squeeze(1))
        # Pass through a feed-forward network
        ffn_output_drug = self.ffn_drug(updated_drug)
        # A second residual connection and layer normalization
        final_drug_vec = self.norm2_drug(updated_drug + ffn_output_drug)

        # --- Pipeline B: ADR sequence absorbs drug information ---
        # adr_seq is used as the Query
        # drug embeddings are used as Key and Value, expanded to match the sequence length
        drug_kv = drug_vec.unsqueeze(1).expand(-1, adr_seq.shape[1], -1)  # -> [B, N, D]
        # Cross-attention computation
        attn_output_adr, _ = self.adr2drug_att(query=adr_seq, key=drug_kv, value=drug_kv)  # -> [B, N, D]
        # Residual connection and layer normalization
        updated_adr = self.norm1_adr(adr_seq + attn_output_adr)
        # Pass through a feed-forward network
        ffn_output_adr = self.ffn_adr(updated_adr)
        # A second residual connection and layer normalization
        final_adr_seq = self.norm2_adr(updated_adr + ffn_output_adr)

        return final_drug_vec, final_adr_seq


class ADRsModel(nn.Module):

    def __init__(self, args, embedding_dim: int=128, class_weights=None, device: str='cuda'):
        super(ADRsModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.text_encoder = BertModel.from_pretrained(args.bert_path)
        self.text_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        hidden_size = self.text_encoder.config.hidden_size
        self.unii_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.reac_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.bid_att = CrossATT(embed_dim=hidden_size, num_heads=8)

        self.unii_batch_norm = nn.BatchNorm1d(self.text_encoder.config.hidden_size)
        self.reac_batch_norm = nn.BatchNorm1d(self.text_encoder.config.hidden_size)

        self.is_single_label = args.only_single_reaction
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        if self.is_single_label:
            if args.use_class_weights:
                assert self.class_weights is not None
                print('using focus loss weights')
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights) # invalid
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.use_unii_desc = args.use_unii_desc
        # demographic information
        self.use_gender = args.use_gender
        self.use_age = args.use_age
        self.use_weight = args.use_weight

    def forward(self, batch_input):

        batch_unii_encodings = []
        batch_reac_encodings = []
        for sample in batch_input:
            # drug encoding
            unii_encodings = []
            for drug in sample['patient_drug']:
                encoded_input = self.text_tokenizer(drug['unii'], return_tensors='pt', padding=True).to(self.device)
                unii_encoding = self.text_encoder(**encoded_input).pooler_output.mean(dim=0)
                unii_encodings.append(unii_encoding)
                if self.use_unii_desc and len(drug['unii_desc']) > 0:
                    encoded_input = self.text_tokenizer(
                        drug['unii'], return_tensors='pt', padding=True, truncation=True, max_length=512
                    ).to(self.device)
                    unii_desc_encoding = self.text_encoder(**encoded_input).pooler_output.mean(dim=0)
                    unii_encodings.append(unii_desc_encoding)

            # Integrate gender information into the UNII representation (may require further refinement)
            if self.use_gender:
                encoded_input = self.text_tokenizer(sample['patient_sex'], return_tensors='pt', padding=True).to(self.device)
                gender_encoding = self.text_encoder(**encoded_input).pooler_output.mean(dim=0)
                unii_encodings.append(gender_encoding)
            # Incorporate age information
            if self.use_age:
                encoded_input = self.text_tokenizer(sample['patient_age'], return_tensors='pt', padding=True).to(self.device)
                gender_encoding = self.text_encoder(**encoded_input).pooler_output.mean(dim=0)
                unii_encodings.append(gender_encoding)
            # Incorporate gender information
            if self.use_gender:
                encoded_input = self.text_tokenizer(sample['patient_weight'], return_tensors='pt', padding=True).to(self.device)
                gender_encoding = self.text_encoder(**encoded_input).pooler_output.mean(dim=0)
                unii_encodings.append(gender_encoding)

            unii_encodings = torch.stack(unii_encodings, dim=0)
            batch_unii_encodings.append(unii_encodings.mean(dim=0))
            # ADR encoding
            encoded_input = self.text_tokenizer(sample['patient_reaction'], return_tensors='pt', padding=True).to(self.device)
            reac_encoding = self.text_encoder(**encoded_input).pooler_output
            batch_reac_encodings.append(reac_encoding)

        batch_unii_encodings = torch.stack(batch_unii_encodings, dim=0)
        batch_reac_encodings = torch.stack(batch_reac_encodings, dim=0)

        batch_unii_encodings, batch_reac_encodings = self.bid_att(
            drug_vec=batch_unii_encodings, adr_seq=batch_reac_encodings
        )

        batch_unii_encodings = self.unii_projection(batch_unii_encodings)
        batch_reac_encodings = self.reac_projection(batch_reac_encodings)

        # invalid
        # batch_unii_encodings = self.unii_batch_norm(batch_unii_encodings)
        # B, N, D = batch_reac_encodings.shape
        # batch_reac_encodings = self.reac_batch_norm(batch_reac_encodings.view(B * N, D)).view(B, N, D)
        # batch_unii_encodings = F.normalize(batch_unii_encodings, p=2, dim=1)
        # batch_reac_encodings = F.normalize(batch_reac_encodings, p=2, dim=2)

        similarity_scores = torch.bmm(batch_reac_encodings, batch_unii_encodings.unsqueeze(2)).squeeze(2)

        if self.training:
            if self.is_single_label:
                target_matrix = torch.tensor(
                    [sample['patient_reaction_pos_ids'][0] for sample in batch_input], device=self.device
                )
            else:
                # Construct the target matrix
                target_matrix = torch.zeros_like(similarity_scores)
                for i, sample in enumerate(batch_input):
                    target_matrix[i, sample['patient_reaction_pos_ids']] = 1
            # compute loss
            loss = self.criterion(similarity_scores, target_matrix)

            return loss
        else:

            # # return similarity_scores
            # return torch.softmax(similarity_scores, dim=1)
            # # return torch.sigmoid(similarity_scores)

            if self.is_single_label:
                # return similarity_scores
                return torch.softmax(similarity_scores, dim=1)
            else:
                return torch.sigmoid(similarity_scores)




if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('/mnt/home/xxx/models/bert/bert_uncased_L-2_H-128_A-2')
    model = BertModel.from_pretrained('/mnt/home/xxx/models/bert/bert_uncased_L-2_H-128_A-2')
    text = "D S D F J A H S D F S A"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    print()