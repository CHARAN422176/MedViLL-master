import os
import torch
from transformers import AutoConfig, BertConfig, BertPreTrainedModel
from cxrbert_origin import CXRBERT

class CXRBertForRetrieval(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        if args.weight_load:
            config = AutoConfig.from_pretrained(args.load_pretrained_model)
            # Load raw checkpoint
            raw_state_dict = torch.load(
                os.path.join(args.load_pretrained_model, 'pytorch_model.bin'),
                map_location=torch.device(args.device)
            )
            
            # Rename keys to match model expected keys
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('cls.predictions'):
                    new_k = k.replace('cls.predictions', 'mlm.predictions')
                    state_dict[new_k] = v
                elif k == 'enc.txt_embeddings.position_ids':
                    # Skip this unexpected key
                    continue
                else:
                    state_dict[k] = v
            
            # Initialize model from config, then load state dict manually
            cxrbert = CXRBERT(config, args)
            cxrbert.load_state_dict(state_dict, strict=False)  # strict=False to ignore missing keys if any

        else:
            config = BertConfig.from_pretrained('bert-base-uncased')
            cxrbert = CXRBERT(config, args)

        self.enc = cxrbert.enc
        self.itm = cxrbert.itm

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        _, cls, _ = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)
        result = self.itm(cls)
        return result
