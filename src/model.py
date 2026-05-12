import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class MedCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        #temperature already a learned param

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        #image-to-text similarity matrix
        return outputs.logits_per_image, outputs.logits_per_text

    def get_embeddings(self, pixel_values=None, input_ids=None, attention_mask=None):
        img_embeds  = None
        text_embeds = None
        if pixel_values is not None:
            img_embeds = self.clip.get_image_features(pixel_values=pixel_values)
            if not isinstance(img_embeds, torch.Tensor):
                img_embeds = img_embeds.pooler_output
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        if input_ids is not None:
            text_embeds = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            if not isinstance(text_embeds, torch.Tensor):
                text_embeds = text_embeds.pooler_output
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return img_embeds, text_embeds