# Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
import clip
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
print(text)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(text_features.size())
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
