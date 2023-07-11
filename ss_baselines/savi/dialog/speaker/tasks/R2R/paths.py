# BSD 2-Clause License

# Copyright (c) 2018, Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach,
# Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko,
# Dan Klein, Trevor Darrell
# All rights reserved.

convolutional_feature_store_paths = {
    'imagenet': './ss_baselines/savi/dialog/speaker/img_features/imagenet_convolutional',
    'places365': './ss_baselines/savi/dialog/speaker/img_features/places365_convolutional',
}

mean_pooled_feature_store_paths = {
    'imagenet': './ss_baselines/savi/dialog/speaker/img_features/ResNet-152-imagenet.tsv',
    'places365': './ss_baselines/savi/dialog/speaker/img_features/ResNet-152-places365.tsv',
}

bottom_up_feature_store_path = "./ss_baselines/savi/dialog/speaker/img_features/bottom_up_10_100"
bottom_up_feature_cache_path = "./ss_baselines/savi/dialog/speaker/img_features/bottom_up_10_100.pkl"
bottom_up_feature_cache_dir = "./ss_baselines/savi/dialog/speaker/img_features/bottom_up_10_100_cache"

bottom_up_attribute_path = "./ss_baselines/savi/dialog/speaker/data/visual_genome/attributes_vocab.txt"
bottom_up_object_path = "./ss_baselines/savi/dialog/speaker/data/visual_genome/objects_vocab.txt"
