from text_to_image import *
from load_coco import resnet18_features

coco = Coco(coco_data, resnet18_features)
model = ImageEmbedder()

#Get a training set
valid_image_ids = coco.img_ids
img_to_caps_map = coco.img_id_to_caption_id
train_triplets, val_triplets = generate_triplets(valid_image_ids, img_to_caps_map)


epoch_size = 100
for epoch_cnt in range(epoch_size):
    print(epoch_cnt)
    train_model(train_triplets, model, coco)

save_weights(model, "image_embedder_weights.pkl")