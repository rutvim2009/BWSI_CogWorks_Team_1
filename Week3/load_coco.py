from cogworks_data.language import get_data_path
from gensim.models import KeyedVectors
from pathlib import Path
import json

# load COCO metadata
filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)
print("Download complete!")


print("Glove Started!")
filename = "glove.6B.200d.txt.w2v"

glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)
print("Glove Finished!")