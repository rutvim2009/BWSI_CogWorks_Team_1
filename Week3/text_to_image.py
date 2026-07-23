import gensim
import pickle
import re, string
from collections import defaultdict
import numpy as np
import mygrad as mg
from load_coco import coco_data, glove

# ---------- Batch-Embeds and Image Library (Part 6) ---------


#------------------------------------------------------------------------------------------
#Aahana (Task 1)
class Coco:
    def init(self, coco_data, descriptors=None):
        """
        builds and organizes the dataset. stores all the mappings.
        """

        #maps all imgs to urls
        self.img_id_to_url = {} 

        for img in coco_data["images"]: 
            img_id = img["id"]
            url = img["coco_url"]
            self.img_id_to_url[img_id] = url

        #maps all img ids to descriptors
        self.img_id_to_descriptor = {}

        if descriptors is not None:
            #flattens the descriptors
            for img_id in descriptors:
                vector = descriptors[img_id]
                vector = np.array(vector)
                vector = vector.reshape(-1)
                self.img_id_to_descriptor[img_id] = vector

            #removes all imgs with no descriptors
            filtered_urls = {}

            for img_id in self.img_id_to_url:
                if img_id in self.img_id_to_descriptor:
                    url = self.img_id_to_url[img_id]
                    filtered_urls[img_id] = url
                else:
                    continue #skips all imgs with no descriptors
            self.img_id_to_url = filtered_urls
        #dictionaries for captions
        self.caption_id_to_caption = {}
        self.caption_id_to_img_id = {}
        self.img_id_to_caption_id = {}

        #goes thru all the captions
        #goes thru all the captions and assigns them vals
        for i in coco_data["annotations"]:
            img_id = i["image_id"]
            caption_id = i["id"]
            caption = i["caption"]

            #ignore filtered imgs
            if img_id not in self.img_id_to_url:
                continue

            #maps captions and imgs
            self.caption_id_to_caption[caption_id] = caption
            self.caption_id_to_img_id[caption_id] = img_id

            #attaches the captions to the img
            if img_id in self.img_id_to_caption_id:
                self.img_id_to_caption_id[img_id].append(caption_id)
            else:
                self.img_id_to_caption_id[img_id] = [caption_id]

        #stores the lists of img + caption ids
        self.img_ids = sorted(self.img_id_to_caption_id)
        self.caption_ids = sorted(self.caption_id_to_caption)

    #returns all the captions for the img
    def captions_for_img(self, img_id):
        caption_ids = self.img_id_to_caption_id[img_id] #each caption id associated with the img
        captions = []
        #caption ids -> caption strings
        for caption_id in caption_ids:
            captions.append(self.caption_id_to_caption[caption_id])
        return captions
    #returns img url
    def url(self, img_id):
        return self.img_id_to_url[img_id]

    #returns img descriptor
    def descriptor(self, img_id):
        return self.img_id_to_descriptor[img_id]

def save(coco, path): #saves the file
    with open(path, "wb") as f:
        pickle.dump(coco, f)

def load(path): #loads the file
    with open(path, "rb") as f:
        return pickle.load(f)

    
     
        
        
        
#------------------------------------------------------------------------------------------
#Aanya (Task 3)

# Make a function that can embed any caption / query text (using GloVe-200 embeddings weighted by IDFs of words across captions)
# An individual word not in the GloVe or IDF vocabulary should yield an embedding vector of just zeros.


def embed_text(tokens, glove, idf):
    embedding = np.zeros(glove.vector_size, dtype = np.float32)
    for word in tokens:
        if word in glove and word in idf:
            embedding += idf[word] * glove[word]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding
    



#------------------------------------------------------------------------------------------
#Spencer (Task 2)
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(corpus):
    return punc_regex.sub('', corpus)
def caption_processor(caption):
    """
    Args:
        caption: Type = str
    
    Returns:
        The same string but as a proccessed list (No puncuation, no caps, tokenized based on white space)
    """
    return strip_punc(caption.lower()).split()
#ex: coco_data["annotations"][0] = {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
ncaption = len(coco_data["annotations"])
vocab = defaultdict(int)#all words across all acptions in COCO
for annotation in coco_data["annotations"]:
    words = set(caption_processor(annotation["caption"]))
    for word in words:
        vocab[word] += 1
vocab = sorted(vocab.items(), key = lambda x: x[1], reverse= True)
words = list(vocab.keys())
counts = np.array(list(vocab.values()))
idf_values = np.log10(ncaption/counts)
IDF = dict(zip(words, idf_values))

#____
_

#----------------------------------------------------------------------------------------
#Mihika 
'''
Model training and funcitonality for saving/loading trained weights
'''

def train_model(training_set): 
    captions = []
    true_images = []
    confusor_images = []
   for caption_ID, image_ID, confusor_image_ID in training_set:
       capt
       #compare similarities
       #compute loss and accuracy
       #take optimization step (mygrad)
    
    .backward()
    


    






# ---------- Batch-Embeds and Image Library (Part 6 : Jesse) ---------

def embed_images(model, features):
    image_ids = list(features.keys()) # Extract Keys
    descriptors = np.vstack([features[img_ids] for img_ids in image_ids]).astype(np.float32) # Stacks features into 2D array

    # Feeds through trained model
    embeddings = model(descriptors)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) # Image magnitudes
    norms[norms == 0] = 1
    embeddings = embeddings / norms # Normalization

    return image_ids, embeddings
    

class ImageDatabase:
    def __init__(self, model, features):
        self.image_ids, self.embeddings = embed_images(model, features)
        self._id_to_row = {img_id: i for i, img_id in enumerate(self.image_ids)} # Image ids + index

    # Get embeddings
    def get_embedding(self, image_id):
        return self.embeddings[self._id_to_row[image_id]]

    # Get number of images in index
    def __len__(self):
        return len(self.image_ids)

    # Save to disk
    def save(self, path):
        np.savez(path, image=np.array(self.image_ids), embeddings=self.embeddings)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls.__new__(cls) # Skips initializing function
        obj.image_ids = data["image_ids"].tolist()
        obj.embeddings = data["embeddings"]
        obj._idd_to_row = {img_id: i for i, img_id in enumerate}







#7 Shriyans
def query_database(caption_embedding, image_database, k=5):
    ids = list(image_database.keys())
    embeddings = np.array([image_database[i] for i in ids])
    scores = embeddings @ caption_embedding
    ranked = sorted(zip(ids, scores), key=lambda pair: - pair[1])
    return [img_id for img_d, score in ranked[:k]]
    