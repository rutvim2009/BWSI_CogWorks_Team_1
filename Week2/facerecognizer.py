# imports

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import numpy as np
import pickle 
from pathlib import Path
import random    
from facenet_models import FacenetModel
    
    
import networkx as nx # for plot_graph
import matplotlib.cm as cm # for plot_graph
import matplotlib.pyplot as plt # for plot_graph
from facenet_models import FacenetModel
# ----------------------------------------------------------------------------------------------------------------

#1 Create a Profile class with functionality to store face descriptors associated with a named individual.
class Profile:
    def _init_(self, name):
        self.name=name 
        self.descriptors=[]
    def add_descriptor(self,descriptor):
        self.descriptors.append(descriptor)

# ----------------------------------------------------------------------------------------------------------------
    


#2
def new_database():    
    return {}
    
def save(self, path): #saves the file
    with open(path, mode="wb") as f:
        pickle.dump(self, f)

def load(path): #loads the file
    with open(path, "rb") as f:
        return pickle.load(f)

def add_profile(db, name, descriptors=None):
    if name in db:    
        raise KeyError(f"{name} already in database")
    prof = Profile(name)    
    if descriptors is not None:
        for d in np.atleast_2d(descriptors):
            prof.add_descriptor(d)
    db[name] = prof    
    
def delete_profile(db, name):        
    if name not in db:
        raise KeyError(f"{name} is not in database") 
    del db[name]    
    
   
def add_images(db, name, image, model):    
    result = model.detect(image)
    boxes = result[0] #the boxes around each person's face
    probs = result[1] #the probability [accuracy rate] of the face detection
    features = result[2] #essential features of the face [e.g. nose, ears]
    descriptors = model.compute_descriptors(image, boxes) #one descriptor per box
    descriptor = descriptors[0]    
    if name not in db:    
        db[name] = Profile(name)
    profile = db[name]
    profile.add_descriptor(descriptor)
    
# ----------------------------------------------------------------------------------------------------------------

#3 

def compute_cos_dist(desc_M, desc_N): #task 3
    norm_M = np.linalg.norm(desc_M, axis=1, keepdims=True) # calc
    norm_N = np.linalg.norm(desc_N, axis=1, keepdims=True)
    norm_M = np.where(norm_M == 0, 1.0, norm_M)
    norm_N = np.where(norm_N == 0, 1.0, norm_N)

    normalized_M = desc_M / norm_M
    normalized_N = desc_N / normalized_N

    cos_simil = np.dot(normalized_M, normalized_N.T)
    cos_dist = 1.0 - cos_simil

    return np.clip(cos_dist, 0.0, 2.0)
    



# ----------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------

#5 
def cosine_threshold(database, sample_size=int): 
   '''
   plots histogram of small sample of overall people and how their distances are from themselves and other people to determine a good cutoff threshold
   '''

   people = random.sample(database.values(), sample_size)
   same_person_distances = []
   diff_person_distances = []

   for person in people:
       for d in range(len(person.descriptors) -1): #for each person, compare against self and add to same_person_distances
           same_person_distances.append(compute_cos_dist(person.descriptors[d], person.descriptors[d+1]))
           for other_person in people: #can also compare against diff person avg (probably better ngl)
                if other_person is person:
                    continue
                diff_person_distances.append(compute_cos_dist(person.descriptors[d], other_person.descriptors[d+1]))   
        #plot and find good cutoff
       plt.hist(same_person_distances, label="same person")
       plt.hist(diff_person_distances, label="different person")
       plt.xlabel("Cosine distance")
       plt.ylabel("density")
       plt.show
   cosine_threshold = 0.01 #temp placeholder val for other ppl to use while testing
   return cosine_threshold
    
# ----------------------------------------------------------------------------------------------------------------



#6
def identify_face(new_descriptor, profile_database, threshold):
    """
    See if a new descriptor has a match in the database
    """
    # Makes sure the new descriptors are a 2D array
    query = np.atleast_2d(new_descriptor)

    # Defaults for every new descriptor
    best_match = "Unknown"
    lowest_dist = float('inf')
    
    for profile in profile_database:
        # Detects if the profile is empty (replace descriptors with whatever element) INCOMPLETE
        if len(profile.descriptors) == 0:
            continue
        
        database = np.array(profile.descriptors)

        # Distance between the new descriptor and the database image
        distances = compute_cos_dist(query, database)

        min_profile_dist = np.min(distances)
        
        # Finds the lowest distance out of all the images (Best Match)
        if min_profile_dist < lowest_dist:
            lowest_dist = min_profile_dist
            best_match = profile.name
    
    # If the distance is beyond the threshold it
    if lowest_dist > threshold:
        return "Unknown", lowest_dist
    
    return best_match, lowest_dist



# ----------------------------------------------------------------------------------------------------------------



#7 

def display_matches(image,boxes,names) : #creates a blank plot and display photo on it
    fig, ax = plt.subplots()
    ax.imshow(image)

    #go through each face detected and its matched name toget
    for (x1,y1,x2,y2), name in zip(boxes, names) :
        #traces the box's 4 corners, ending back where it started
        xs=[x1,x2,x2,x1,x1]
        ys = [y1,y1,y2,y2,y1]
        ax.plot(xs,ys,color="red") #draws the box outline           

        #uses the matched name or "Unknown" if there wasn't one          
        label = name if name else "Unknown"
        ax.text(x1,y2+15,label,color="red") #writes the name below the box using coordinates

    return fig,ax
    return fig,ax


