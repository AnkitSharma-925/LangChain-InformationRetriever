from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import json
import numpy as np
import Utils.Constants as CNS



model_name = "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}
  
embedder = HuggingFaceEmbeddings(  
    model_name=model_name,  
    model_kwargs=model_kwargs,  
    encode_kwargs=encode_kwargs  
)



#---------- MAIN DB STUFF -----------

# db_COA = Chroma(persist_directory = CNS.DB_DIR_COA, embedding_function=embedder)
# db_CD = Chroma(persist_directory = CNS.DB_DIR_CD, embedding_function=embedder)

# DB_DICT = {
#     "coa" : db_COA,
#     "cd" : db_CD
# }
# CENTROID_DICT = {
#     "coa" : np.load(os.path.join(CNS.DB_CENTROIDS, CNS.DB_DIR_COA+".npy")),
#     "cd" : np.load(os.path.join(CNS.DB_CENTROIDS, CNS.DB_DIR_CD+".npy"))
# }




#---------- PUBLIC DEFS -----------

def CosineSimilarity(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / ( np.linalg.norm(a) * np.linalg.norm(b) )


def GetMatchingDB(input_prompt) -> Chroma:
    vector_prompt = embedder.embed_query(input_prompt)
    nearest_centroid : str = ""
    similarity_score = 0
    vector_db : Chroma = Chroma()

    # load mapping json
    with open (CNS.VECTORS_DIR / "mappings.json", 'r') as maps:
        mappings : dict = json.load(maps)
    
    # retrieve all centroids
    centroids = [c for c in os.listdir(CNS.VECTOR_CENTROIDS_PATH) if os.path.isfile(os.path.join(CNS.VECTOR_CENTROIDS_PATH, c))]

    # find nearest centroid
    for cent in centroids:
        cent_name = cent.split('.')[0]
        centroid = np.load(CNS.VECTOR_CENTROIDS_PATH / cent)

        score = CosineSimilarity(vector_prompt, centroid)
        print("score: ", score, "for: ", cent_name )

        if (score > similarity_score):
            similarity_score = score
            nearest_centroid = cent_name

    # return vector_db w.r.t nearest centroid
    if (nearest_centroid != ""):
        db_path : str = mappings[nearest_centroid]
        vector_db = Chroma(persist_directory=db_path, embedding_function=embedder)

    return vector_db




    # for key, value in DB_DICT.items():
    #     centroid = np.array(CENTROID_DICT[key])
    #     new_score = CosineSimilarity(vector_prompt, centroid)
    #     if (new_score > similarity_score) :
    #         similarity_score = new_score
    #         nearest_centroid = key
    
    # return DB_DICT[nearest_centroid]