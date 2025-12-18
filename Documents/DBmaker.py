from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import numpy as np
import Utils.Constants as CNS
import dotenv
import json


dotenv.load_dotenv()

model_name = "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}  

embedder = HuggingFaceEmbeddings(  
model_name=model_name,  
model_kwargs=model_kwargs,  
encode_kwargs=encode_kwargs  
)

DB_FILE_NAME = "db_entername"
DB_SAVE_PATH = CNS.VECTORS_DIR / DB_FILE_NAME

SOURCE_PATH = CNS.ROOT_DIR / "ConvertToVectorDB"
TEXTSAVE_PATH = SOURCE_PATH / "Texts"


# read the text file
textfile_content = ""
with open(TEXTSAVE_PATH / CNS.TEMP_TEXT_NAME, encoding = "utf-8") as coa:
    for line in coa.readlines():
        textfile_content += line


# conver to documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splits = splitter.split_text(textfile_content)
print(len(text_splits))

txt_doc = [Document(page_content=text) for text in text_splits]
print(len(txt_doc))


# Create and save vector db
db = Chroma.from_documents(documents=txt_doc, embedding=embedder, persist_directory=str(DB_SAVE_PATH) )
db.persist()


# calculate and save centroid of db
#embeddings = embedder.embed_documents([doc.page_content for doc in txt_doc])
#centroid = np.mean(embeddings, axis=0)
db_embeddings = db.get(include=["embeddings"])["embeddings"]
db_centroid = np.mean(db_embeddings, axis=0)

centroid_save = CNS.VECTOR_CENTROIDS_PATH / DB_FILE_NAME
np.save(centroid_save, db_centroid)


# Updating centroid - dp mappings
cent_json = CNS.VECTORS_DIR / "mappings.json"

with open(cent_json, "r") as c_json:
    jsondata : dict = json.load(c_json)

jsondata[str(DB_FILE_NAME)] = str(DB_SAVE_PATH)
with open(cent_json, "w") as c_json:
    json.dump(jsondata, c_json, indent=4)
