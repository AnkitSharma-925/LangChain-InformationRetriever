import fitz
import os
from Utils import Constants as CNS

# to convert pdf or text into vector,
# put the file (or files) in ConvertToVectorDB folder

print("im working")

SOURCE_PATH = CNS.ROOT_DIR / "ConvertToVectorDB"
TEXTSAVE_PATH = SOURCE_PATH / "Texts"

files = [f for f in os.listdir(SOURCE_PATH) if os.path.isfile(os.path.join(SOURCE_PATH, f))]

text = ""
for file in files:
    if(file[-3:] == "pdf"):
        filepath = SOURCE_PATH / file
        doc = fitz.open(filepath)
        for page in doc:
            text = text + "\n" + str(page.get_text())


with open(TEXTSAVE_PATH / CNS.TEMP_TEXT_NAME, "w", encoding="utf-8") as text_savefile:
    text_savefile.write(text)