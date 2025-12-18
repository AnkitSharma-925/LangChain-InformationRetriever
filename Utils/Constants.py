import pathlib


# DIRECTORIES
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
VECTORS_DIR = ROOT_DIR / "VectorDB"
VECTOR_CENTROIDS_PATH = VECTORS_DIR / "Centroids"
TEMP_TEXT_NAME = "temptextfile.txt"

print((ROOT_DIR))

DB_DIR_COA = "db_COA"
DB_DIR_CD = "db_CD"
DB_DIR_CN = "db_CN"
DB_DIR_TOC = "db_TOC"
DB_DIR_OS = "db_OS"
DB_DIR_DBMS = "db_DBMS"
