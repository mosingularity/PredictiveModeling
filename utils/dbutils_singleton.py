# utils/dbutils_singleton.py
_dbutils = None

def set_dbutils(instance):
    global _dbutils
    _dbutils = instance

def get_dbutils():
    return _dbutils
