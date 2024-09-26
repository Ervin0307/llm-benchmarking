import hashlib

#get hash of a string or object
def get_hash(obj):
    return hashlib.sha256(str(obj).encode()).hexdigest()