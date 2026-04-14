import hashlib
from bl_nodes.comfy.cli_args import args

class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3
    PREVIEW_IMAGE_WITH_METADATA = 4

def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs[args.default_hashing_function]