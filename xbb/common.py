import os, json

def get_dsid(fpath):
    assert os.path.isdir(fpath), f'{fpath} is not a directory'
    return int(os.path.basename(fpath).split('.')[2])

def is_dijet(dsid, restricted=False):
    if restricted:
        return 361024 <= dsid <= 361028
    return 361020 <= dsid <= 361032

def is_ditop(dsid):
    return 301322 <= dsid <= 301335

def is_dihiggs(dsid, restricted=False):
    if restricted:
        return 301500 <= dsid <= 301505
    return 301488 <= dsid <= 301507

def get_denom_dict(denom_file):
    return {int(k): v for k, v in json.load(denom_file).items()}

