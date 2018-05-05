import os, json

def get_dsid(fpath):
    assert os.path.isdir(fpath), f'{fpath} is not a directory'
    return int(os.path.basename(fpath).split('.')[2])

def is_dijet(dsid, restricted=True):
    if restricted and 361020 <= dsid <= 361023:
        return False
    return 361020 <= dsid <= 361032

def is_ditop(dsid):
    return 301322 <= dsid <= 301335

def is_dihiggs(dsid):
    return 301488 <= dsid <= 301507

def get_denom_dict(denom_file):
    return {int(k): v for k, v in json.load(denom_file).items()}

