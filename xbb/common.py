import os, json

def get_dsid(fpath):
    assert os.path.isdir(fpath), f'{fpath} is not a directory'
    return int(os.path.basename(fpath).split('.')[2])

def is_dijet(dsid, restricted=False):
    if restricted:
        return 361024 <= dsid <= 361027
    return 361020 <= dsid <= 361032

def is_ditop(dsid, restricted=False):
    return 301322 <= dsid <= 301335

def is_dihiggs(dsid, restricted=False):
    # extended high mass samples
    if 305776 <= dsid <= 305780:
        return True
    if restricted:
        return 301500 <= dsid <= 301507
    return 301488 <= dsid <= 301507

def get_denom_dict(denom_file):
    return {int(k): v for k, v in json.load(denom_file).items()}

SELECTORS = {
    'higgs': is_dihiggs,
    'dijet': is_dijet,
    'top': is_ditop,
}
