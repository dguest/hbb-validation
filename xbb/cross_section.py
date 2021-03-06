class CrossSections:
    def __init__(self, infile, denom_dict, lumi_fb=1):
        self.datasets = {}
        self.lumi_fb = lumi_fb
        for rawline in infile:
            self._add_line(rawline, denom_dict)

    def _add_line(self, rawline, denom_dict):
        line = rawline.strip()
        if not line or line.startswith('#'):
            return
        shortname, dsid_s, xsec_nb_s, filteff_s = line.split()
        dsid = int(dsid_s)
        self.datasets[dsid] = {
            'xsec_fb': float(xsec_nb_s) * 1e6,
            'filteff': float(filteff_s),
            'denominator': denom_dict.get(dsid, float('nan')),
            'shortname': shortname}

    def get_weight(self, dsid):
        rec = self.datasets[dsid]
        return self.lumi_fb * rec['xsec_fb'] * rec['filteff'] / rec['denominator']

    def is_signal(self, dsid):
        rec = self.datasets[dsid]
        if rec['shortname'].startswith('JZ'):
            return False
        return True
