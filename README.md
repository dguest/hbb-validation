Hbb Training Dataset Validation Scripts
=======================================

This repository contains several scripts to verify that the training
datsets for H->bb tagging are doing what we think they should be
doing. Since several stages are required to properly assign weights to
the samples, it's important that a few scripts be run first.

 - To calculate the right normalization for the dijet samples, you
   have to run `get_weight_denominator.py`. Note that this _only_
   needs to be run over the dijet samples. This will produce a
   "denominator" json file that the other scripts require.

 - To calculate the right normalization for the higgs samples, you
   need to run `make_jet_pt_hist.py`. This will produce a few files in
   `pt-hists` which are used to scale the higgs jet pt spectrum to
   match the dijet spectrum.

 - After these steps are run you can run `make_roc_curves.py` which
   will plot a few roc curves for the baseline H->bb tagging
   discriminants.
