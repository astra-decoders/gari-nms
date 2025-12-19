# GARI-NMS decoder

Note: This codebase is to reproduce the results from a paper; a cleaned-up user-friendly version is in preparation.

Directory structure:
- data/
  - final/  ← contains the data used for plots in the paper
  - circuits/ ← contains the stim circuits used
- my_decoders/
  - [hbp_decoder_v2.c](my_decoders/hbp_decoder_v2.c) ← actual decoder implementation in c
- stim_batched_data_v2.py ← main file

Setup:
- python 3.11
  - `pip install ldpc stim sinter`
  - `cd my_decoders && make clean && make && cd ..`

Run:
- python stim_batched_data_v2.py