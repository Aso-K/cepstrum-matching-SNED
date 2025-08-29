# cepstrum-matching-SNED

This repository provides the Python scripts used for cepstrum-based analysis of 
scanning nanobeam electron diffraction (SNED) data.  
It accompanies the manuscript currently under peer review.

All scripts are placed in the `code/` folder:
- `1simulatedCBED2cepstrum.py` : Generate simulated cepstrum from diffraction patterns
- `2experimentalCBED2cepstrum.py` : Convert experimental SNED data to cepstrum
- `3cepstrum_calibration.py` : Calibration of experimental cepstrum
- `4cepstrum-matching.py` : Matching of experiment and simulation
- `5show_ewpc_crosscorr_interp1d_20250825.py` : Visualization of cross-correlation results

Input data should be placed under `data/` (excluded from version control).
