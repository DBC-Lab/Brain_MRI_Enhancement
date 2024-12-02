These images are provided for model testing and include T2-weighted (fetal) and T1-weighted (from 0 months to adult) images, organized in the BIDS structure.

## Age information ##

Age information is required for the BME-X model and can be retrieved from the BIDS scans.tsv file (preferred), sessions.tsv file, or the corresponding *.json file.

For example, the BME-X model could read _age information_ for sub-0001/ses-V01/anat/sub-0001_ses-V01_T1w.nii.gz from sub-0001/ses-V01/anat/sub-0001_ses-V01_scans.tsv:

