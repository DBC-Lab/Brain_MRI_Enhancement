These images are provided for model testing and include T2-weighted (fetal) and T1-weighted (from 0 months to adult) images, organized in the BIDS structure.

## Age information ##

Age information is required for the BME-X model and can be retrieved from the BIDS scans.tsv file (preferred), sessions.tsv file, or the corresponding *.json file.

For example, the BME-X model could read _age information_ for sub-0001/ses-V01/anat/sub-0001_ses-V01_T1w.nii.gz from sub-0001/ses-V01/anat/sub-0001_ses-V01_scans.tsv:

| Filename                     | Subject   | Session   | Modality | Age     |
|------------------------------|-----------|-----------|----------|---------|
| sub-0001_ses-V01_T1w.nii.gz  | sub-0001  | ses-V01   | T1w      | 6 years |

or from sub-0001/sub-0001_sessions.tsv:
| Subject   | Session   | Age     |
|-----------|-----------|---------|
| sub-0001  | ses-V01   | 6 years |
| sub-0001  | ses-V02   | 6 years |

or from sub-0001/ses-V01/anat/sub-0001_ses-V01_T1w.json:

    {
        "Manufacturer": "Siemens",
        "MagneticFieldStrength": 3.0,
        "ReceiverCoil": "32-channel",
        "MRAcquisitionType": "3D",
        "IsotropicVoxelSize": 0.8,
        "Age": "6 years"
    }

## Supported age types ##
The supported age types are: 'years', 'months', and 'gestational_weeks'. 

| Filename                     | Subject   | Session   | Modality | Age     |
|------------------------------|-----------|-----------|----------|---------|
| sub-0001_ses-V01_T1w.nii.gz  | sub-0001  | ses-V01   | T1w      | 6 years |
| sub-0001_ses-V02_T1w.nii.gz  | sub-0001  | ses-V02   | T1w      | 6 years |
| sub-0002_ses-V01_T1w.nii.gz  | sub-0002  | ses-V01   | T1w      | 23 months |
| sub-0003_ses-V01_T1w.nii.gz  | sub-0003  | ses-V01   | T1w      | 19 months |
| sub-0004_ses-V01_T1w.nii.gz  | sub-0004  | ses-V01   | T1w      | 11 months |
| sub-0005_ses-V01_T1w.nii.gz  | sub-0005  | ses-V01   | T1w      | 9 months |
| sub-0006_ses-V01_T1w.nii.gz  | sub-0006  | ses-V01   | T1w      | 6 months |
| sub-0007_ses-V01_T1w.nii.gz  | sub-0007  | ses-V01   | T1w      | 4 months |
| sub-0008_ses-V01_T1w.nii.gz  | sub-0008  | ses-V01   | T1w      | 0 months |
| sub-0009_ses-V01_T1w.nii.gz  | sub-0009  | ses-V01   | T2w      | 30 gestational_weeks |
