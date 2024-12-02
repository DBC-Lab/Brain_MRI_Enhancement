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
