# This code is from https://github.com/nipreps/nibabies/blob/master/nibabies/utils/bids.py
# Licensed under the Apache License, Version 2.0


# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
import os
import sys
import typing as ty
import warnings
from pathlib import Path
import argparse  # NEWLY ADDED

import pandas as pd
import shutil  # NEWLY ADDED

SUPPORTED_AGE_UNITS = (
    'weeks',
    'months',
    'years',
    'gestational_weeks',  # NEWLY ADDED: [add a 'gestational_weeks' unit]
)


def write_bidsignore(deriv_dir):
    # TODO: Port to niworkflows
    bids_ignore = (
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
        # Unspecified functional outputs
        '*_boldref.nii.gz',
        '*_bold.func.gii',
        '*_mixing.tsv',
        '*_AROMAnoiseICs.csv',
        '*_timeseries.tsv',
    )
    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.
    Examples
    --------
    >>> extract_entities('sub-01/anat/sub-01_T1w.nii.gz')
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_T1w.nii.gz'] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(['sub-01/anat/sub-01_run-1_T1w.nii.gz',
    ...                   'sub-01/anat/sub-01_run-2_T1w.nii.gz'])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    """
    from collections import defaultdict

    from bids.layout import parse_file_entities
    from niworkflows.utils.connections import listify

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def validate_input_dir(exec_env, bids_dir, participant_label):
    # Ignore issues and warnings that should not influence NiBabies
    import subprocess
    import tempfile

    validator_config_dict = {
        'ignore': [
            'EVENTS_COLUMN_ONSET',
            'EVENTS_COLUMN_DURATION',
            'TSV_EQUAL_ROWS',
            'TSV_EMPTY_CELL',
            'TSV_IMPROPER_NA',
            'VOLUME_COUNT_MISMATCH',
            'BVAL_MULTIPLE_ROWS',
            'BVEC_NUMBER_ROWS',
            'DWI_MISSING_BVAL',
            'INCONSISTENT_SUBJECTS',
            'INCONSISTENT_PARAMETERS',
            'BVEC_ROW_LENGTH',
            'B_FILE',
            'PARTICIPANT_ID_COLUMN',
            'PARTICIPANT_ID_MISMATCH',
            'TASK_NAME_MUST_DEFINE',
            'PHENOTYPE_SUBJECTS_MISSING',
            'STIMULUS_FILE_MISSING',
            'DWI_MISSING_BVEC',
            'EVENTS_TSV_MISSING',
            'TSV_IMPROPER_NA',
            'ACQTIME_FMT',
            'Participants age 89 or higher',
            'DATASET_DESCRIPTION_JSON_MISSING',
            'FILENAME_COLUMN',
            'WRONG_NEW_LINE',
            'MISSING_TSV_COLUMN_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_ELECTRODES',
            'UNUSED_STIMULUS',
            'CHANNELS_COLUMN_SFREQ',
            'CHANNELS_COLUMN_LOWCUT',
            'CHANNELS_COLUMN_HIGHCUT',
            'CHANNELS_COLUMN_NOTCH',
            'CUSTOM_COLUMN_WITHOUT_DESCRIPTION',
            'ACQTIME_FMT',
            'SUSPICIOUSLY_LONG_EVENT_DESIGN',
            'SUSPICIOUSLY_SHORT_EVENT_DESIGN',
            'MALFORMED_BVEC',
            'MALFORMED_BVAL',
            'MISSING_TSV_COLUMN_EEG_ELECTRODES',
            'MISSING_SESSION',
        ],
        'error': [],
        'ignoredFiles': ['/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = {s.name[4:] for s in bids_dir.glob('sub-*')}
        selected_subs = {s[4:] if s.startswith('sub-') else s for s in participant_label}
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict['ignoredFiles'].append(f'/sub-{sub}/**')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', str(bids_dir), '-c', temp.name])  # noqa: S607
        except FileNotFoundError:
            print('bids-validator does not appear to be installed', file=sys.stderr)


def parse_bids_for_age_months(
    bids_root: str | Path,
    subject_id: str,
    session_id: str | None = None,
) -> int | None:
    """
    Given a BIDS root, query the BIDS metadata files for participant age, and return in
    chronological months.

    The heuristic followed is:
    1) Check `sub-<subject_id>[/ses-<session_id>]/<sub-<subject_id>[_ses-<session-id>]_scans.tsv
    2) Check `sub-<subject_id>/sub-<subject_id>_sessions.tsv`
    3) Check `<root>/participants.tsv`
    """
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]
    if session_id and session_id.startswith('ses-'):
        session_id = session_id[4:]

    # Play nice with sessions
    subject = f'sub-{subject_id}'
    session = f'ses-{session_id}' if session_id else ''
    prefix = f'{subject}' + (f'_{session}' if session else '')

    subject_level = session_level = Path(bids_root) / subject
    if session_id:
        session_level = subject_level / session

    age = None

    scans_tsv = session_level / f'{prefix}_scans.tsv'
    if scans_tsv.exists():
        age = _get_age_from_tsv(
            scans_tsv,
            index_column='filename',
            index_value=r'^sub.*',
        )

    if age is not None:
        return age

    sessions_tsv = subject_level / f'{subject}_sessions.tsv'
    if sessions_tsv.exists() and session_id is not None:
        age = _get_age_from_tsv(sessions_tsv, index_column='session_id', index_value=session)

    if age is not None:
        return age

    participants_tsv = Path(bids_root) / 'participants.tsv'
    if participants_tsv.exists() and age is None:
        age = _get_age_from_tsv(
            participants_tsv, index_column='participant_id', index_value=subject
        )

    return age


def _get_age_from_tsv(
    bids_tsv: Path,
    index_column: str | None = None,
    index_value: str | None = None,
) -> float | None:
    df = pd.read_csv(str(bids_tsv), sep='\t')
    age_col = None

    for column in ('age_weeks', 'age_months', 'age_years', 'age'):
        if column in df.columns:
            age_col = column
            break
    if age_col is None:
        return

    df = df[df[index_column].str.fullmatch(index_value)]

    # Multiple indices may be present after matching
    if len(df) > 1:
        warnings.warn(
            f'Multiple matches for {index_column}:{index_value} found in {bids_tsv.name}.',
            stacklevel=1,
        )

    try:
        # extract age value from row
        age = float(df.loc[df.index[0], age_col].item())
    except Exception:  # noqa: BLE001
        return

    if age_col == 'age':
        # verify age is in months
        bids_json = bids_tsv.with_suffix('.json')
        age_units = _get_age_units(bids_json)
        if age_units is False:
            raise FileNotFoundError(
                f'Could not verify age unit for {bids_tsv.name} - ensure a sidecar JSON '
                'describing column `age` units is available.'
            )
    else:
        age_units = age_col.split('_')[-1]

    age_months = age_to_months(age, units=age_units)
    return age_months


def _get_age_units(bids_json: Path) -> ty.Literal['weeks', 'months', 'years', False]:
    try:
        data = json.loads(bids_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    units = data.get('age', {}).get('Units', '')
    if not isinstance(units, str):
        # Multiple units consfuse us
        return False

    if units.lower() in SUPPORTED_AGE_UNITS:
        return units.lower()
    return False


def age_to_months(age: int | float, units: ty.Literal['weeks', 'months', 'years', 'gestational_weeks']) -> int:
    """
    Convert a given age, in either "weeks", "months", or "years", into months.

    >>> age_to_months(1, 'years')
    12
    >>> age_to_months(0.5, 'years')
    6
    >>> age_to_months(2, 'weeks')
    0
    >>> age_to_months(3, 'weeks')
    1
    >>> age_to_months(8, 'months')
    8
    >>> age_to_months(20, 'gestational_weeks')
    -5
    """
    WEEKS_TO_MONTH = 0.230137
    YEARS_TO_MONTH = 12

    if units == 'weeks':
        age *= WEEKS_TO_MONTH
    elif units == 'years':
        age *= YEARS_TO_MONTH
    elif units == 'gestational_weeks':  # NEWLY ADDED: [add a 'gestational_weeks' unit]
        age = -age * WEEKS_TO_MONTH
    return int(round(age))


# NEWLY ADDED

# Function to ensure the necessary directories exist
def ensure_path_exists(path):
    if not os.path.exists(path):
        print(f"{path} does not exist, creating it!")
        os.makedirs(path, mode=0o777)

# Function to save output in the proper session folder
def save_output(output_dir, subject, session):
    # Create necessary session directories
    session_output_dir = os.path.join(output_dir, f"{subject}", f"{session}", "anat")
    ensure_path_exists(session_output_dir)

def copy_specific_bids_metadata(input_path: Path, output_path: Path, subject: str):
    """
    Copies .tsv and .json files for a specific subject and session in a BIDS dataset to the corresponding output location.

    Args:
        input_path (Path): Root directory of the input BIDS dataset.
        output_path (Path): Root directory for the output BIDS dataset.
        subject (str): Subject ID (e.g., "sub-001").
        session (str, optional): Session ID (e.g., "ses-01"). If None, session-specific files are ignored.
    """
    # Build the search path based on subject and session

    search_path = Path(input_path) / subject
        
    # Traverse all files in the input directory
    for root, _, files in os.walk(search_path):
        for file in files:
            # Check if the file has a .tsv or .json extension
            if file.endswith((".tsv", ".json")):
                # Build the full path of the source file
                src_file = Path(root, file)
                
                # Construct the corresponding destination path
                relative_path = src_file.relative_to(input_path)
                dest_file = Path(output_path, relative_path)
                
                # Ensure the destination directory exists
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} -> {dest_file}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brain MRIs Enhancement Foundation Model (BME-X)\n"
                    "Version: v1.0.2\n"
                    "Authors: Yue Sun, Limei Wang, Gang Li, Weili Lin, Li Wang\n"
                    "Reference: A foundation model for enhancing magnetic resonance images and downstream "
                    "segmentation, registration, and diagnostic tasks, Nat. Biomed. Eng (2024).https://doi.org/10.1038/s41551-024-01283-7\n"
                    "Contacts: li_wang@med.unc.edu, yuesun@med.unc.edu\n"
                    "Code: <https://github.com/DBC-Lab/Brain_MRI_Enhancement>\n"
                    "------------------------------------------",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--bids_dir", type=str, required=True, help="BIDS dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., sub-0001)")
    parser.add_argument("--session", type=str, required=True, help="Session ID (e.g., ses-V01)")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix (e.g., T1w)")
    
    args = parser.parse_args()
    save_output(args.output_dir, args.subject, args.session)
    copy_specific_bids_metadata(args.bids_dir, args.output_dir, args.subject)

    # Run the data processing
    age_months = parse_bids_for_age_months(args.bids_dir, args.subject, args.session)
    print(f"Age in months: {age_months}")