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

import pandas as pd

SUPPORTED_AGE_UNITS = (
    'weeks',
    'months',
    'years',
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
    '''
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]
    if session_id and session_id.startswith('ses-'):
        session_id = session_id[4:]
    '''

    # Play nice with sessions
    subject = f'sub-{subject_id}'
    session = f'ses-{session_id}' if session_id else ''
    prefix = f'{subject}' + (f'_{session}' if session else '')


    subject_level = session_level = Path(bids_root) / subject
    #print('subject_level:', subject_level)
    if session_id:
        session_level = subject_level / session
        #print('session_level:', session_level)

    age = None

    #print('prefix:', prefix)

    scans_tsv = session_level / f'{prefix}_scans.tsv'
    if scans_tsv.exists():
        print('scans_tsv exists')
        age = _get_age_from_tsv(
            scans_tsv,
            index_column='filename',
            index_value=r'^anat*',
        )

    if age is not None:
        print('scans_tsv_age:', scans_tsv, age)
        return age

    sessions_tsv = subject_level / f'{subject}_sessions.tsv'
    if sessions_tsv.exists() and session_id is not None:
        age = _get_age_from_tsv(sessions_tsv, index_column='session_id', index_value=session)

    if age is not None:
        print('sessions_tsv_age:', sessions_tsv, age)
        return age

    participants_tsv = Path(bids_root) / 'participants.tsv'
    subses = subject + '-' + session
    if participants_tsv.exists() and age is None:
        age = _get_age_from_tsv(
            participants_tsv, index_column='participant_id', index_value=subses
        )
    print('participants_tsv_age:', participants_tsv, age)
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
    print('age_unit:', units)
    if not isinstance(units, str):
        # Multiple units consfuse us
        return False

    if units.lower() in SUPPORTED_AGE_UNITS:
        return units.lower()
    return False


def age_to_months(age: int | float, units: ty.Literal['weeks', 'months', 'years']) -> int:
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
    """
    WEEKS_TO_MONTH = 0.230137
    YEARS_TO_MONTH = 12

    if units == 'weeks':
        age *= WEEKS_TO_MONTH
    elif units == 'years':
        age *= YEARS_TO_MONTH
    return int(round(age))
