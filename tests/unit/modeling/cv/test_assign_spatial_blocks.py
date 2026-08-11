# tests/unit/modeling/cv/test_assign_spatial_blocks.py
"""Regression coverage for DatasetPrep.assign_spatial_blocks - the grid
partitioning step every spatial CV strategy (SpatialGroupKFoldCV,
StratifiedSpatialBlockCV) relies on for its `groups` argument. This had
zero direct test coverage before this file."""

import pandas as pd
import pytest

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep


@pytest.fixture
def prep():
    return DatasetPrep({"modeling": {}})


def test_assign_spatial_blocks_groups_points_in_same_cell_together(prep):
    df = pd.DataFrame({
        "_x": [100.0, 200.0, 4900.0],
        "_y": [100.0, 4900.0, 200.0],
    })
    blocks = prep.assign_spatial_blocks(df, block_size_m=5000.0)

    # all three points fall inside the same 5km cell (block (0, 0))
    assert blocks.nunique() == 1
    assert blocks.iloc[0] == blocks.iloc[1] == blocks.iloc[2]


def test_assign_spatial_blocks_separates_points_in_different_cells(prep):
    df = pd.DataFrame({
        "_x": [100.0, 5100.0, 100.0, 5100.0],
        "_y": [100.0, 100.0, 5100.0, 5100.0],
    })
    blocks = prep.assign_spatial_blocks(df, block_size_m=5000.0)

    # four distinct 5km cells: (0,0), (1,0), (0,1), (1,1)
    assert blocks.nunique() == 4


def test_assign_spatial_blocks_boundary_point_falls_in_upper_cell(prep):
    """floor-division semantics: a point exactly on a grid line belongs to
    the cell whose lower edge it sits on, not the cell below/left of it."""
    df = pd.DataFrame({"_x": [4999.0, 5000.0, 5001.0], "_y": [0.0, 0.0, 0.0]})
    blocks = prep.assign_spatial_blocks(df, block_size_m=5000.0)

    assert blocks.iloc[0] != blocks.iloc[1]
    assert blocks.iloc[1] == blocks.iloc[2]  # x=5000 and x=5001 share a cell
    assert blocks.iloc[0] != blocks.iloc[2]  # x=4999 is in the previous cell


def test_assign_spatial_blocks_smaller_block_size_yields_more_blocks(prep):
    df = pd.DataFrame({
        "_x": [i * 500.0 for i in range(20)],
        "_y": [0.0] * 20,
    })
    coarse = prep.assign_spatial_blocks(df, block_size_m=5000.0)
    fine = prep.assign_spatial_blocks(df, block_size_m=1000.0)

    assert fine.nunique() > coarse.nunique()


def test_assign_spatial_blocks_two_points_one_metre_apart_can_land_in_different_blocks(prep):
    """Documents the mechanism behind the missing-buffer gap (see
    test_spatial_group_kfold_buffer_gap.py): two points 1m apart, straddling
    a cell boundary, are assigned to different blocks with no exclusion
    zone between them - nothing here checks *distance*, only which side of
    the grid line a point falls on."""
    df = pd.DataFrame({"_x": [4999.5, 5000.5], "_y": [2500.0, 2500.0]})
    blocks = prep.assign_spatial_blocks(df, block_size_m=5000.0)

    assert blocks.iloc[0] != blocks.iloc[1]
