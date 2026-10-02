import pytest
from src.utils.physionet_mapping import parse_physionet_filename

def test_parse_physionet_filename():
    # S001R04
    subj, run, exp_type = parse_physionet_filename("S001R04.edf")
    assert subj == "S001"
    assert run == 4
    assert exp_type == "motor_imagery_left_right_fist"

    # S001R06
    subj, run, exp_type = parse_physionet_filename("S001R06.edf")
    assert subj == "S001"
    assert run == 6
    assert exp_type == "motor_imagery_both_fists_both_feet"

    # S001R10
    subj, run, exp_type = parse_physionet_filename("S001R10.edf")
    assert subj == "S001"
    assert run == 10
    assert exp_type == "motor_imagery_both_fists_both_feet"

    # S001R09 (Execution)
    subj, run, exp_type = parse_physionet_filename("S001R09.edf")
    assert subj == "S001"
    assert run == 9
    assert exp_type == "motor_execution_both_fists_both_feet"

    # T0 is ignored by extraction logic, we can test it indirectly or via the script
