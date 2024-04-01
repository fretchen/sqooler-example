"""
Test module for the spooler_singlequdit.py file.
"""

from typing import Iterator, Callable

import pytest
from pydantic import ValidationError

import numpy as np

from sqooler.schemes import get_init_status
from sqooler.spoolers import gate_dict_from_list
from sqooler.utils import run_json_circuit

from singlequdit.config import (
    spooler_object as sq_spooler,
    SingleQuditExperiment,
    LoadInstruction,
    MeasureBarrierInstruction,
    LocalSqueezingInstruction,
    RlzInstruction,
    RlxInstruction,
    SinglequditFullInstruction,
)


# pylint: disable=W0613, W0621
@pytest.fixture
def sqooler_setup_teardown() -> Iterator[None]:
    """
    Make sure that the storage folder is empty before and after the test.
    """
    # setup code here if required one day
    sq_spooler.display_name = "singlequdit"

    yield  # this is where the testing happens

    # teardown code if required


def test_pydantic_exp_validation() -> None:
    """
    Test that the validation of the experiment is working
    """
    experiment = {
        "instructions": [
            ["rlz", [0], [0.7]],
            ["measure", [0], []],
        ],
        "num_wires": 1,
        "shots": 3,
    }
    SingleQuditExperiment(**experiment)

    with pytest.raises(ValidationError):
        poor_experiment = {
            "instructions": [
                ["load", [7], []],
                ["load", [2], []],
                ["measure", [2], []],
                ["measure", [6], []],
                ["measure", [7], []],
            ],
            "num_wires": 2,
            "shots": 4,
            "wire_order": "interleaved",
        }
        SingleQuditExperiment(**poor_experiment)

    with pytest.raises(ValidationError):
        poor_experiment = {
            "instructions": [
                ["load", [7], []],
                ["load", [2], []],
                ["measure", [2], []],
                ["measure", [6], []],
                ["measure", [7], []],
            ],
            "num_wires": 1,
            "shots": 1e7,
            "wire_order": "interleaved",
        }
        SingleQuditExperiment(**poor_experiment)

    inst_config = {
        "name": "rlx",
        "parameters": ["omega"],
        "qasm_def": "gate lrx(omega) {}",
        "coupling_map": [[0]],
        "description": "Evolution under Lx",
    }
    assert inst_config == RlxInstruction.config_dict()


def test_load_instruction(sqooler_setup_teardown: Callable) -> None:
    """
    Test that the load instruction instruction is properly constrained.
    """
    inst_list = ["load", [0], [200.0]]
    gate_dict = gate_dict_from_list(inst_list)
    LoadInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["loads", [0], [200.0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LoadInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["load", [0, 1], [200.0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LoadInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["load", [1], [200.0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LoadInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["load", [0], [7e9]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LoadInstruction(**gate_dict.model_dump())

    # test that the load gate works out properly
    n_shots = 5
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [0], [50.0]],
                ["rlx", [0], [np.pi]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": n_shots,
            "wire_order": "interleaved",
        }
    }
    job_id = "2"
    data = run_json_circuit(job_payload, job_id, sq_spooler)
    shots_array = data["results"][0]["data"]["memory"]
    assert shots_array[0] == "50", "job result got messed up"
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) == n_shots, "shots_array got messed up"
    assert data["display_name"] == "singlequdit"
    assert data["backend_version"] == "0.3"


def test_local_rot_instruction() -> None:
    """
    Test that the rotation instruction is properly constrained.
    """
    inst_list = ["rlx", [0], [np.pi / 2]]
    gate_dict = gate_dict_from_list(inst_list)
    RlxInstruction(**gate_dict.model_dump())

    inst_list = ["rlz", [0], [np.pi / 2]]
    gate_dict = gate_dict_from_list(inst_list)
    RlzInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rly", [0], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [0, 1], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [1], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [0], [4 * np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())


def test_squeezing_instruction() -> None:
    """
    Test that the rotation instruction is properly constrained.
    """
    inst_list = ["rlz2", [0], [np.pi / 2]]
    gate_dict = gate_dict_from_list(inst_list)
    LocalSqueezingInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz", [0], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [0, 1], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [1], [np.pi / 2]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [0], [400 * np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # test the config
    inst_config = {
        "name": "rlz2",
        "parameters": ["chi"],
        "qasm_def": "gate rlz2(chi) {}",
        "coupling_map": [[0]],
        "description": "Evolution under lz2",
    }
    assert inst_config == LocalSqueezingInstruction.config_dict()


def test_measure_instruction() -> None:
    """
    Test that the rotation instruction is properly constrained.
    """
    inst_list = ["measure", [0], []]
    gate_dict = gate_dict_from_list(inst_list)
    MeasureBarrierInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["measures", [0], []]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MeasureBarrierInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["measure", [0, 1], []]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MeasureBarrierInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["measure", [1], []]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MeasureBarrierInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["measure", [0], [np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MeasureBarrierInstruction(**gate_dict.model_dump())


def test_check_json_dict() -> None:
    """
    See if the check of the json dict works out properly.
    """
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "interleaved",
        },
        "experiment_1": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
        },
    }
    # the wire order is missing
    with pytest.raises(KeyError):
        _, json_is_fine = sq_spooler.check_json_dict(job_payload)

    job_payload = {
        "experiment_0": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "interleaved",
        },
        "experiment_1": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "interleaved",
        },
    }
    err_msg, json_is_fine, _ = sq_spooler.check_json_dict(job_payload)
    assert json_is_fine


def test_z_gate() -> None:
    """
    Test that the z gate is properly applied.
    """

    # first submit the job
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "interleaved",
        },
        "experiment_1": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "interleaved",
        },
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, sq_spooler)

    shots_array = data["results"][0]["data"][  # pylint: disable=unsubscriptable-object
        "memory"
    ]
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # test the config
    inst_config = {
        "name": "rlz",
        "parameters": ["delta"],
        "qasm_def": "gate rlz(delta) {}",
        "coupling_map": [[0]],
        "description": "Evolution under the Z gate",
    }
    assert inst_config == RlzInstruction.config_dict()


def test_rydberg_full_instruction() -> None:
    """
    Test that the SinglequditFull  instruction is properly working.
    """
    inst_list = ["sq_full", [0], [0.7, 1, 3]]
    gate_dict = gate_dict_from_list(inst_list)
    SinglequditFullInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rydberg_full", [0], [0.7, 1, 3]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        SinglequditFullInstruction(**gate_dict.model_dump())

    # test that we cannot give too few wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["sq_full", [], [0.7, 1, 3]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        SinglequditFullInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["sq_full", [0], [0.7, 1, 3e7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        SinglequditFullInstruction(**gate_dict.model_dump())

    inst_config = {
        "name": "sq_full",
        "parameters": ["omega, delta, chi"],
        "qasm_def": "gate sq_full(omega, delta, chi) {}",
        "coupling_map": [[0]],
        "description": "Apply the full time evolution on the array.",
    }
    assert inst_config == SinglequditFullInstruction.config_dict()

    # also spins of same length
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [0], [50]],
                [
                    "sq_full",
                    [
                        0,
                    ],
                    [np.pi, 0, 0],
                ],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 5,
            "wire_order": "interleaved",
        }
    }

    job_id = "2"
    data = run_json_circuit(job_payload, job_id, sq_spooler)
    shots_array = data["results"][0]["data"]["memory"]
    assert shots_array[0] == "50", "job result got messed up"
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # are the instructions in ?
    assert len(data["results"][0]["data"]["instructions"]) == 3


def test_spooler_config(sqooler_setup_teardown: Callable) -> None:
    """
    Test that the back-end is properly configured and we can indeed provide those parameters
     as we would like.
    """
    sq_config_dict = {
        "description": "Setup of a cold atomic gas experiment with a single qudit.",
        "version": "0.3",
        "cold_atom_type": "spin",
        "gates": [
            {
                "name": "rlx",
                "parameters": ["omega"],
                "qasm_def": "gate lrx(omega) {}",
                "coupling_map": [[0]],
                "description": "Evolution under Lx",
            },
            {
                "name": "rlz",
                "parameters": ["delta"],
                "qasm_def": "gate rlz(delta) {}",
                "coupling_map": [[0]],
                "description": "Evolution under the Z gate",
            },
            {
                "name": "rlz2",
                "parameters": ["chi"],
                "qasm_def": "gate rlz2(chi) {}",
                "coupling_map": [[0]],
                "description": "Evolution under lz2",
            },
            {
                "coupling_map": [[0]],
                "description": "Apply the full time evolution on the array.",
                "name": "sq_full",
                "parameters": ["omega, delta, chi"],
                "qasm_def": "gate sq_full(omega, delta, chi) {}",
            },
        ],
        "max_experiments": 1000,
        "max_shots": 1000000,
        "simulator": True,
        "supported_instructions": [
            "rlx",
            "rlz",
            "rlz2",
            "barrier",
            "measure",
            "load",
            "sq_full",
        ],
        "num_wires": 1,
        "wire_order": "interleaved",
        "num_species": 1,
        "display_name": "singlequdit",
        "operational": True,
        "pending_jobs": None,
        "status_msg": None,
        "last_queue_check": None,
        "sign": False,
    }
    spooler_config_info = sq_spooler.get_configuration()
    assert spooler_config_info.model_dump() == sq_config_dict


def test_number_experiments() -> None:
    """
    Make sure that we cannot submit too many experiments.
    """

    # first test the system that is fine.

    inst_dict = {
        "instructions": [
            ["rlz", [0], [0.7]],
            ["measure", [0], []],
        ],
        "num_wires": 1,
        "shots": 3,
        "wire_order": "interleaved",
    }
    job_payload = {"experiment_0": inst_dict}
    job_id = "1"
    data = run_json_circuit(job_payload, job_id, sq_spooler)

    shots_array = data["results"][0]["data"][  # pylint: disable=unsubscriptable-object
        "memory"
    ]
    assert len(shots_array) > 0, "shots_array got messed up"

    # and now run too many experiments
    n_exp = 2000
    job_payload = {}
    for ii in range(n_exp):
        job_payload[f"experiment_{ii}"] = inst_dict
    job_id = "1"
    with pytest.raises(AssertionError):
        data = run_json_circuit(job_payload, job_id, sq_spooler)


def test_add_job() -> None:
    """
    Test if we can simply add jobs as we should be able too.
    """

    # first test the system that is fine.
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["rlx", [0], [np.pi]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 150,
            "wire_order": "interleaved",
        }
    }

    job_id = "1"
    status_msg_dict = get_init_status()
    status_msg_dict.job_id = job_id
    result_dict, status_msg_dict = sq_spooler.add_job(job_payload, job_id)
    # assert that all the elements in the result dict memory are of string '1 0'
    expected_value = "1"
    for element in result_dict.results[  # pylint: disable=unsubscriptable-object
        0
    ].data.memory:
        assert (
            element == expected_value
        ), f"Element {element} is not equal to {expected_value}"
