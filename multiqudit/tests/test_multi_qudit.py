"""
Test module for the spooler_multiqudit.py file.
"""

from typing import Iterator, Callable

import numpy as np
import pytest

from pydantic import ValidationError

from sqooler.spoolers import gate_dict_from_list
from sqooler.utils import run_json_circuit
from multiqudit.config import (
    spooler_object as mq_spooler,
    MultiQuditExperiment,
    LocalSqueezingInstruction,
    RlxInstruction,
    RlzInstruction,
    RlxlyInstruction,
    RlzlzInstruction,
    MultiQuditFullInstruction,
)


# pylint: disable=W0613, W0621
@pytest.fixture
def sqooler_setup_teardown() -> Iterator[None]:
    """
    Make sure that the storage folder is empty before and after the test.
    """
    # setup code here if required one day
    mq_spooler.display_name = "multiqudit"

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
    MultiQuditExperiment(**experiment)

    with pytest.raises(ValidationError):
        poor_experiment = {
            "instructions": [
                ["load", [7], []],
                ["load", [2], []],
                ["measure", [2], []],
                ["measure", [6], []],
                ["measure", [7], []],
            ],
            "num_wires": 400,
            "shots": 4,
            "wire_order": "sequential",
        }
        MultiQuditExperiment(**poor_experiment)


def test_local_rot_instruction(sqooler_setup_teardown: Callable) -> None:
    """
    Test that the hop instruction instruction is properly constrained.
    """
    inst_list = ["rlx", [0], [0.7]]
    gate_dict = gate_dict_from_list(inst_list)
    RlxInstruction(**gate_dict.model_dump())

    inst_list = ["rlz", [0], [0.7]]
    gate_dict = gate_dict_from_list(inst_list)
    RlzInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rly", [0], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [0, 1], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [200], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlx", [0], [3 * np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxInstruction(**gate_dict.model_dump())

    inst_config = {
        "name": "rlx",
        "parameters": ["omega"],
        "qasm_def": "gate lrx(omega) {}",
        "coupling_map": [[0], [1], [2], [3], [4]],
        "description": "Evolution under Lx",
    }
    assert inst_config == RlxInstruction.config_dict()

    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [1], [50.0]],
                ["rlx", [1], [np.pi]],
                ["measure", [1], []],
            ],
            "num_wires": 2,
            "shots": 3,
            "wire_order": "sequential",
        },
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)
    shots_array = data["results"][0]["data"]["memory"]
    assert shots_array[0] == "50", "job result got messed up"

    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # are the instructions in ?
    assert len(data["results"][0]["data"]["instructions"]) == 3


def test_mq_full_instruction(sqooler_setup_teardown: Callable) -> None:
    """
    Test that the hop instruction instruction is properly constrained.
    """
    inst_list = ["multiqudit_full", [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]
    gate_dict = gate_dict_from_list(inst_list)
    MultiQuditFullInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["mq_full", [0, 1, 2], [0, 0, 0, 0, 0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MultiQuditFullInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["multiqudit_full", [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MultiQuditFullInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["multiqudit_full", [0, 7, 8], [0, 0, 0, 0, 0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MultiQuditFullInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["multiqudit_full", [0, 1, 2], [1e9, 0, 0, 0, 0]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        MultiQuditFullInstruction(**gate_dict.model_dump())

    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [1], [50.0]],
                ["multiqudit_full", [0, 1], [np.pi, 0, 0, 0, 0]],
                ["measure", [1], []],
            ],
            "num_wires": 2,
            "shots": 3,
            "wire_order": "sequential",
        },
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)
    shots_array = data["results"][0]["data"]["memory"]
    assert shots_array[0] == "50", "job result got messed up"

    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # are the instructions in ?
    assert len(data["results"][0]["data"]["instructions"]) == 3


def test_squeezing_instruction() -> None:
    """
    Test that the local squeezing instruction constrained.
    """
    inst_list = ["rlz2", [0], [0.7]]
    gate_dict = gate_dict_from_list(inst_list)
    LocalSqueezingInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz22", [0], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # test that we cannot give too many wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [0, 1], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [200], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlz2", [0], [200 * np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        LocalSqueezingInstruction(**gate_dict.model_dump())

    # test the config
    inst_config = {
        "name": "rlz2",
        "parameters": ["chi"],
        "qasm_def": "gate rlz2(chi) {}",
        "coupling_map": [[0], [1], [2], [3], [4]],
        "description": "Evolution under lz2",
    }
    assert inst_config == LocalSqueezingInstruction.config_dict()


def test_qudit_qudit_instruction() -> None:
    """
    Test that the qudit qudit instruction instruction is properly constrained.
    """
    inst_list = ["rlxly", [0, 1], [0.7]]
    gate_dict = gate_dict_from_list(inst_list)
    RlxlyInstruction(**gate_dict.model_dump())

    inst_list = ["rlzlz", [0, 1], [0.7]]
    gate_dict = gate_dict_from_list(inst_list)
    RlzlzInstruction(**gate_dict.model_dump())

    # test that the name is nicely fixed
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlzls", [0, 1], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxlyInstruction(**gate_dict.model_dump())

    # test that we cannot give too few wires
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlxly", [0], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxlyInstruction(**gate_dict.model_dump())

    # make sure that the wires cannot be above the limit
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlxly", [0, 200], [0.7]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxlyInstruction(**gate_dict.model_dump())

    # make sure that the parameters are enforced to be within the limits
    with pytest.raises(ValidationError):
        poor_inst_list = ["rlxly", [0, 1], [200 * np.pi]]
        gate_dict = gate_dict_from_list(poor_inst_list)
        RlxlyInstruction(**gate_dict.model_dump())

    inst_config = {
        "name": "rlxly",
        "parameters": ["J"],
        "qasm_def": "gate rlylx(J) {}",
        "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 1, 2, 3, 4]],
        "description": "Entanglement between neighboring gates with an xy interaction",
    }
    assert inst_config == RlxlyInstruction.config_dict()

    inst_config = {
        "name": "rlzlz",
        "parameters": ["J"],
        "qasm_def": "gate rlzlz(J) {}",
        "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 1, 2, 3, 4]],
        "description": "Entanglement between neighboring gates with a zz interaction",
    }
    assert inst_config == RlzlzInstruction.config_dict()


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
            "wire_order": "sequential",
        },
        "experiment_1": {
            "instructions": [
                ["rlz", [0], [0.7]],
                ["measure", [0], []],
            ],
            "num_wires": 1,
            "shots": 3,
            "wire_order": "sequential",
        },
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)

    shots_array = data["results"][0]["data"]["memory"]
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # test the config
    inst_config = {
        "name": "rlz",
        "parameters": ["delta"],
        "qasm_def": "gate rlz(delta) {}",
        "coupling_map": [[0], [1], [2], [3], [4]],
        "description": "Evolution under the Z gate",
    }
    assert inst_config == RlzInstruction.config_dict()


def test_barrier_gate() -> None:
    """
    Test that the barrier can be properly applied.
    """

    # first submit the job
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["barrier", [0, 1], []],
                ["measure", [0], []],
            ],
            "num_wires": 2,
            "shots": 3,
            "wire_order": "sequential",
        }
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)

    shots_array = data["results"][0]["data"]["memory"]
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"


def test_rlxly_gate() -> None:
    """
    Test that the barrier can be properly applied.
    """

    # first submit the job
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [0], [10.0]],
                ["load", [1], [1.0]],
                ["rlx", [0], [1.5707963267948966]],
                ["barrier", [0, 1], []],
                ["rlz2", [0], [0.0]],
                ["rlz", [1], [0.0]],
                ["rlxly", [0, 1], [0.0]],
                ["barrier", [0, 1], []],
                ["measure", [0], []],
                ["measure", [1], []],
            ],
            "num_wires": 2,
            "shots": 150,
            "wire_order": "sequential",
        }
    }

    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)

    shots_array = data["results"][0]["data"]["memory"]
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"

    # also spins of same length
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [0], [1.0]],
                ["load", [1], [1.0]],
                ["rlx", [0], [np.pi]],
                ["rlxly", [0, 1], [np.pi / 2]],
                ["measure", [0], []],
                ["measure", [1], []],
            ],
            "num_wires": 2,
            "shots": 150,
            "wire_order": "sequential",
        }
    }

    job_id = "2"
    data = run_json_circuit(job_payload, job_id, mq_spooler)

    shots_array = data["results"][0]["data"]["memory"]
    assert shots_array[0] == "0 1", "job_id got messed up"
    assert data["job_id"] == job_id, "job_id got messed up"
    assert len(shots_array) > 0, "shots_array got messed up"


def test_spooler_config(sqooler_setup_teardown: Callable) -> None:
    """
    Test that the back-end is properly configured and we can indeed provide those parameters
     as we would like.
    """

    mq_config_dict = {
        "description": "Setup of a cold atomic gas experiment with a multiple qudits.",
        "version": "0.2",
        "cold_atom_type": "spin",
        "gates": [
            {
                "name": "rlx",
                "parameters": ["omega"],
                "qasm_def": "gate lrx(omega) {}",
                "coupling_map": [[0], [1], [2], [3], [4]],
                "description": "Evolution under Lx",
            },
            {
                "name": "rlz",
                "parameters": ["delta"],
                "qasm_def": "gate rlz(delta) {}",
                "coupling_map": [[0], [1], [2], [3], [4]],
                "description": "Evolution under the Z gate",
            },
            {
                "name": "rlz2",
                "parameters": ["chi"],
                "qasm_def": "gate rlz2(chi) {}",
                "coupling_map": [[0], [1], [2], [3], [4]],
                "description": "Evolution under lz2",
            },
            {
                "name": "rlxly",
                "parameters": ["J"],
                "qasm_def": "gate rlylx(J) {}",
                "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 1, 2, 3, 4]],
                "description": "Entanglement between neighboring gates with an xy interaction",
            },
            {
                "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 1, 2, 3, 4]],
                "description": "Entanglement between neighboring gates with a zz interaction",
                "name": "rlzlz",
                "parameters": ["J"],
                "qasm_def": "gate rlzlz(J) {}",
            },
            {
                "coupling_map": [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
                "description": "Apply the Rydberg and Rabi coupling over the whole "
                "array.",
                "name": "multiqudit_full",
                "parameters": ["omega, delta, chi, Jxy, Jzz"],
                "qasm_def": "gate multiqudit_full(omega, delta, chi, Jxy, Jzz) " "{}",
            },
        ],
        "max_experiments": 1000,
        "max_shots": 1e6,
        "simulator": True,
        "supported_instructions": [
            "rlx",
            "rlz",
            "rlz2",
            "rlxly",
            "rlzlz",
            "barrier",
            "measure",
            "load",
            "multiqudit_full",
        ],
        "num_wires": 5,
        "wire_order": "interleaved",
        "num_species": 1,
        "display_name": "multiqudit",
        "pending_jobs": None,
        "status_msg": None,
        "last_queue_check": None,
        "sign": True,
        "kid": None,
        "operational": True,
    }
    spooler_config_info = mq_spooler.get_configuration()
    assert spooler_config_info.model_dump() == mq_config_dict


def test_number_experiments() -> None:
    """
    Make sure that we cannot submit too many experiments.
    """

    # first test the system that is fine.
    job_payload = {
        "experiment_0": {
            "instructions": [
                ["load", [0], [1.0]],
                ["load", [1], [1.0]],
                ["rlx", [0], [np.pi]],
                ["rlxly", [0, 1], [np.pi / 2]],
                ["measure", [0], []],
                ["measure", [1], []],
            ],
            "num_wires": 2,
            "shots": 150,
            "wire_order": "sequential",
        }
    }
    job_id = "1"
    data = run_json_circuit(job_payload, job_id, mq_spooler)

    shots_array = data["results"][0]["data"]["memory"]
    assert len(shots_array) > 0, "shots_array got messed up"
    inst_dict = {
        "instructions": [
            ["load", [0], [1.0]],
            ["load", [1], [1.0]],
            ["rlx", [0], [np.pi]],
            ["rlxly", [0, 1], [np.pi / 2]],
            ["measure", [0], []],
            ["measure", [1], []],
        ],
        "num_wires": 2,
        "shots": 150,
        "wire_order": "sequential",
    }

    # and now run too many experiments
    n_exp = 2000
    job_payload = {}
    for ii in range(n_exp):
        job_payload[f"experiment_{ii}"] = inst_dict
    job_id = "1"
    with pytest.raises(AssertionError):
        data = run_json_circuit(job_payload, job_id, mq_spooler)


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
            "wire_order": "sequential",
        }
    }

    job_id = "1"
    result_dict, _ = mq_spooler.add_job(job_payload, job_id)
    # assert that all the elements in the result dict memory are of string '1 0'
    expected_value = "1"
    for element in result_dict.results[  # pylint: disable=unsubscriptable-object
        0
    ].data.memory:
        assert (
            element == expected_value
        ), f"Element {element} is not equal to {expected_value}"
