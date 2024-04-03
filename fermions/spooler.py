"""
The module that contains all the necessary logic for the fermions.
"""

import numpy as np
from scipy.sparse.linalg import expm


from sqooler.schemes import ExperimentDict, ExperimentalInputDict
from sqooler.spoolers import create_memory_data


def nested_kronecker_product(a: list) -> np.ndarray:
    """putting together a large operator from a list of matrices.

    Provide an example here.

    Args:
        a (list): A list of matrices that can connected.

    Returns:
        array: An matrix that operates on the connected Hilbert space.
    """
    if len(a) == 2:
        return np.kron(a[0], a[1])
    else:
        return np.kron(a[0], nested_kronecker_product(a[1:]))


def jordan_wigner_transform(j: int, lattice_length: int) -> np.ndarray:
    """
    Builds up the fermionic operators in a 1D lattice.
    For details see : https://arxiv.org/abs/0705.1928

    Args:
        j : site index
        lattice_length :  how many sites does the lattice have ?

    Returns:
        psi_x: the field operator of creating a fermion on size j
    """
    p_arr = np.array([[0, 1], [0, 0]])
    z_arr = np.array([[1, 0], [0, -1]])
    id_arr = np.eye(2)
    operators = []
    for dummy in range(j):
        operators.append(z_arr)
    operators.append(p_arr)
    for dummy in range(lattice_length - j - 1):
        operators.append(id_arr)
    return nested_kronecker_product(operators)


def gen_circuit(exp_name: str, json_dict: ExperimentalInputDict) -> ExperimentDict:
    """The function the creates the instructions for the circuit.

    json_dict: The list of instructions for the specific run.
    """
    ins_list = json_dict.instructions
    n_shots = json_dict.shots
    if json_dict.seed is not None:
        np.random.seed(json_dict.seed)
    tweezer_len = 4  # length of the tweezer array
    n_states = 2 ** (2 * tweezer_len)

    # create all the raising and lowering operators
    lattice_length = 2 * tweezer_len
    lowering_op_list = []
    for i in range(lattice_length):
        lowering_op_list.append(jordan_wigner_transform(i, lattice_length))

    number_operators = []
    for i in range(lattice_length):
        number_operators.append(lowering_op_list[i].T.conj().dot(lowering_op_list[i]))
    # interaction Hamiltonian
    h_int = 0 * number_operators[0]
    for ii in range(tweezer_len):
        spindown_ind = 2 * ii
        spinup_ind = 2 * ii + 1
        h_int += number_operators[spindown_ind].dot(number_operators[spinup_ind])

    # work our way through the instructions
    psi = 1j * np.zeros(n_states)
    psi[0] = 1
    measurement_indices = []
    shots_array = []
    # pylint: disable=C0200
    # Fix this pylint issue whenever you have time, but be careful !
    for i in range(len(ins_list)):
        inst = ins_list[i]
        if inst.name == "load":
            latt_ind = inst.wires[0]
            psi = np.dot(lowering_op_list[latt_ind].T, psi)
        if inst.name == "fhop":
            # the first two indices are the starting points
            # the other two indices are the end points
            latt_ind = inst.wires
            theta = inst.params[0]
            # couple
            h_hop = lowering_op_list[latt_ind[0]].T.dot(lowering_op_list[latt_ind[2]])
            h_hop += lowering_op_list[latt_ind[2]].T.dot(lowering_op_list[latt_ind[0]])
            h_hop += lowering_op_list[latt_ind[1]].T.dot(lowering_op_list[latt_ind[3]])
            h_hop += lowering_op_list[latt_ind[3]].T.dot(lowering_op_list[latt_ind[1]])
            u_hop = expm(-1j * theta * h_hop)
            psi = np.dot(u_hop, psi)
        if inst.name == "fint":
            # the first two indices are the starting points
            # the other two indices are the end points
            theta = inst.params[0]
            u_int = expm(-1j * theta * h_int)
            psi = np.dot(u_int, psi)
        if inst.name == "fphase":
            # the first two indices are the starting points
            # the other two indices are the end points
            h_phase = 0 * number_operators[0]
            for ii in inst.wires:
                h_phase += number_operators[ii]
            theta = inst.params[0]
            u_phase = expm(-1j * theta * h_phase)
            psi = np.dot(u_phase, psi)
        if inst.name == "measure":
            measurement_indices.append(inst.wires[0])

    # only give back the needed measurments
    if measurement_indices:
        probs = np.abs(psi) ** 2
        result_inds = np.random.choice(np.arange(n_states), p=probs, size=n_shots)

        measurements = np.zeros((n_shots, len(measurement_indices)), dtype=int)
        for jj in range(n_shots):
            result = np.zeros(n_states)
            result[result_inds[jj]] = 1

            for ii, ind in enumerate(measurement_indices):
                observed = number_operators[ind].dot(result)
                observed = observed.dot(result)
                measurements[jj, ii] = int(observed)
        shots_array = measurements.tolist()

    exp_sub_dict = create_memory_data(shots_array, exp_name, n_shots)
    return exp_sub_dict
