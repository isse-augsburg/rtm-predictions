import logging

import h5py
import numpy as np


# data_function must return [(data, label) ... (data, label)]

def get_index_sequence(filename):  #
    """ 
    Returns: 
        ([data, label)]: a sequence of simulation steps as data and the
        filling percentage of the last step as label
    """
    indices = [10, 20, 30, 40, 50]

    f = h5py.File(filename, 'r')
    # Cut off last column (z), since it is filled with 1s anyway
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()

    for state in all_states:
        try:
            filling_factor = \
                f['post']['singlestate'][state]['entityresults']['NODE'][
                    'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                    'res'][
                    ()]

        except KeyError:
            continue

        filling_factors_at_certain_times.append(filling_factor)

    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    last_filling = flat_fillings[-1]
    non_zeros = np.count_nonzero(last_filling)
    state_count = np.shape(last_filling)[0]
    filling_percentage = np.array(non_zeros / state_count)

    try:
        flat_fillings = [flat_fillings[j] for j in indices]
    except IndexError:
        return None

    return [(flat_fillings, filling_percentage)]


def get_all_sequences_for_file(filename):
    """
    Returns: 
        [(data, label)]:	All Sequences of a specified length that can be
        extracted from a file,
        filling percentage at that timestep.
    """
    all_sequences = list()
    t_begin = 0
    t_end = 10
    t_delta = 5
    t_target_offset = 1
    t_sequence_distance = 10
    t_final = 1000

    while t_end + t_target_offset < t_final:
        try:
            instance = __get_fillings_at_times(filename, t_begin, t_end,
                                               t_delta, t_end + t_target_offset)
        except Exception:
            break

        t_begin = t_begin + t_sequence_distance
        t_end = t_end + t_sequence_distance

        if instance is None:
            continue
        else:
            all_sequences.append(instance)

    if len(all_sequences) == 0:
        return None

    return all_sequences


def __get_fillings_at_times(filename, t_start, t_finish, t_delta, t_target):
    t_now = t_start

    try:
        f = h5py.File(filename, 'r')
    except OSError:
        logger = logging.getLogger(__name__)
        logger.error(">>> ERROR: FILE", filename,
                     "COULD NOT BE OPEND BY H5PY. THIS IS BAD. BIG OOooOF")
        return None

    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = list()
    filling_percentage = -1

    final_state = ""
    for s in all_states:
        final_state = s

    end_time = \
        f['post']['singlestate'][final_state]['entityresults']['NODE'][
            'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
            'indexval'][()]
    if end_time < t_target:
        raise Exception

    for state in all_states:
        try:
            time = f['post']['singlestate'][state]['entityresults']['NODE'][
                'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'indexval'][()]

            if time >= t_target:
                target_fillingstate = filling_factor = \
                    f['post']['singlestate'][state]['entityresults']['NODE'][
                        'FILLING_FACTOR']['ZONE1_set1'][
                        'erfblock'][
                        'res'][()]
                non_zeros = np.count_nonzero(target_fillingstate)
                state_count = np.shape(target_fillingstate)[0]
                filling_percentage = np.array(non_zeros / state_count)
                t_target = 9999999
                break
            if time >= t_finish:
                continue
            if time >= t_now:
                filling_factor = \
                    f['post']['singlestate'][state]['entityresults']['NODE'][
                        'FILLING_FACTOR']['ZONE1_set1'][
                        'erfblock'][
                        'res'][()]
                filling_factors_at_certain_times.append(filling_factor)
                t_now += t_delta

        except KeyError:
            continue

    if t_target != 9999999 or filling_factors_at_certain_times.__len__() != (
            t_finish - t_start) / t_delta:
        return None

    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]

    return flat_fillings, filling_percentage


def get_single_states_and_fillings(filename):
    """
    Returns: 
        [(data, label)]: Returns single simulation states as data and label
        for an autoencoder.
    """

    f = h5py.File(filename, 'r')
    all_states = f['post']['singlestate']

    filling_factors_at_certain_times = list()

    for state in all_states:
        try:
            filling_factor = \
                f['post']['singlestate'][state]['entityresults']['NODE'][
                    'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                    'res'][
                    ()]

        except KeyError:
            continue

        filling_factors_at_certain_times.append(filling_factor)

    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    single_states = [(data, data) for data in flat_fillings]

    return single_states
