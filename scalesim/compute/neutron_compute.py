"""
Neutron matmul architecture model.

This is a local port of the Neutron matmul engine used in
`scalesim/transfer_neutron/neutron.py`, placed under `scalesim/compute`
to make the architecture model available to Scale-Sim code.
"""

from __future__ import annotations

import math
from math import log2
import numpy as np


def smallest_multiple_above(base: int, value: int) -> int:
    """
    Return the smallest multiple of `base` that is >= `value`.
    """
    base_int = int(base)
    value_int = int(value)

    if base_int <= 0:
        raise ValueError("base must be > 0")
    if base_int != base:
        # Accept integer-like floats (e.g., 32.0) but reject others.
        if not math.isclose(base, base_int):
            raise ValueError("base must be an integer or integer-like float")
    if value_int != value:
        if not math.isclose(value, value_int):
            raise ValueError("value must be an integer or integer-like float")

    rem = value_int % base_int
    if rem == 0:
        return value_int
    return value_int + (base_int - rem)


class neutron_compute:
    def __init__(self, num_pipelines: int, num_macs: int, num_accums: int):
        """
        num_pipelines: number of pipelines
        num_macs: number of macs per pipeline
        num_accums: total number of 32-bit accumulator registers for the system
                    (num_accums / num_pipelines gives number of accum regs per pipeline)
        """

        if num_pipelines <= 0 or num_macs <= 0 or num_accums <= 0:
            raise ValueError("num_pipelines, num_macs, num_accums must be > 0")
        if num_accums % num_pipelines != 0:
            raise ValueError("num_accums must be divisible by num_pipelines")

        self.P = int(num_pipelines)
        self.M = int(num_macs)
        self.A = int(num_accums // num_pipelines)   # accum regs per pipeline

        self.param_total_number_of_macs = self.P * self.M
        self.param_total_number_of_accumulators = num_accums
        self.param_max_bandwidth = max(self.P, self.M)

    def name(self) -> str:
        return f"Neutron({self.P},{self.M},{self.A * self.P})"

    def properties(self) -> str:
        return (
            f"{self.name()}: "
            f"({self.param_total_number_of_macs}, "
            f"{self.param_total_number_of_accumulators}, "
            f"{self.param_max_bandwidth})"
        )

    def matrix_multiply(self, a_dims: tuple[int, int], b_dims: tuple[int, int]) -> dict:
        straight_result = self._matrix_multiply_data_stationary(a_dims, b_dims)
        transposed_result = self._matrix_multiply_data_stationary(
            (b_dims[1], b_dims[0]),
            (a_dims[1], a_dims[0]),
        )

        if straight_result["total_clock_cycles"] <= transposed_result["total_clock_cycles"]:
            return straight_result
        return transposed_result

    def _get_empty_usage_dict(self) -> dict:
        # define counters
        add_dict = {
            f"{m}+{m}": 0 for m in range(16, 16 + int(log2(self.M)))
        }

        usage_dict = {
            # multiplier related counters
            "8x8": 0,

            # adder related counters
            # **add_dict,

            # accumulation counters
            "accumulate": 0,

            # memory related counters
            "accum_read_byte": 0,
            "accum_write_byte": 0,

            "tcm_data_read_byte": 0,
            "tcm_weight_read_byte": 0,
            "tcm_result_write_byte": 0,

            # performance related counters
            "total_clock_cycles": 0,
        }

        return usage_dict

    def _matrix_multiply_data_stationary(
        self, a_dims: tuple[int, int], b_dims: tuple[int, int]
    ) -> dict:
        # calculate dimensions according to hardware constraints (ensures no performance penalty)
        assert a_dims[1] == b_dims[0], "The matrix multiply dimensions do not match"

        a_rows = smallest_multiple_above(self.P, a_dims[0])
        a_cols = smallest_multiple_above(self.M, a_dims[1])
        b_rows = smallest_multiple_above(self.M, b_dims[0])
        b_cols = smallest_multiple_above(self.A, b_dims[1])

        # get empty statistics dictionary
        usage_dict = self._get_empty_usage_dict()

        # calculate statistics
        ramp_up_time = 0
        ramp_down_time = 0

        # this is to fill the canvas at the very beginning,
        # no ramp up for weights as we need 1 per cycle (we have another dedicated bus)
        ramp_up_time += self.P

        for a_row_group in range(0, a_rows // self.P):

            # following calculates the number of accumulations needed for this a_row_group
            a_row_start = a_row_group * self.P
            a_row_end = a_row_start + self.P
            effective_P = self.P
            if a_row_end > a_dims[0]:
                effective_P = a_dims[0] - a_row_start

            for b_col_group in range(0, b_cols // self.A):

                # following calculates the number of accumulations needed for this b_col_group
                b_col_start = b_col_group * self.A
                b_col_end = b_col_start + self.A
                effective_A = self.A
                if b_col_end > b_dims[1]:
                    effective_A = b_dims[1] - b_col_start

                for a_col_group in range(0, a_cols // self.M):

                    # here, we get the data from the canvas
                    usage_dict["tcm_data_read_byte"] += self.M * self.P

                    # instead of loop on accumulators (for a in range(0, effective_A)) we do multiply A
                    if True:
                        # the reason for max is that in case effective_A is smaller than self.P,
                        # we still need to wait till the next P pipelines are fetched before we move
                        usage_dict["total_clock_cycles"] += 1 * max(effective_A, effective_P)
                        usage_dict["8x8"] += self.M * self.P * effective_A

                        number_of_additions = int(self.M / 2)
                        for m in range(16, 16 + int(log2(self.M))):
                            adder_name = f"{m}+{m}"
                            # usage_dict[adder_name] += number_of_additions * self.P * effective_A
                            number_of_additions = int(number_of_additions / 2)

                    if a_col_group != 0:
                        # if it is 0, we don't need to accumulate with the previous value at all
                        usage_dict["accumulate"] += 1 * self.P * effective_A
                        # if it is 0, we don't need to read the previous accumulator value, it should be 0
                        usage_dict["accum_read_byte"] += 4 * self.P * effective_A

                    usage_dict["accum_write_byte"] += 4 * self.P * effective_A

                    # only for the b matrix (weights)
                    usage_dict["tcm_weight_read_byte"] += self.M * effective_A

                # read the accumulator values
                usage_dict["accum_read_byte"] += 4 * (effective_A * self.P)
                usage_dict["tcm_result_write_byte"] += effective_A * self.P

        # we assume storing the final accum values after all iterations is done during the last loops
        # (note that 32-bit values are quantized back to 8)
        ramp_down_time = 0

        usage_dict["total_clock_cycles"] += ramp_up_time

        # compute additional statistics
        tcm_data_active_cycles = usage_dict["tcm_data_read_byte"] / self.M
        tcm_weight_active_cycles = usage_dict["tcm_weight_read_byte"] / self.M
        tcm_result_active_cycles = usage_dict["tcm_result_write_byte"] / self.M

        usage_dict["tcm_data_bandwidth_efficiency"] = (
            tcm_data_active_cycles / usage_dict["total_clock_cycles"]
        )
        usage_dict["tcm_weight_bandwidth_efficiency"] = (
            tcm_weight_active_cycles / usage_dict["total_clock_cycles"]
        )
        usage_dict["tcm_result_bandwidth_efficiency"] = (
            tcm_result_active_cycles / usage_dict["total_clock_cycles"]
        )

        return usage_dict

    def _matrix_multiply_weight_stationary(
        self, a_dims: tuple[int, int], b_dims: tuple[int, int]
    ) -> dict:
        """
        In this function the weights are semi-stationary...
        meaning that the same set of weights are kept in TCM and multiplied by different pipeline data
        """

        assert a_dims[1] == b_dims[0], "The matrix multiply dimensions do not match"

        a_rows = smallest_multiple_above(self.P, a_dims[0])
        a_cols = smallest_multiple_above(self.M, a_dims[1])
        b_rows = smallest_multiple_above(self.M, b_dims[0])
        b_cols = smallest_multiple_above(self.A, b_dims[1])

        # get empty statistics dictionary
        usage_dict = self._get_empty_usage_dict()

        ramp_up_time = 0
        ramp_down_time = 0

        # this is to fill the canvas at the very beginning,
        # no ramp up for weights as we need 1 per cycle (we have another dedicated bus)
        ramp_up_time += self.P

        for b_col_group in range(0, b_cols // self.A):
            for a_row_group in range(0, a_rows // self.P):
                for a_col_group in range(0, a_cols // self.M):

                    # here, we get the data from the canvas
                    usage_dict["tcm_data_read_byte"] += self.M * self.P

                    # instead of loop on accumulators (for a in range(0, self.A)) we do multiply A
                    if True:
                        usage_dict["total_clock_cycles"] += 1 * self.A
                        usage_dict["8x8"] += self.M * self.P * self.A

                        number_of_additions = int(self.M / 2)
                        for m in range(16, 16 + int(log2(self.M))):
                            adder_name = f"{m}+{m}"
                            # usage_dict[adder_name] += number_of_additions * self.P * self.A
                            number_of_additions = int(number_of_additions / 2)

                    if a_col_group != 0:
                        # if it is 0, we don't need to accumulate with the previous value at all
                        usage_dict["accumulate"] += 1 * self.P * self.A
                        # if it is 0, we don't need to read the previous accumulator value, it should be 0
                        usage_dict["accum_read_byte"] += 4 * self.P * self.A

                    usage_dict["accum_write_byte"] += 4 * self.P * self.A

                    # only for the b matrix (weights)
                    usage_dict["tcm_weight_read_byte"] += self.M * self.A

                # read the accumulator values
                usage_dict["accum_read_byte"] += 4 * (self.A * self.P)
                usage_dict["tcm_result_write_byte"] += self.A * self.P

        # we assume storing the final accum values after all iterations is done during the last loops
        # (note that 32-bit values are quantized back to 8)
        ramp_down_time = 0

        usage_dict["total_clock_cycles"] += ramp_up_time

        # compute additional statistics
        tcm_data_active_cycles = usage_dict["tcm_data_read_byte"] / self.M
        tcm_weight_active_cycles = usage_dict["tcm_weight_read_byte"] / self.M
        tcm_result_active_cycles = usage_dict["tcm_result_write_byte"] / self.M

        usage_dict["tcm_data_bandwidth_efficiency"] = (
            tcm_data_active_cycles / usage_dict["total_clock_cycles"]
        )
        usage_dict["tcm_weight_bandwidth_efficiency"] = (
            tcm_weight_active_cycles / usage_dict["total_clock_cycles"]
        )
        usage_dict["tcm_result_bandwidth_efficiency"] = (
            tcm_result_active_cycles / usage_dict["total_clock_cycles"]
        )

        return usage_dict


class neutron_compute:
    """
    Adapter to use the Neutron matmul model inside the Scale-Sim compute flow.
    This provides the same interface as systolic_compute_* classes.
    """

    def __init__(self):
        self.config = None

        self.ifmap_op_mat = None
        self.filter_op_mat = None
        self.ofmap_op_mat = None

        self.ifmap_prefetch_matrix = None
        self.filter_prefetch_matrix = None

        self.ifmap_demand_matrix = None
        self.filter_demand_matrix = None
        self.ofmap_demand_matrix = None

        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        self.total_cycles = 0
        self.neutron = None

        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

    def set_params(self, config_obj, ifmap_op_mat, filter_op_mat, ofmap_op_mat, **_kwargs):
        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat

        arr_row, arr_col = self.config.get_array_dims()
        num_accums = arr_row * arr_col
        self.neutron = neutron_compute(arr_row, arr_col, num_accums)

        # Matrix multiply dimensions: (Sr x T) * (T x Sc)
        Sr = int(self.ifmap_op_mat.shape[0])
        T = int(self.ifmap_op_mat.shape[1])
        Sc = int(self.filter_op_mat.shape[1])

        usage = self.neutron.matrix_multiply((Sr, T), (T, Sc))
        self.total_cycles = int(usage["total_clock_cycles"])
        self.ifmap_reads = int(usage["tcm_data_read_byte"])
        self.filter_reads = int(usage["tcm_weight_read_byte"])
        self.ofmap_writes = int(usage["tcm_result_write_byte"])

        total_macs = self.neutron.param_total_number_of_macs
        if self.total_cycles > 0 and total_macs > 0:
            util = usage["8x8"] / (self.total_cycles * total_macs)
        else:
            util = 0.0

        self.mapping_efficiency_per_fold = [util]
        self.compute_utility_per_fold = [util]

        self.params_set_flag = True

    def _flatten_unique(self, mat):
        flat = [int(x) for x in mat.flatten() if int(x) != -1]
        if not flat:
            return []
        seen = set()
        uniq = []
        for v in flat:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def _build_demand_matrix(self, address_pool, total_requests, total_cycles):
        if total_cycles <= 0:
            return np.ones((1, 1)) * -1

        if total_requests <= 0:
            return np.ones((total_cycles, 1)) * -1

        bw = max(1, int(math.ceil(total_requests / total_cycles)))
        demand = np.ones((total_cycles, bw)) * -1

        if not address_pool:
            address_pool = list(range(max(1, total_requests)))

        for i in range(total_requests):
            row = i // bw
            col = i % bw
            if row >= total_cycles:
                break
            demand[row][col] = address_pool[i % len(address_pool)]

        return demand

    def create_prefetch_matrices(self):
        assert self.params_set_flag, "Parameters are not set"

        ifmap_addrs = self._flatten_unique(self.ifmap_op_mat)
        filter_addrs = self._flatten_unique(self.filter_op_mat)

        if not ifmap_addrs:
            self.ifmap_prefetch_matrix = np.ones((1, 1)) * -1
        else:
            self.ifmap_prefetch_matrix = np.asarray(ifmap_addrs).reshape((1, len(ifmap_addrs)))

        if not filter_addrs:
            self.filter_prefetch_matrix = np.ones((1, 1)) * -1
        else:
            self.filter_prefetch_matrix = np.asarray(filter_addrs).reshape((1, len(filter_addrs)))

        self.prefetch_mat_ready_flag = True

    def create_demand_matrices(self):
        assert self.params_set_flag, "Parameters are not set"

        ifmap_pool = self._flatten_unique(self.ifmap_op_mat)
        filter_pool = self._flatten_unique(self.filter_op_mat)
        ofmap_pool = self._flatten_unique(self.ofmap_op_mat)

        self.ifmap_demand_matrix = self._build_demand_matrix(
            ifmap_pool, self.ifmap_reads, self.total_cycles
        )
        self.filter_demand_matrix = self._build_demand_matrix(
            filter_pool, self.filter_reads, self.total_cycles
        )
        self.ofmap_demand_matrix = self._build_demand_matrix(
            ofmap_pool, self.ofmap_writes, self.total_cycles
        )

        self.demand_mat_ready_flag = True

    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()
        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, "Computes not ready yet"
        return sum(self.mapping_efficiency_per_fold) / len(self.mapping_efficiency_per_fold)

    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, "Computes not ready yet"
        return sum(self.compute_utility_per_fold) / len(self.compute_utility_per_fold)

    def get_ifmap_requests(self):
        assert self.demand_mat_ready_flag, "Computes not ready yet"
        return self.ifmap_reads

    def get_filter_requests(self):
        assert self.demand_mat_ready_flag, "Computes not ready yet"
        return self.filter_reads

    def get_ofmap_requests(self):
        assert self.demand_mat_ready_flag, "Computes not ready yet"
        return self.ofmap_writes
