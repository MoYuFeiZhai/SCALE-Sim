import math
from npumatmul.model.common.utilities import smallest_multiple_above
from math import log2


class Neutron:
    def __init__(self, num_pipelines, num_macs, num_accums):
        """
        num_pipelines: number of pipelines
        num_macs: number of macs per pipeline
        num_accums: total number of 32-bit accumulator registers for the system
                    (num_accums / num_pipelines gives number of accum regs per pipeline)
        """

        self.P = num_pipelines
        self.M = num_macs
        self.A = num_accums / num_pipelines   # number of accumulator registers per pipeline

        self.param_total_number_of_macs = num_pipelines * num_macs
        self.param_total_number_of_accumulators = num_accums
        self.param_max_bandwidth = max(num_pipelines, num_macs)

    def name(self):
        return f"Neutron({self.P},{self.M},{self.A * self.P})"

    def properties(self):
        return f"{self.name()}: ({self.param_total_number_of_macs}, {self.param_total_number_of_accumulators}, {self.param_max_bandwidth})"

    def matrix_multiply(self, a_dims, b_dims):
        straight_result = self._matrix_multiply_data_stationary(a_dims, b_dims)
        transposed_result = self._matrix_multiply_data_stationary(
            (b_dims[1], b_dims[0]),
            (a_dims[1], a_dims[0])
        )

        if straight_result["total_clock_cycles"] <= transposed_result["total_clock_cycles"]:
            return straight_result
        else:
            return transposed_result

    def _get_empty_usage_dict(self):
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
            "total_clock_cycles": 0
        }

        return usage_dict

    def _matrix_multiply_data_stationary(self, a_dims, b_dims):
        # calculate dimensions according to hardware constraints (ensures no performance penalty is ensued)
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

        for a_row_group in range(0, int(a_rows / self.P)):

            # following calculates the number of accumulations needed for this a_row_group
            a_row_start = a_row_group * self.P
            a_row_end = a_row_start + self.P
            effective_P = self.P
            if a_row_end > a_dims[0]:
                effective_P = a_dims[0] - a_row_start

            for b_col_group in range(0, int(b_cols / self.A)):

                # following calculates the number of accumulations needed for this b_col_group
                b_col_start = b_col_group * self.A
                b_col_end = b_col_start + self.A
                effective_A = self.A
                if b_col_end > b_dims[1]:
                    effective_A = b_dims[1] - b_col_start

                for a_col_group in range(0, int(a_cols / self.M)):

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

                    if (a_col_group != 0):
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

        usage_dict["tcm_data_bandwidth_efficiency"] = tcm_data_active_cycles / usage_dict["total_clock_cycles"]
        usage_dict["tcm_weight_bandwidth_efficiency"] = tcm_weight_active_cycles / usage_dict["total_clock_cycles"]
        usage_dict["tcm_result_bandwidth_efficiency"] = tcm_result_active_cycles / usage_dict["total_clock_cycles"]

        return usage_dict

    def _matrix_multiply_weight_stationary(self, a_dims, b_dims):
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

        for b_col_group in range(0, int(b_cols / self.A)):
            for a_row_group in range(0, int(a_rows / self.P)):
                for a_col_group in range(0, int(a_cols / self.M)):

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

                    if (a_col_group != 0):
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

        usage_dict["tcm_data_bandwidth_efficiency"] = tcm_data_active_cycles / usage_dict["total_clock_cycles"]
        usage_dict["tcm_weight_bandwidth_efficiency"] = tcm_weight_active_cycles / usage_dict["total_clock_cycles"]
        usage_dict["tcm_result_bandwidth_efficiency"] = tcm_result_active_cycles / usage_dict["total_clock_cycles"]

        return usage_dict


"""
n1 = Neutron(4, 4, 32)
res = n1.matrix_multiply((3, 16), (16, 2))
print(res)
"""