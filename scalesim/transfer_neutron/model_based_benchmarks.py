import sys

from npumatmul.benchmarks.common.network_parser import *
from multiprocessing import Process, Manager

from npumatmul.model.neutron import *
from npumatmul.model.coral import *
from npumatmul.model.nvidia import *
from npumatmul.model.systolic_tensor_array import *

import csv
import os
import time
from tqdm import tqdm


repo_root = "../"

neural_network_list = [
    # {"addr": repo_root+"data/neural_network_models/llama_3.2_1b_stats.csv", "name": "LLAMA"},
    # {"addr": repo_root+"data/neural_network_models/deepseek_r1_distill_qwen_1.5b_stats.csv", "name": "Deepseek"},
    {"addr": repo_root+"data/neural_network_models/gemma_3_270m_stats.csv", "name": "Gemma"},
    # {"addr": repo_root+"data/neural_network_models/h2o_danube3_500m_base_stats.csv", "name": "H2O"},
    # {"addr": repo_root+"data/neural_network_models/mobilellm_r1_950m_stats.csv", "name": "MobileLLM"},
    # {"addr": repo_root+"data/neural_network_models/qwen3_0.6b_stats.csv", "name": "QWEN"},
    # {"addr": repo_root+"data/neural_network_models/smollm2_135m_stats.csv", "name": "SMO-135"},
    # {"addr": repo_root+"data/neural_network_models/smollm2_360m_stats.csv", "name": "SMO-2360"}
]

neural_networks = [Neural_Network(nn["addr"], nn["name"]) for nn in neural_network_list]


npu_models_equivalent_macs_256 = [
    Neutron(16, 16, 512),
    Neutron(16, 16, 256),
    Coral(16, 16, 512),
    Coral(16, 16, 256),
    Nvidia(4),
    systolic_tensor_array(4, 2, 4),
    systolic_tensor_array(16, 1, 1)
]

npu_models_equivalent_macs_64_except_nvidia = [
    Neutron(8, 8, 128),
    Neutron(8, 8, 64),
    Coral(8, 8, 128),
    Coral(8, 8, 64),
    Nvidia(2),
    systolic_tensor_array(2, 2, 4),
    systolic_tensor_array(8, 1, 1)
]

npu_models_equivalent_macs_32 = [
    Neutron(4, 8, 64),
    Neutron(4, 8, 32),
    Coral(4, 8, 64),
    Coral(4, 8, 32),
    Nvidia(2),
    systolic_tensor_array(2, 2, 2)
]

npu_models_equivalent_bandwidth = [
    Neutron(16, 16, 512),
    Neutron(16, 16, 256),
    Coral(16, 16, 512),
    Coral(16, 16, 256),
    Nvidia(2),
    systolic_tensor_array(2, 2, 4),
    systolic_tensor_array(4, 2, 2),
]

# npu_models = npu_models_equivalent_macs_256 + npu_models_equivalent_macs_64_except_nvidia + npu_models_equivalent_macs_32 + npu_models_equivalent_bandwidth
npu_models = npu_models_equivalent_macs_256  # use one set for test


# this method runs the matmul part of a neural network on a the model of an NPU.
# It uses memoization for faster performance.
def run_model_based_benchmarks(npu_model, neural_network_model, memoized_performance,
                               progress_cb=None, report_every=50):

    matmuls = neural_network_model.parse()

    # will contain all statistics about the test
    total_stats = {}

    if progress_cb:
        progress_cb("add_total", len(matmuls))

    tick = 0

    for matmul in matmuls:

        a_matrix_dims = (matmul.a_matrix.rows, matmul.a_matrix.cols)
        b_matrix_dims = (matmul.b_matrix.rows, matmul.b_matrix.cols)
        all_matrix_dims = (a_matrix_dims, b_matrix_dims)

        res = {}

        if all_matrix_dims in memoized_performance:
            res = memoized_performance[all_matrix_dims]
        else:
            # calculate npu performance on model
            res = npu_model.matrix_multiply(a_matrix_dims, b_matrix_dims)

            # memoize it for next time
            memoized_performance[all_matrix_dims] = res.copy()

        if not total_stats:
            for key, value in res.items():
                # warning: need to take into account the batch dimension for each matmul!
                total_stats[key] = (value * matmul.batch)
        else:
            for key, value in res.items():
                # warning: need to take into account the batch dimension for each matmul!
                total_stats[key] += (value * matmul.batch)

        # ----- report progress -----
        if progress_cb:
            tick += 1
            if tick >= report_every:
                progress_cb("update", tick)
                tick = 0

    # unreported ticks
    if progress_cb and tick:
        progress_cb("update", tick)

    return total_stats.copy()


def print_csv(list_of_dics, name):

    keys = list_of_dics[0].keys()

    with open(name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dics)


# ========== Top Worker: each NPU a process ==========
def run_all_networks_on_npu_worker(npu, param_neural_networks, repo_root, progress_q):
    all_results_for_npu = []
    memoization_for_npu = {}
    npu_name = npu.name()

    def progress_cb(kind, value):
        progress_q.put((npu_name, kind, value))

    for network in param_neural_networks:
        progress_cb("set_model", network.name())
        result = run_model_based_benchmarks(
            npu, network, memoization_for_npu,
            progress_cb=progress_cb,
            report_every=50
        )
        result["network"] = network.name()
        result["npu"] = npu_name
        all_results_for_npu.append(result)

    csv_path = os.path.join(repo_root, "data/model_based_benchmark_results")
    os.makedirs(csv_path, exist_ok=True)
    print_csv(all_results_for_npu, os.path.join(csv_path, f"{npu_name}.csv"))

    # NPU finished
    progress_q.put((npu_name, "done", 0))


# ========== Top level (tqdm) ==========
def parallel_run_npus_on_neural_networks(param_npus, param_neural_networks, repo_root):
    manager = Manager()
    progress_q = manager.Queue()

    processes = []
    npu_names = [npu.name() for npu in param_npus]

    for npu in param_npus:
        p = Process(
            target=run_all_networks_on_npu_worker,
            args=(npu, param_neural_networks, repo_root, progress_q)
        )
        p.start()
        processes.append(p)

    bars = {}
    bars["TOTAL"] = tqdm(total=0, desc="TOTAL", position=0, dynamic_ncols=True)

    for i, name in enumerate(npu_names, start=1):
        bars[name] = tqdm(total=0, desc=name, position=i, dynamic_ncols=True)

    finished = set()

    while len(finished) < len(npu_names):
        npu_name, kind, value = progress_q.get()

        if kind == "add_total":
            # dynamic total++ (each NPU + overall progress)
            bars[npu_name].total = (bars[npu_name].total or 0) + value
            bars[npu_name].refresh()

            bars["TOTAL"].total = (bars["TOTAL"].total or 0) + value
            bars["TOTAL"].refresh()

        elif kind == "update":
            bars[npu_name].update(value)
            bars["TOTAL"].update(value)

        elif kind == "set_model":
            bars[npu_name].set_description(f"{npu_name} | {value}")

        elif kind == "done":
            finished.add(npu_name)

    for p in processes:
        p.join()

    for b in bars.values():
        b.close()


if __name__ == "__main__":
    parallel_run_npus_on_neural_networks(npu_models, neural_networks, repo_root)


"""
for model in npu_models_equivalent_bandwidth:
    print(model.properties())
"""