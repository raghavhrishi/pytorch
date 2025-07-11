"""
This script is used for binding the rank-process to cpu-cores.
The current binding options are as follows:
1. Node (node)
2. Socket (called socket)
3. Exclusive (called exclusive)
4. Core-Complex (Called core-complex)
"""

import os
import re
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser, REMAINDER
from collections import defaultdict
from enum import Enum

import pynvml
from torch.distributed.elastic.utils.logging import get_logger

# Commands/Paths to get system-level information
DEVICE_NUMA_NODE_ABSOLUTE_FILE_PATH_FORMAT = (
    "/sys/bus/pci/devices/{domain}:{bus}:{device_func}/numa_node"
)
NUMA_CPU_MAP_CMD = "/sys/devices/system/node/node{value}/cpumap"
CPU_MAP_CMD = '/proc/self/status | grep "Cpus_allowed:"'
THREAD_SIBLINGS_CMD = "/sys/devices/system/cpu/cpu{value}/topology/thread_siblings"
SHARED_CACHE_CMD = (
    "/sys/devices/system/node/node{value_1}/cpu{value_2}/cache/{value_3}/shared_cpu_map"
)
CACHE_CMD = "/sys/devices/system/node/node{value_1}/cpu{value_2}/cache"
SOCKET_CMD = "/sys/devices/system/node/node{value}/cpulist"
PHYSICAL_PACKAGE_ID_CMD = (
    "/sys/devices/system/cpu/cpu{value}/topology/physical_package_id"
)
POSSIBLE_NODES_CMD = "/sys/devices/system/node/possible"


class AffinityMode(str, Enum):
    NODE = "node"
    SOCKET = "socket"
    EXCLUSIVE = "exclusive"
    CORE_COMPLEX = "core-complex"


@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode


logger = get_logger(__name__)


def wrap_command_with_numa_bindings(
    *,
    command_args: tuple[str],
    gpu_index: int,
    numa_options: NumaOptions,
) -> tuple[str, tuple[str]]:
    """
    Args:
        command_args: The args for a command, such as might be input to Popen.
            Example: ("python", "trainer.py")
        gpu_index: The index of the GPU used by the process.
            Example: 0
        numa_options: See NumaOptions for details.

    Returns:
        Depending on numa_options, something like
            ("numactl", ("--cpunodebind=0", "--preferred=0", "python", "script.py"))
    """

    _throw_if_numactl_not_available()
    if numa_options.affinity_mode == AffinityMode.NODE:
        numactl_args = _get_numactl_args_for_node_affinity(gpu_index=gpu_index)
    # elif numa_options.affinity_mode == AffinityMode.SOCKET:
    #     numactl_args = _get_numactl_args_for_socket_affinity(gpu_index=gpu_index)
    # elif numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
    #     numactl_args = _get_numactl_args_for_exclusive_affinity(gpu_index=gpu_index)
    # elif numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
    #     numactl_args = _get_numactl_args_for_core_complex(gpu_index=gpu_index)
    else:
        raise ValueError(f"Unknown affinity mode: {numa_options.affinity_mode}")

    # Syntax for invoking a command but with numactl activated is numactl <args> command <args>
    return (*numactl_args, *command_args)


def _throw_if_numactl_not_available() -> None:
    if not shutil.which("numactl"):
        raise RuntimeError("numactl shell command is required for NUMA bindings.")


def _get_numa_args_for_node_affinity(*, gpu_index: int) -> tuple[str]:
    """
    Implements NODE affinity strategy.

    Returns a numactl command, minus the command it would normally execute. E.g.,
    ("numactl", "--cpunodebind=0", "--preferred=0").
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index)

    return (
        "numactl",
        f"--cpunodebind={numa_node_index}",
        f"--preferred={numa_node_index}",
    )


def _get_numa_node_index_for_gpu_index(gpu_index: int) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
    # Like 00000000:00:01.3
    bus_id = pci_info.busId

    # @nocommit print to make sure it's right
    domain, bus, device_func = bus_id.split(":")
    absolute_device_numa_node_path = DEVICE_NUMA_NODE_ABSOLUTE_FILE_PATH_FORMAT.format(
        domain=domain[-4:], bus=bus, device_func=device_func
    )

    with open(numa_file) as f:
        return int(f.read())


# def _get_gpu_index() -> int:
#     """
#     NOTE: We assume that a process will only schedule work onto the GPU
#     whose index matches the process's local rank. If this is not an accurate
#     assumption, you should not use this module, because it is likely to be a pessimization
#     compared to just skipping bindings or doing them manually.
#     """
#     return os.environ["LOCAL_RANK"]


def _get_gpu_index_to_numa_node_index() -> list[int]:
    return [
        _get_numa_node_index_for_gpu_index(gpu_index)
        for gpu_index in range(_get_system_gpu_count())
    ]


class System:
    # returns full set of CPUs affined to the NUMA node
    # includes reserved and non-reserved CPUs
    def get_numa_node_to_cpus(self, affinedNumaNode: int) -> int:
        numaCpuMapCmd = NUMA_CPU_MAP_CMD.format(value=affinedNumaNode)
        try:
            with open(numaCpuMapCmd) as file:
                outs = file.read()
        except FileNotFoundError:
            print(f"The file {numaCpuMapCmd} does not exist.")
        numaCpuMap = outs.split(",")
        numaCpuMapHex = "0x{}".format("".join(numaCpuMap))
        numaCpuMapVal = int(numaCpuMapHex, 16)
        return numaCpuMapVal

    # returns a bitmap for each core, its sibling cores
    def get_thread_siblings(self, cpu: int) -> int:
        threadsFile = THREAD_SIBLINGS_CMD.format(value=cpu)
        try:
            with open(threadsFile) as threads:
                threadsMap = threads.read()
                threadsMapHex = "0x{}".format("".join(threadsMap.split(",")))
                threadsMapVal = int(threadsMapHex, 16)
        except FileNotFoundError:
            print(f"The file {threadsFile} does not exist.")

        return threadsMapVal

    # get a bitmap of cpus that are allowed to the process
    def get_available_cpu(self) -> int:
        with open("/proc/self/status", "r") as file:
            for line in file:
                if "Cpus_allowed:" in line:
                    outs = line.strip()
                    break
        cpuMap = outs.split()[1].split(",")
        cpuMapHex = "0x{}".format("".join(cpuMap))
        cpuMapVal = int(cpuMapHex, 16)
        return cpuMapVal

    # get a sorted list of shared caches for the given cpu in the given node
    def get_shared_caches(self, node: int, cpuid: int) -> list[str]:
        rootdir = CACHE_CMD.format(value_1=node, value_2=cpuid)
        caches = []
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d) and len(file) >= 5 and file[:5] == "index":
                with open(os.path.join(d, "type")) as cacheTypeInfo:
                    cacheType = cacheTypeInfo.read()
                    if cacheType.find("Unified") != -1 or cacheType.find("Data") != -1:
                        caches.append(file)
        return sorted(caches)

    # returns for a cpu on a node, a bitmap of cpus sharing its given shared cache
    def get_cpus_in_shared_cache(self, node: int, cpuid: int, cacheName: str) -> int:
        sharedCacheCmd = SHARED_CACHE_CMD.format(
            value_1=node, value_2=cpuid, value_3=cacheName
        )
        try:
            with open(sharedCacheCmd) as cache_info:
                cpuInSharedCacheMap = cache_info.read()
                cpuInSharedCacheMapHex = "0x{}".format(
                    "".join(cpuInSharedCacheMap.split(","))
                )
                cpuInSharedCacheVal = int(cpuInSharedCacheMapHex, 16)
        except FileNotFoundError:
            print(
                f"The file {sharedCacheCmd.format(node, cpuid, cacheName)} does not exist."
            )
        return cpuInSharedCacheVal

    # returns a map for each socket its nodes
    def get_socket_node(self) -> dict[str, list[int]]:
        try:
            with open(POSSIBLE_NODES_CMD) as file:
                num_nodes = file.read()
        except FileNotFoundError:
            print(f"The file {POSSIBLE_NODES_CMD} does not exist.")

        num_nodes = re.split("-", num_nodes)
        socket_node = defaultdict(list)
        for n in range(0, int(num_nodes[1]) + 1):
            socketCmd = SOCKET_CMD.format(value=n)
            try:
                with open(socketCmd) as file:
                    print_node = file.read()
            except FileNotFoundError:
                print(f"The file {socketCmd} does not exist.")
            print_node = re.split("-", print_node)
            print_node[1] = print_node[1].split(",")
            try:
                physicalPackageIDCmd = PHYSICAL_PACKAGE_ID_CMD.format(
                    value=print_node[0]
                )
                with open(physicalPackageIDCmd) as file:
                    socket_id = file.read()
            except FileNotFoundError:
                print(f"The file {physicalPackageIDCmd} does not exist.")
            if socket_id not in socket_node.keys():
                socket_node[socket_id].append(n)
            else:
                socket_node[socket_id].append(n)
        return socket_node


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="This script is used for binding NUMA nodes with the program of choice. "
    )
    parser.add_argument(
        "--affinity",
        type=str,
        help="NUMA Config needed to launch (node,exclusive,core-complex,socket)",
    )
    parser.add_argument("--local_rank", type=int, required=False, help="Rank on node")
    # positional
    parser.add_argument(
        "training_executable",
        type=str,
        help="This contains the executable needed to run the program (eg: python,gcc,g++,nvcc)",
    )
    # Training Script added as an argument to the NUMA script
    parser.add_argument(
        "training_script_args",
        nargs=REMAINDER,
        help="This contains the rest of the launching script apart from the executable",
    )
    return parser.parse_args()


def get_local_rank(args) -> tuple[int, bool]:
    local_rank = None
    is_rank_required = args.local_rank is not None
    local_rank = os.environ["LOCAL_RANK"]
    return int(local_rank), is_rank_required


def get_cmd(
    args, numactlargs: list[str], local_rank: int, is_rank_required: bool
) -> list[str]:
    numactl_path = check_numactl()
    cmd = (
        [numactl_path]
        + numactlargs
        + [args.training_executable]
        + args.training_script_args
    )
    if is_rank_required:
        cmd = cmd + ["--local_rank={}".format(local_rank)]
    return cmd


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    processes = []
    process = subprocess.Popen(cmd, env=env)
    processes.append(process)
    for process in processes:
        process.wait()


class _NumaStrategy(ABC):
    def wrap_command_with_numa_bindings(
        *, command_args: tuple[str], gpu_index: int
    ) -> tuple[str]:
        pass

    def get_affined_gpus(self, numa_nodes: list[int]) -> tuple[int, int]:
        affinedNumaNode = numa_nodes[self.local_rank]
        affinedGpuIndex = 0
        affinedGpuCount = 0
        for i in range(len(numa_nodes)):
            if numa_nodes[i] == affinedNumaNode:
                if i == self.local_rank:
                    affinedGpuIndex = affinedGpuCount
                affinedGpuCount += 1
        return affinedGpuCount, affinedGpuIndex

    def get_args_string(self, core_ranges: list[str], key_index: int) -> list[str]:
        ranges_join = ",".join(core_ranges)
        phys_cpu = "--physcpubind="
        numactlargs = []
        numactlargs_phys_cpu = phys_cpu + ranges_join
        numactlargs += [numactlargs_phys_cpu, "--membind={}".format(key_index)]
        return numactlargs

    def rangify(self, cpulist: list[int]) -> list[str]:
        ranges = []
        first = last = cpulist[0]
        for index in cpulist[1:]:
            if index - last > 1:
                if first == last:
                    ranges.append("{}".format(first))
                else:
                    ranges.append("{}-{}".format(first, last))
                first = index
            last = index
        if first == last:
            ranges.append("{}".format(first))
        else:
            ranges.append("{}-{}".format(first, last))
        return ranges

    def get_numactl_args(self) -> list[str]:
        return []


class Node(Numa):
    """
    implements node numa-binding
    """

    def get_numactl_args(self) -> list[str]:
        numa_gpu = self.system.get_numa_nodes()
        return [
            "--cpunodebind={}".format(int(numa_gpu[self.local_rank])),
            "--membind={}".format(int(numa_gpu[self.local_rank])),
        ]


class Exclusive(Numa):
    """
    implements exclusive numa-binding
    """

    def __init__(self, local_rank: int, system: System):
        super().__init__(local_rank, system)

    def get_numactl_args(self) -> list[str]:
        # Step 1: get the number of gpus
        numGPUs = self.system.get_gpu_count()

        # Step 2: find numa nodes
        numaNodes = self.system.get_numa_nodes()
        # Step 3: find affinedNumaNode, number of gpus affined to that node
        # and the relative index of the gpu on that node
        affinedNumaNode = numaNodes[self.local_rank]
        affinedGpuCount, affinedGpuIndex = self.get_affined_gpus(numaNodes)

        # Step 4: Get all cpus affined to the numa node
        numaCpuMapVal = self.system.get_numa_node_to_cpus(affinedNumaNode)

        # Step 5: Get CPUs available
        cpuMapVal = self.system.get_available_cpu()

        # Step 6: find available cpus on the numa node
        numaCpus = numaCpuMapVal & cpuMapVal

        # Step 7: find number of cpus
        numaCpuLen = len(bin(numaCpus)) - 2

        # Step 8: get cpu ids on numa node
        cpuList = []
        for i in range(numaCpuLen):
            if (numaCpus >> i) & 1 == 1:
                cpuList.append(i)

        # Step 9: get list of cpus that are fully available
        # considering thread siblings
        coreIDs = {}
        cpuMask = 0
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                threadsMapVal = self.system.get_thread_siblings(i)
                coreID = (threadsMapVal & (-threadsMapVal)).bit_length() - 1
                if (numaCpus & threadsMapVal) == threadsMapVal:
                    coreIDs[coreID] = threadsMapVal
                cpuMask |= threadsMapVal

        # Step 10: allocate the cores to this gpu
        # based on relative index on this node
        numCpus = len(coreIDs) // affinedGpuCount
        startIndex = numCpus * affinedGpuIndex
        endIndex = startIndex + numCpus
        resultCpuVal = 0
        count = 0
        for i in sorted(coreIDs.keys()):
            if count >= startIndex and count < endIndex:
                resultCpuVal |= coreIDs[i]
            count += 1

        # Step 11: create a list of cpus
        resultCpuLen = len(bin(resultCpuVal)) - 2
        resultCpuList = []
        for i in range(resultCpuLen):
            if (resultCpuVal >> i) & 1 == 1:
                resultCpuList.append(i)

        # Step 12: range-ify the cpu list
        cpu_ranges = self.rangify(resultCpuList)
        numactlargs = self.get_args_string(cpu_ranges, affinedNumaNode)
        return numactlargs


class CoreComplex(Numa):
    """
    implements core-complex numa-binding
    """

    def __init__(self, local_rank: int, system: System):
        super().__init__(local_rank, system)

    def get_numactl_args(self) -> list[str]:
        # Step 1: get the number of gpus
        numGPUs = self.system.get_gpu_count()

        # Step 2: find numa nodes
        numaNodes = self.system.get_numa_nodes()

        # Step 3: find affinedNumaNode, number of gpus affined to that node
        # and the relative index of the gpu on that node
        affinedNumaNode = numaNodes[self.local_rank]
        affinedGpuCount, affinedGpuIndex = self.get_affined_gpus(numaNodes)

        # Step 4: Get all cpus affined to the numa node
        numaCpuMapVal = self.system.get_numa_node_to_cpus(affinedNumaNode)

        # Step 5: Get CPUs available
        cpuMapVal = self.system.get_available_cpu()

        # Step 6: find available cpus on the numa node
        numaCpus = numaCpuMapVal & cpuMapVal

        # Step 7: find number of cpus
        numaCpuLen = len(bin(numaCpus)) - 2

        # Step 8: get cpu ids on numa node
        cpuList = []
        for i in range(numaCpuLen):
            if (numaCpus >> i) & 1 == 1:
                cpuList.append(i)

        # Step 9: get list of cpus that are fully available
        # considering thread siblings
        coreIDs = {}
        cpuMask = 0
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                threadsMapVal = self.system.get_thread_siblings(i)
                coreID = (threadsMapVal & (-threadsMapVal)).bit_length() - 1
                if (numaCpus & threadsMapVal) == threadsMapVal:
                    coreIDs[coreID] = threadsMapVal
                cpuMask |= threadsMapVal

        # Step 10: get a list of highest level shared caches in the node
        # sorted by number of available cpus
        cpuMask = 0
        allSharedCacheCpuIDs = []
        cacheCpuCount = {}
        allCacheNames = {}
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                caches = self.system.get_shared_caches(affinedNumaNode, i)
                # select the highest level cache
                cache = caches[-1]
                cpusInSharedCacheVal = self.system.get_cpus_in_shared_cache(
                    affinedNumaNode, i, cache
                )
                cpusInSharedCacheValAvailable = cpusInSharedCacheVal & numaCpus
                allSharedCacheCpuIDs.append(i)
                cacheCpuCount[i] = len(
                    [
                        ones
                        for ones in bin(cpusInSharedCacheValAvailable)[2:]
                        if ones == "1"
                    ]
                )
                allCacheNames[i] = cache
                cpuMask |= cpusInSharedCacheVal
        # stable sort caches by number of cpus available
        allSharedCacheCpuIDs.sort(key=lambda i: -1 * cacheCpuCount[i])

        # Step 11: map gpu to a cache
        affinedCacheID = affinedGpuIndex % len(allSharedCacheCpuIDs)
        cpusSharedCacheVal = self.system.get_cpus_in_shared_cache(
            affinedNumaNode,
            allSharedCacheCpuIDs[affinedCacheID],
            allCacheNames[allSharedCacheCpuIDs[affinedCacheID]],
        )
        cpusSharedCacheVal = cpusSharedCacheVal & numaCpus

        # Step 12.1: create a list of cpus
        resultCpuLen = len(bin(cpusSharedCacheVal)) - 2
        resultCpuList = []
        for i in range(resultCpuLen):
            if (cpusSharedCacheVal >> i) & 1 == 1:
                resultCpuList.append(i)

        # Step 12.2: range-ify the cpu list
        cpu_ranges = self.rangify(resultCpuList)
        numactlargs = self.get_args_string(cpu_ranges, affinedNumaNode)
        return numactlargs


class Socket(Numa):
    """
    implements socket numa-binding
    """

    def __init__(self, local_rank: int, system: System):
        super().__init__(local_rank, system)

    def socket_get_nodes_for_rank(self) -> tuple[int, int]:
        numa_node_list = self.system.get_numa_nodes()
        socket_node = self.system.get_socket_node()
        socket_min = 0
        socket_max = 0
        for key, value in socket_node.items():
            if numa_node_list[self.local_rank] in value:
                socket_min = min(value)
                socket_max = max(value)
        return socket_min, socket_max

    def get_numactl_args(self) -> list[str]:
        socket_min, socket_max = self.socket_get_nodes_for_rank()
        numactlargs = []
        if socket_min != socket_max:
            numactlargs += [
                "--cpunodebind={}-{}".format(socket_min, socket_max),
                "--membind={}-{}".format(socket_min, socket_max),
            ]
        else:
            numactlargs += [
                "--cpunodebind={}".format(socket_min),
                "--membind={}".format(socket_max),
            ]
        return numactlargs


def main() -> None:
    args = parse_args()
    current_env = os.environ.copy()
    system = System()

    local_rank, is_rank_required = get_local_rank(args)
    if local_rank == None:
        raise Exception("local_rank is unknown")

    gpu_count = system.get_gpu_count()
    if local_rank >= gpu_count:
        raise Exception(
            "binding not supported, local_rank:{} is greater than gpu count:{}".format(
                local_rank, gpu_count
            )
        )

    if args.affinity == AffinityMode.NODE.value:
        numa = Node(local_rank, system)
    elif args.affinity == AffinityMode.EXCLUSIVE.value:
        numa = Exclusive(local_rank, system)
    elif args.affinity == AffinityMode.CORE_COMPLEX.value:
        numa = CoreComplex(local_rank, system)
    elif args.affinity == AffinityMode.SOCKET.value:
        numa = Socket(local_rank, system)
    else:
        raise Exception(f"Unknown affinity: {args.affinity}")

    numactlargs = numa.get_numactl_args()
    cmd = get_cmd(args, numactlargs, local_rank, is_rank_required)
    print(socket.gethostname(), "Local Rank:", local_rank, "Binding:", cmd)
    logger.info("%s.Local Rank:%s Binding:%s", socket.gethostname(), local_rank, cmd)
    run_cmd(cmd, current_env)


if __name__ == "__main__":
    main()
