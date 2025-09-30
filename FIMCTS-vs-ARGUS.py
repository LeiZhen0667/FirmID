import os
import random
from typing import Dict, Set, List, Tuple, Optional
import time
# import matplotlib.pyplot as plt
import numpy as np
import math

# Set global random seed for reproducibility
random.seed(42)
np.random.seed(42)

""""
FIMCTS vs. ARGUS Comparison Test
In 30 rounds test
"""


# Simulated device class
class SimulatedDevice:
    def __init__(self, model_to_files, target_version=None):
        self.model_to_files = model_to_files
        self.request_count = 0

        # Use specified target version if provided
        if target_version:
            self.device_model = target_version
        else:
            # Otherwise choose a random version
            self.device_model = random.choice(list(model_to_files.keys()))

        print(f"[Simulation] Device model is: {self.device_model}")

    def request_file(self, file_name):
        self.request_count += 1
        response = file_name in self.model_to_files[self.device_model]
        print(f"Device received file request: {file_name}, Response: {response}")
        return response


def scan_model_folder(model_path: str) -> Dict[str, Set[str]]:
    """
    Scan model folder to create firmware version to files mapping dictionary

    Args:
        model_path: Path to the model folder

    Returns:
        Dict[str, Set[str]]: Mapping from firmware version to set of files
    """
    # Initialize result dictionary
    model_to_files = {}

    # Ensure input path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path does not exist: {model_path}")

    # Traverse all firmware version folders under model folder
    for firmware_version in os.listdir(model_path):
        firmware_path = os.path.join(model_path, firmware_version)

        # Ensure it's a directory
        if not os.path.isdir(firmware_path):
            continue

        # Initialize file set for this firmware version
        model_to_files[firmware_version] = set()

        # Traverse all files in firmware version folder
        for root, _, files in os.walk(firmware_path):
            for file in files:
                # Get relative path to firmware version folder
                rel_path = os.path.relpath(root, firmware_path)
                if rel_path == '.':
                    # File is in root of firmware version folder
                    file_path = file
                else:
                    # File is in subdirectory
                    file_path = os.path.join(rel_path, file)

                # Add file path to firmware version's set
                model_to_files[firmware_version].add(file_path)

    return model_to_files


class ARGUS:
    def __init__(self, model_to_files: Dict[str, Set[str]]):
        """
        Initialize ARGUS inference engine

        Args:
            model_to_files: Mapping from firmware version to file set
        """
        self.model_to_files = model_to_files
        self.all_files = sorted(set().union(*model_to_files.values()))
        self.fingerprint_table = {}
        self.request_history = []

    def build_feature_table(self) -> Dict[str, Dict[str, bool]]:
        """
        Build feature table: file x version -> existence

        Returns:
            Feature table dictionary {file: {version: existence}}
        """
        feature_table = {}
        for file in self.all_files:
            file_features = {}
            for version, files in self.model_to_files.items():
                file_features[version] = (file in files)
            feature_table[file] = file_features
        return feature_table

    def filter_identical_rows(self, feature_table: Dict[str, Dict[str, bool]],
                              remaining_versions: Set[str]) -> Dict[str, Dict[str, bool]]:
        """
        Filter out file rows that can't distinguish between remaining versions

        Args:
            feature_table: Current feature table
            remaining_versions: Set of remaining versions to distinguish

        Returns:
            Filtered feature table
        """
        filtered_table = {}
        for file, features in feature_table.items():
            # Check if file existence is consistent across all remaining versions
            values = set()
            for ver in remaining_versions:
                values.add(features[ver])

            # Only keep files that can distinguish versions (existence varies)
            if len(values) > 1:
                filtered_table[file] = features

        return filtered_table

    def infer_version(self, device) -> Optional[str]:
        """
        Infer device firmware version using ARGUS scheme

        Args:
            device: Simulated device instance

        Returns:
            Inferred firmware version, or None if undetermined
        """
        # Reset request history
        self.request_history = []

        # Initialize feature table
        feature_table = self.build_feature_table()
        remaining_versions = set(self.model_to_files.keys())

        # Prioritize known fingerprints if available
        if self.fingerprint_table:
            for file in self.fingerprint_table.keys():
                if file in feature_table:
                    # Move known fingerprints to front
                    feature_table = {file: feature_table[file],
                                     **{k: v for k, v in feature_table.items() if k != file}}

        while len(remaining_versions) > 1:
            # Step 1: Remove rows with identical features
            feature_table = self.filter_identical_rows(feature_table, remaining_versions)

            if not feature_table:
                print("Cannot further distinguish versions")
                return None

            # Get current most effective file (first in order)
            current_file = next(iter(feature_table.keys()))

            # Step 2: Request file from device
            exists_in_device = device.request_file(current_file)
            self.request_history.append(current_file)

            # Step 3: Remove incompatible versions
            new_remaining_versions = set()
            for version in remaining_versions:
                # File existence in this version
                exists_in_version = feature_table[current_file][version]
                if exists_in_version == exists_in_device:
                    new_remaining_versions.add(version)

            # Update remaining versions
            remaining_versions = new_remaining_versions

            # Remove requested file from feature table
            del feature_table[current_file]

        # Confirm final version
        if len(remaining_versions) == 1:
            detected_version = next(iter(remaining_versions))

            # Save fingerprint to table (keep up to 3 key files)
            if self.request_history:
                self.fingerprint_table = {file: True for file in self.request_history[:3]}
                print(f"Updated fingerprint table: {list(self.fingerprint_table.keys())}")

            return detected_version

        return None

    def get_request_count(self) -> int:
        """Get total file request count"""
        return len(self.request_history)

    def get_request_history(self) -> List[str]:
        """Get request history"""
        return self.request_history


class Node:
    def __init__(self, parent=None, file_name=None):
        self.parent = parent
        self.file_name = file_name
        self.children = []
        self.visits = 0
        self.wins = 0

    def ucb1(self, exploration_param=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self, all_files):
        return len(self.children) == len(all_files)

    def best_child(self):
        return max(self.children, key=lambda child: child.ucb1())

    def select_next_file_greedy(self, remaining_models: Set[str],
                                model_to_files: Dict[str, Set[str]],
                                requested_files: Set[str]) -> str:
        """
        Greedy strategy for selecting next file to request

        Args:
            remaining_models: Set of remaining possible models
            model_to_files: Mapping from model to file set
            requested_files: Set of already requested files

        Returns:
            Best file to request next
        """
        all_files = set().union(*model_to_files.values())
        available_files = all_files - requested_files

        if not available_files:
            return None

        best_files = []
        best_split_score = float('-inf')

        for file in available_files:
            # Count models containing this file
            models_with_file = sum(1 for model in remaining_models
                                   if file in model_to_files[model])
            models_without_file = len(remaining_models) - models_with_file

            # Calculate split score - closer to even split is better
            split_score = -abs(models_with_file - models_without_file)

            if split_score > best_split_score:
                best_split_score = split_score
                best_files = [file]
            elif split_score == best_split_score:
                best_files.append(file)

        # Randomly select one of the best files
        best_file = random.choice(best_files) if best_files else None

        return best_file

    def simulate(self, device, remaining_models, model_to_files, requested_files):
        """
        Simulate file request sequence using greedy strategy

        Args:
            device: Simulated device instance
            remaining_models: Set of remaining possible models
            model_to_files: Mapping from model to file set
            requested_files: Set of already requested files

        Returns:
            Tuple of (remaining models after simulation, file sequence)
        """
        print(f"[Simulation] Starting simulation from file {self.file_name}...")
        local_models = remaining_models.copy()
        file_sequence = []
        local_requested_files = requested_files.copy()

        # If simulation start, first file already selected by caller
        if self.file_name:
            response = device.request_file(self.file_name)
            local_requested_files.add(self.file_name)
            file_sequence.append(self.file_name)

            if response:
                local_models = {m for m in local_models
                                if self.file_name in model_to_files[m]}
            else:
                local_models = {m for m in local_models
                                if self.file_name not in model_to_files[m]}

        # Use greedy strategy for subsequent files
        while len(local_models) > 1:
            next_file = self.select_next_file_greedy(local_models,
                                                     model_to_files,
                                                     local_requested_files)

            if not next_file:
                break

            response = device.request_file(next_file)
            local_requested_files.add(next_file)
            file_sequence.append(next_file)

            if response:
                local_models = {m for m in local_models
                                if next_file in model_to_files[m]}
            else:
                local_models = {m for m in local_models
                                if next_file not in model_to_files[m]}

        print(f"[Simulation] Remaining possible models: {local_models}")
        return local_models, file_sequence


class FIMCTS:
    def __init__(self, model_to_files):
        self.model_to_files = model_to_files
        self.root = Node()
        self.remaining_models = set(model_to_files.keys())
        self.all_file_sequences = []
        self.simulation_results = []
        self.path_lengths = []
        self.shortest_path = None
        self.best_path_index = None
        self.used_first_files = set()  # Track files used as first request

    def infer_model(self, device, simulations=10):
        """
        Infer device model using FIMCTS algorithm

        Args:
            device: Simulated device instance
            simulations: Number of simulations to run

        Returns:
            Final inferred model set
        """
        all_files = set().union(*self.model_to_files.values())

        for i in range(simulations):
            print(f"\n[Simulation Count] {i + 1}/{simulations}")

            # Ensure first file is random and different each simulation
            available_first_files = all_files - self.used_first_files
            if not available_first_files:
                self.used_first_files = set()
                available_first_files = all_files.copy()

            first_file = random.choice(list(available_first_files))
            self.used_first_files.add(first_file)

            # Reset requested files for each simulation
            requested_files = set()
            node = self.root
            current_path = []

            # Selection phase
            while node.is_fully_expanded(all_files) and node.children:
                node = node.best_child()
                if node.file_name:
                    current_path.append(node.file_name)
                    requested_files.add(node.file_name)

            # Expansion phase
            if not node.is_fully_expanded(all_files):
                untried_files = list(all_files - {child.file_name for child in node.children})
                # Ensure random first file selection
                if not node.file_name:
                    new_file = first_file
                else:
                    new_file = random.choice(untried_files)
                new_node = Node(parent=node, file_name=new_file)
                node.add_child(new_node)
                node = new_node
                if new_file:
                    current_path.append(new_file)
                    requested_files.add(new_file)

            # Simulation phase
            remaining_models, file_sequence = node.simulate(
                device,
                self.remaining_models,
                self.model_to_files,
                requested_files
            )

            current_path.extend(file_sequence)
            self.simulation_results.append(frozenset(remaining_models))
            self.all_file_sequences.append(current_path)
            self.path_lengths.append(len(current_path))

            if self.shortest_path is None or len(current_path) < len(self.shortest_path):
                self.shortest_path = current_path.copy()
                self.best_path_index = i

            # Backpropagation phase
            while node is not None:
                node.visits += 1
                node.wins += 1 if device.device_model in remaining_models else 0
                node = node.parent

        return self.determine_final_model()

    def determine_final_model(self):
        """Determine final model based on simulation results"""
        # Collect and output statistics
        model_frequency = {}
        for model_set in self.simulation_results:
            if model_set in model_frequency:
                model_frequency[model_set] += 1
            else:
                model_frequency[model_set] = 1

        final_model_set = max(model_frequency, key=model_frequency.get)
        avg_path_length = sum(self.path_lengths) / len(self.path_lengths)

        return final_model_set


def compare_algorithms(model_to_files, num_tests=10):
    """
    Compare performance of ARGUS and FIMCTS algorithms

    Args:
        model_to_files: Mapping from firmware version to file set
        num_tests: Number of test rounds
    """
    # Prepare test data
    versions = list(model_to_files.keys())

    # Pre-generate target versions for consistent results
    target_versions = random.choices(versions, k=num_tests)

    # Result storage
    ARGUS_request_counts = []
    FIMCTS_request_counts = []
    ARGUS_times = []
    FIMCTS_times = []
    consistency_results = []

    for i in range(num_tests):
        # Use pre-generated target version
        target_version = target_versions[i]
        print(f"\n===== Test {i + 1}/{num_tests} [Target Version: {target_version}] =====")

        # Test ARGUS
        device_ARGUS = SimulatedDevice(model_to_files, target_version)
        agrus_inference = ARGUS(model_to_files)

        start_time = time.time()
        detected_version = agrus_inference.infer_version(device_ARGUS)
        ARGUS_time = time.time() - start_time

        ARGUS_request_counts.append(device_ARGUS.request_count)
        ARGUS_times.append(ARGUS_time)

        # Test FIMCTS
        device_FIMCTS = SimulatedDevice(model_to_files, target_version)
        fimcts = FIMCTS(model_to_files)

        start_time = time.time()
        best_model_set = fimcts.infer_model(device_FIMCTS, simulations=1)  # Single simulation
        FIMCTS_time = time.time() - start_time

        # Get request count for this simulation
        if fimcts.path_lengths:
            FIMCTS_request_counts.append(fimcts.path_lengths[-1])
        else:
            FIMCTS_request_counts.append(0)
        FIMCTS_times.append(FIMCTS_time)

        # Check result consistency
        if detected_version is not None:
            # Check if ARGUS result is in FIMCTS result set
            ARGUS_in_FIMCTS = detected_version in best_model_set
            # Check if FIMCTS result is unique and matches ARGUS
            FIMCTS_unique = len(best_model_set) == 1 and detected_version in best_model_set
        else:
            ARGUS_in_FIMCTS = False
            FIMCTS_unique = False

        consistency_results.append(ARGUS_in_FIMCTS)
        print(f"Results consistent: {ARGUS_in_FIMCTS}")
        print(f"ARGUS Result: {detected_version}, Requests: {device_ARGUS.request_count}, Time: {ARGUS_time:.4f}s")
        print(f"FIMCTS Result: {best_model_set}, Requests: {FIMCTS_request_counts[-1]}, Time: {FIMCTS_time:.4f}s")

    # Performance statistics
    avg_ARGUS_requests = np.mean(ARGUS_request_counts)
    avg_FIMCTS_requests = np.mean(FIMCTS_request_counts)
    avg_ARGUS_time = np.mean(ARGUS_times)
    avg_FIMCTS_time = np.mean(FIMCTS_times)
    consistency_rate = np.mean(consistency_results) * 100

    # Print summary results
    print("\n===== Performance Summary =====")
    print(f"ARGUS Avg Requests: {avg_ARGUS_requests:.2f}, Avg Time: {avg_ARGUS_time:.4f}s")
    print(f"FIMCTS Avg Requests: {avg_FIMCTS_requests:.2f}, Avg Time: {avg_FIMCTS_time:.4f}s")
    print(f"Result Consistency Rate: {consistency_rate:.1f}%")

    return {
        'ARGUS_avg_requests': avg_ARGUS_requests,
        'FIMCTS_avg_requests': avg_FIMCTS_requests,
        'ARGUS_avg_time': avg_ARGUS_time,
        'FIMCTS_avg_time': avg_FIMCTS_time,
        'consistency_rate': consistency_rate
    }


# Run comparison test
if __name__ == "__main__":
    # Device model and file mapping
    model_path = ("FIMCTS-test")
    result_dict = scan_model_folder(model_path)

    # Print dictionary preview
    print("Firmware versions:", len(result_dict))
    print("Total files:", sum(len(files) for files in result_dict.values()))

    # Run comparison test
    results = compare_algorithms(result_dict, num_tests=30)

    # Print efficiency improvement
    efficiency_gain = ((results['ARGUS_avg_requests'] - results['FIMCTS_avg_requests']) /
                       results['ARGUS_avg_requests']) * 100
    print(f"\nFIMCTS reduces file requests by {efficiency_gain:.1f}% compared to ARGUS")