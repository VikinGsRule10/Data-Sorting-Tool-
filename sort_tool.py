#!/usr/bin/env python3
import os
import sys
import time
import random
import argparse
from typing import List, Tuple, Any


class ProgressBar:
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: int):
        self.current = current
        percent = min(100.0 * current / self.total, 100.0)
        filled_width = int(self.width * current / self.total)
        
        bar = '#' * filled_width + '-' * (self.width - filled_width)
        elapsed_time = time.time() - self.start_time
        
        if current > 0:
            eta = elapsed_time * (self.total - current) / current
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% {eta_str}')
        sys.stdout.flush()
    
    def finish(self):
        self.update(self.total)
        print()


class SortingAlgorithms:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.progress_bar = None
        self.progress_counter = 0
    
    def reset_counters(self):
        self.comparisons = 0
        self.swaps = 0
        self.progress_counter = 0
    
    def set_progress_bar(self, total_operations: int):
        """Initialize progress bar for sorting operations"""
        self.progress_bar = ProgressBar(total_operations)
        self.progress_counter = 0
    
    def update_progress(self):
        """Update progress bar if it exists"""
        if self.progress_bar:
            self.progress_counter += 1
            if self.progress_counter % max(1, self.progress_bar.total // 100) == 0:
                self.progress_bar.update(self.progress_counter)
    
    def finish_progress(self):
        """Finish progress bar"""
        if self.progress_bar:
            self.progress_bar.finish()
            self.progress_bar = None
    
    def bubble_sort(self, arr: List[Any]) -> List[Any]:
        self.reset_counters()
        n = len(arr)
        total_ops = n * n // 2  # Approximate operations
        self.set_progress_bar(total_ops)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                self.update_progress()
        
        self.finish_progress()
        return arr
    
    def selection_sort(self, arr: List[Any]) -> List[Any]:
        self.reset_counters()
        n = len(arr)
        total_ops = n * n // 2
        self.set_progress_bar(total_ops)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
                self.update_progress()
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
        
        self.finish_progress()
        return arr
    
    def insertion_sort(self, arr: List[Any]) -> List[Any]:
        self.reset_counters()
        n = len(arr)
        self.set_progress_bar(n)
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0:
                self.comparisons += 1
                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    self.swaps += 1
                    j -= 1
                else:
                    break
            arr[j + 1] = key
            self.update_progress()
        
        self.finish_progress()
        return arr
    
    def merge_sort(self, arr: List[Any]) -> List[Any]:
        self.reset_counters()
        n = len(arr)
        total_ops = n * (n.bit_length() - 1) if n > 0 else 1  # Approximate for O(n log n)
        self.set_progress_bar(total_ops)
        result = self._merge_sort_helper(arr)
        self.finish_progress()
        return result
    
    def _merge_sort_helper(self, arr: List[Any]) -> List[Any]:
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self._merge_sort_helper(arr[:mid])
        right = self._merge_sort_helper(arr[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[Any], right: List[Any]) -> List[Any]:
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            self.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
            self.swaps += 1
            self.update_progress()
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    def heap_sort(self, arr: List[Any]) -> List[Any]:
        self.reset_counters()
        n = len(arr)
        total_ops = n * (n.bit_length() - 1) if n > 0 else 1
        self.set_progress_bar(total_ops)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        
        # Extract elements one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            self._heapify(arr, i, 0)
            self.update_progress()
        
        self.finish_progress()
        return arr
    
    def _heapify(self, arr: List[Any], n: int, i: int):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n:
            self.comparisons += 1
            if arr[left] > arr[largest]:
                largest = left
        
        if right < n:
            self.comparisons += 1
            if arr[right] > arr[largest]:
                largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.swaps += 1
            self._heapify(arr, n, largest)
    
    def counting_sort(self, arr: List[int]) -> List[int]:
        self.reset_counters()
        if not arr:
            return arr
        
        n = len(arr)
        self.set_progress_bar(n * 2)  # Two main passes
        
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val + 1
        
        count = [0] * range_val
        output = [0] * len(arr)
        
        # Count occurrences
        for num in arr:
            count[num - min_val] += 1
            self.comparisons += 1
            self.update_progress()
        
        # Cumulative count
        for i in range(1, range_val):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
            self.swaps += 1
            self.update_progress()
        
        self.finish_progress()
        return output
    
    def radix_sort(self, arr: List[int]) -> List[int]:
        self.reset_counters()
        if not arr:
            return arr
        
        max_val = max(arr)
        digits = len(str(max_val))
        self.set_progress_bar(len(arr) * digits)
        
        exp = 1
        while max_val // exp > 0:
            self._counting_sort_for_radix(arr, exp)
            exp *= 10
        
        self.finish_progress()
        return arr
    
    def _counting_sort_for_radix(self, arr: List[int], exp: int):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
            self.comparisons += 1
            self.update_progress()
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            self.swaps += 1
            i -= 1
        
        for i in range(n):
            arr[i] = output[i]
    
    def bucket_sort(self, arr: List[float]) -> List[float]:
        self.reset_counters()
        if not arr:
            return arr
        
        n = len(arr)
        self.set_progress_bar(n * 2)
        
        min_val = min(arr)
        max_val = max(arr)
        range_val = max_val - min_val
        
        if range_val == 0:
            self.finish_progress()
            return arr
        
        buckets = [[] for _ in range(n)]
        
        # Put array elements into buckets
        for num in arr:
            bucket_index = int(n * (num - min_val) / range_val)
            if bucket_index == n:
                bucket_index = n - 1
            buckets[bucket_index].append(num)
            self.comparisons += 1
            self.update_progress()
        
        # Sort individual buckets using insertion sort
        for bucket in buckets:
            self._insertion_sort_for_bucket(bucket)
        
        # Concatenate buckets
        result = []
        for bucket in buckets:
            result.extend(bucket)
            self.update_progress()
        
        self.finish_progress()
        return result
    
    def _insertion_sort_for_bucket(self, arr: List[float]):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0:
                self.comparisons += 1
                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    self.swaps += 1
                    j -= 1
                else:
                    break
            arr[j + 1] = key


class DataGenerator:
    @staticmethod
    def generate_random_dataset(n: int) -> str:
        """Generate random dataset of size 10^n and save to file"""
        size = 10 ** n
        filename = f"random_dataset_{size}.txt"
        
        print(f"\nGenerating random dataset of size {size:,}...")
        
        # Create progress bar for data generation
        progress_bar = ProgressBar(size)
        
        try:
            with open(filename, 'w') as file:
                for i in range(size):
                    # Generate random integer between 1 and 100000
                    random_num = random.randint(1, 100000)
                    file.write(f"{random_num}\n")
                    
                    # Update progress bar every 1000 operations or for smaller datasets
                    if i % max(1, size // 100) == 0 or size <= 1000:
                        progress_bar.update(i + 1)
            
            progress_bar.finish()
            print(f"Dataset successfully generated and saved as: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
            return None


class DataSortingTool:
    def __init__(self):
        self.sorter = SortingAlgorithms()
        self.algorithms = {
            '1': ('Bubble Sort', self.sorter.bubble_sort),
            '2': ('Selection Sort', self.sorter.selection_sort),
            '3': ('Insertion Sort', self.sorter.insertion_sort),
            '4': ('Merge Sort', self.sorter.merge_sort),
            '5': ('Heap Sort', self.sorter.heap_sort),
            '6': ('Counting Sort', self.sorter.counting_sort),
            '7': ('Radix Sort', self.sorter.radix_sort),
            '8': ('Bucket Sort', self.sorter.bucket_sort)
        }
    
    def display_initial_menu(self):
        """Display initial menu for user choice"""
        print("\n" + "="*50)
        print("DATA SORTING TOOL - Welcome")
        print("="*50)
        print("What would you like to do?")
        print("1. Sort existing data from file")
        print("2. Generate random dataset")
        print("0. Exit")
        print("="*50)
    
    def get_initial_choice(self) -> str:
        """Get user's initial choice"""
        while True:
            choice = input("Enter your choice (0-2): ").strip()
            if choice in ['0', '1', '2']:
                return choice
            print("Invalid choice. Please enter 0, 1, or 2.")
    
    def get_dataset_size(self) -> int:
        """Get dataset size (n) for 10^n elements"""
        while True:
            try:
                n = input("Enter the value of 'n' for dataset size 10^n (recommended: 1-6): ").strip()
                n = int(n)
                if n < 0:
                    print("Please enter a non-negative integer.")
                    continue
                if n > 7:
                    confirm = input(f"Dataset size will be {10**n:,} elements. This might take a while. Continue? (y/n): ")
                    if confirm.lower() not in ['y', 'yes']:
                        continue
                return n
            except ValueError:
                print("Please enter a valid integer.")
    
    def load_data_from_file(self, file_path: str) -> List[Any]:
        """Load data from file, supporting different formats"""
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()
                
                # Try to parse as numbers (integers or floats)
                try:
                    # Check if comma-separated
                    if ',' in content:
                        data = [self._parse_number(x.strip()) for x in content.split(',')]
                    # Check if space-separated
                    elif ' ' in content:
                        data = [self._parse_number(x.strip()) for x in content.split()]
                    # Check if newline-separated
                    elif '\n' in content:
                        data = [self._parse_number(x.strip()) for x in content.split('\n') if x.strip()]
                    else:
                        # Single number or treat as single item
                        data = [self._parse_number(content)]
                    
                    return data
                except ValueError:
                    # If parsing as numbers fails, treat as strings
                    if ',' in content:
                        return [x.strip() for x in content.split(',')]
                    elif ' ' in content:
                        return [x.strip() for x in content.split()]
                    elif '\n' in content:
                        return [x.strip() for x in content.split('\n') if x.strip()]
                    else:
                        return [content]
                        
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def _parse_number(self, s: str):
        """Parse string as int or float"""
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            return s  # Return as string if not a number
    
    def display_menu(self):
        """Display sorting algorithm menu"""
        print("\n" + "="*50)
        print("DATA SORTING TOOL - Choose Sorting Algorithm")
        print("="*50)
        for key, (name, _) in self.algorithms.items():
            print(f"{key}. {name}")
        print("0. Exit")
        print("="*50)
    
    def get_user_choice(self) -> str:
        """Get user's choice for sorting algorithm"""
        while True:
            choice = input("Enter your choice (0-8): ").strip()
            if choice in ['0'] + list(self.algorithms.keys()):
                return choice
            print("Invalid choice. Please enter a number between 0-8.")
    
    def get_file_path(self, initial_prompt: bool = True) -> str:
        """Get file path from user with option to exit"""
        if initial_prompt:
            prompt = "Enter the path to your data file (or type '0' to go back): "
        else:
            prompt = "Enter the path to your new data file (or type '0' to go back): "
        
        while True:
            file_path = input(prompt).strip()
            
            if file_path == '0':
                return None
            
            if not file_path:
                print("Please enter a valid file path or type '0' to go back.")
                continue
            
            # Try to load data from file
            data = self.load_data_from_file(file_path)
            if data is not None:
                print(f"\nSuccessfully loaded {len(data)} elements from file.")
                print(f"Data preview: {data[:5]}{'...' if len(data) > 5 else ''}")
                return file_path, data
            else:
                print("Failed to load data. Please try again with a valid file path.")
    
    def run_sorting(self, data: List[Any], choice: str) -> Tuple[List[Any], float]:
        """Run the selected sorting algorithm and measure performance"""
        if choice not in self.algorithms:
            return None, 0
        
        algorithm_name, algorithm_func = self.algorithms[choice]
        data_copy = data.copy()  # Don't modify original data
        
        print(f"\nRunning {algorithm_name}...")
        print("Progress:")
        
        start_time = time.time()
        
        # Special handling for algorithms that require specific data types
        if choice in ['6', '7'] and not all(isinstance(x, int) for x in data_copy):
            print("Warning: Counting Sort and Radix Sort work best with integers.")
            print("Converting data to integers where possible...")
            try:
                data_copy = [int(float(x)) if isinstance(x, (int, float)) else hash(str(x)) % 1000 for x in data_copy]
            except:
                print("Error: Cannot convert data for this algorithm.")
                return None, 0
        
        if choice == '8' and not all(isinstance(x, (int, float)) for x in data_copy):
            print("Warning: Bucket Sort works best with numeric data.")
            print("Converting data to floats where possible...")
            try:
                data_copy = [float(x) if isinstance(x, (int, float)) else float(hash(str(x)) % 1000) for x in data_copy]
            except:
                print("Error: Cannot convert data for this algorithm.")
                return None, 0
        
        sorted_data = algorithm_func(data_copy)
        end_time = time.time()
        
        return sorted_data, end_time - start_time
    
    def display_results(self, original_data: List[Any], sorted_data: List[Any], 
                       execution_time: float, algorithm_name: str):
        """Display sorting results and statistics"""
        print("\n" + "="*60)
        print(f"SORTING RESULTS - {algorithm_name}")
        print("="*60)
        
        print(f"Original data ({len(original_data)} elements):")
        print(f"{original_data[:10]}{'...' if len(original_data) > 10 else ''}")
        
        print(f"\nSorted data:")
        print(f"{sorted_data[:10]}{'...' if len(sorted_data) > 10 else ''}")
        
        print(f"\nPerformance Statistics:")
        print(f"- Comparisons: {self.sorter.comparisons:,}")
        print(f"- Swaps/Moves: {self.sorter.swaps:,}")
        print(f"- Execution Time: {execution_time:.6f} seconds")
        print(f"- Data Size: {len(original_data):,} elements")
        print("="*60)
    
    def save_results(self, sorted_data: List[Any], original_filename: str):
        """Save sorted results to a new file"""
        base_name = os.path.splitext(original_filename)[0]
        output_filename = f"{base_name}_sorted.txt"
        
        try:
            with open(output_filename, 'w') as file:
                for item in sorted_data:
                    file.write(f"{item}\n")
            print(f"\nSorted data saved to: {output_filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def get_next_action_choice(self):
        """Get user's choice for next action after sorting"""
        print("\n" + "-"*50)
        print("What would you like to do next?")
        print("1. Try another sorting algorithm on the same dataset")
        print("2. Load a different dataset")
        print("3. Go back to main menu")
        print("0. Exit the tool")
        print("-"*50)
        
        while True:
            choice = input("Enter your choice (0-3): ").strip()
            if choice in ['0', '1', '2', '3']:
                return choice
            print("Invalid choice. Please enter 0, 1, 2, or 3.")
    
    def run(self):
        """Main application loop"""
        print("DATA SORTING TOOL")
        print("Cross-platform CLI sorting utility")
        print("-" * 40)
        
        while True:
            self.display_initial_menu()
            initial_choice = self.get_initial_choice()
            
            if initial_choice == '0':
                print("Thank you for using Data Sorting Tool!")
                break
            
            elif initial_choice == '1':
                # Sort existing data from file
                result = self.get_file_path()
                if result is None:  # User chose to go back
                    continue
                
                current_file_path, current_data = result
                self._sorting_workflow(current_data, current_file_path)
            
            elif initial_choice == '2':
                # Generate random dataset
                n = self.get_dataset_size()
                filename = DataGenerator.generate_random_dataset(n)
                
                if filename:
                    data = self.load_data_from_file(filename)
                    if data:
                        self._sorting_workflow(data, filename)
                    else:
                        print("Failed to load generated dataset.")
                else:
                    print("Failed to generate dataset.")
    
    def _sorting_workflow(self, current_data: List[Any], current_file_path: str):
        """Handle the sorting workflow for a given dataset"""
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == '0':
                return  # Go back to main menu
            
            algorithm_name = self.algorithms[choice][0]
            sorted_data, execution_time = self.run_sorting(current_data, choice)
            
            if sorted_data is not None:
                self.display_results(current_data, sorted_data, execution_time, algorithm_name)
                
                # Ask if user wants to save results
                save_choice = input("\nSave sorted data to file? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    self.save_results(sorted_data, current_file_path)
                
                # Ask what to do next
                next_action = self.get_next_action_choice()
                
                if next_action == '0':  # Exit tool
                    print("Thank you for using Data Sorting Tool!")
                    sys.exit(0)
                elif next_action == '1':  # Try another algorithm on same data
                    continue  # Continue inner loop with same data
                elif next_action == '2':  # Load different dataset
                    result = self.get_file_path(initial_prompt=False)
                    if result is None:
                        continue
                    current_file_path, current_data = result
                    continue
                elif next_action == '3':  # Go back to main menu
                    return
            else:
                print("Sorting failed. Please try again.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CLI Data Sorting Tool')
    parser.add_argument('--version', action='version', version='Data Sorting Tool v1.1')
    parser.add_argument('--file', '-f', help='Input file path')
    
    args = parser.parse_args()
    
    tool = DataSortingTool()
    
    if args.file:
        # If file is provided via command line, load it directly
        data = tool.load_data_from_file(args.file)
        if data is not None:
            print(f"Loaded {len(data)} elements from {args.file}")
            print(f"Data preview: {data[:5]}{'...' if len(data) > 5 else ''}")
            tool._sorting_workflow(data, args.file)
    else:
        tool.run()


if __name__ == "__main__":
    main()
