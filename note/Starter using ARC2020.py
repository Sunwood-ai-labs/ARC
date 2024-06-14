# %% [markdown]
# # A Starter using ARC2020 Notebook 🏆📝
#
# ## Introduction
# Welcome to the ARC Prize 2024 Submission Notebook! This notebook is designed to facilitate the submission process for the ARC (Abstraction and Reasoning Corpus) Prize 2024 competition. The competition challenges participants to develop algorithms capable of solving a variety of abstract reasoning tasks.
#
# ## Functionality
# This notebook provides functionalities to:
# - Translate submissions from an old CSV format to the new JSON format required for ARC Prize 2024.
# - Predict solutions for ARC tasks using various techniques such as object detection and manipulation.
# - Generate a submission CSV file following the competition guidelines.
#
# ## Guidelines
# To use this notebook effectively:
# 1. **Data Preparation**:
#    - Ensure the required JSON files (`arc-agi_test_challenges.json`) are located in the specified directory (`/kaggle/input/arc-prize-2024/`).
#    - The script automatically creates a `test` directory to store individual task JSON files.
#
# **🌟 Explore my profile and other public projects, and don't forget to share your feedback!**
#
# ## 👉 [Visit my Profile]( https://www.kaggle.com/code/zulqarnainalipk) 👈
#
#
# 2. **Submission Process**:
#    - Implement and refine the `predict_part_types` function to predict solutions based on input tasks.
#    - Utilize methods such as object detection (`get_objects_by_connectivity_`, `get_objects_by_color_`) to generate predictions.
#    - Ensure predictions are formatted correctly into a submission CSV (`sample_submission.csv`).
#
# 3. **Execution**:
#    - Execute the `main()` function to initiate the submission process.
#    - Review the generated `submission.json` file to verify the format and contents of your submission.
#
# ## Working Details
# - **Data Loading**: JSON files are loaded and split into individual task files for processing.
# - **Prediction**: Tasks are analyzed using functions like `predict_part_types` to generate predictions based on detected objects and their attributes.
# - **Submission**: Predictions are compiled into a CSV file (`old_submission.csv`) and translated into the required JSON format (`submission.json`) using the `translate_submission` function.
#
#
# ## Acknowledgments 🙏
# I acknowledge Abstraction and Reasoning Corpus organizers for providing the dataset and the competition platform.
#
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainalipk) 👈
#

# %%
# !cd

# %% [markdown]
# # Importing libraries & functions 📊

# %% [code] {"papermill": {"duration": 1.655086, "end_time": "2024-06-13T14:56:16.141729", "exception": false, "start_time": "2024-06-13T14:56:14.486643", "status": "completed"}, "tags": []}
# 📚 Importing necessary libraries 📊

import os  # For operating system related operations
import json  # For JSON manipulation
import csv  # For CSV file operations
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation
from pathlib import Path  # Pathlib for path operations
import sys

import pprint

pprint.pprint(sys.path)
# Check if running on Kaggle and adjust path if necessary
if os.path.exists('/kaggle/input'):

    sys.path.append('/kaggle/input/arclib4/')
else:
    sys.path.append('C:/Prj/ARC/input/arclib/')

pprint.pprint(sys.path)

# Importing specific functions and classes from arclib 📦
from arclib.dsl import *
from arclib.util import evaluate_predict_func, get_string, data_path
from arclib.dsl import Task, unique_arrays
from arclib.check import check_output_color_from_input


# %% [markdown]
# # 📂 Loading and processing JSON data 📄
#

# %% [markdown]
# ### Overview
#
# This script loads JSON content from a specified file, creates a directory named 'test' if it doesn't exist, and then splits the loaded JSON content into individual files. Each task from the JSON data is saved as a separate JSON file in the 'test' directory.
#
# ### Steps and Functionality
#
# 1. **Loading JSON Content**
#    - Opens and loads JSON data from the file specified by `json_file_path`.
#    - Stores the loaded JSON data into the `data` variable for further processing.
#
# 2. **Creating Output Directory**
#    - Defines `output_dir` as the path where individual JSON files will be stored (`'/kaggle/working/test'` by default).
#    - Uses `os.makedirs(output_dir, exist_ok=True)` to create the directory if it doesn't already exist (`exist_ok=True` ensures no errors if the directory already exists).
#
# 3. **Splitting JSON Content**
#    - Iterates through each `task_id` and `task_data` pair in the loaded JSON data (`data.items()`).
#    - Constructs the output file path for each task using `os.path.join(output_dir, f'{task_id}.json')`.
#    - Opens each output file in write mode (`'w'`) and writes the corresponding `task_data` into the file using `json.dump(task_data, output_file, indent=4)` for better readability.
#
# ### Usage
#
# - Ensure that `json_file_path` points to the correct location and filename of your JSON data (`arc-agi_test_challenges.json` in this case).
# - The script will create a directory named 'test' in the working directory (`/kaggle/working/`) and store each task from the JSON data as a separate JSON file within that directory.
# - This approach is useful for organizing and managing individual tasks or datasets that are originally stored in a single JSON file.
#
# ---
#

# %% [code] {"execution": {"iopub.execute_input": "2024-06-13T14:56:16.151456Z", "iopub.status.busy": "2024-06-13T14:56:16.150939Z", "iopub.status.idle": "2024-06-13T14:56:16.371466Z", "shell.execute_reply": "2024-06-13T14:56:16.370201Z"}, "papermill": {"duration": 0.228738, "end_time": "2024-06-13T14:56:16.374433", "exception": false, "start_time": "2024-06-13T14:56:16.145695", "status": "completed"}, "tags": []}

# Load the JSON content
json_file_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'  # 🌐 Replace with the actual path to your JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Create the 'test' directory
output_dir = '/kaggle/working/test'  # 📁 You can change this path as needed
os.makedirs(output_dir, exist_ok=True)

# Split the JSON content into individual files
for task_id, task_data in data.items():
    output_file_path = os.path.join(output_dir, f'{task_id}.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(task_data, output_file, indent=4)


# %% [markdown]
# # 📄 Load the JSON content
#

# %% [markdown]
# ### Overview
#
# This script reads a JSON file containing submission data, converts each entry into a specified CSV format, and writes it to a CSV file. The CSV file is structured with two columns: `output_id` and `output`, where `output` is derived from the JSON data after converting 2D matrices into a string format suitable for CSV.
#
# ### Steps and Functionality
#
# 1. **Reading JSON Data**
#    - The script starts by opening and loading data from the JSON file specified by `json_file_path`.
#
# 2. **Defining Output CSV File**
#    - Defines the output path for the CSV file where the converted data will be saved (`output_csv_path`).
#
# 3. **Converting 2D List to CSV String Format**
#    - Defines a function `convert_to_csv_format(matrix)` that converts a 2D list (matrix) into a string format suitable for CSV. It joins elements of each row with '|' and encloses them between '|' symbols.
#
# 4. **Writing to CSV File**
#    - Opens the `output_csv_path` in write mode (`'w'`) and initializes a `csv.DictWriter` instance to write data to the CSV file.
#    - Writes the header row (`'output_id', 'output'`) using `writer.writeheader()`.
#
# 5. **Processing Each Task**
#    - Iterates through each `task_id` and associated `attempts` in the loaded JSON data (`data.items()`).
#    - Checks if there is only one attempt (`len(attempts) == 1`). If true:
#      - Formats `output_id` as `{task_id}_0`.
#      - Converts `attempt['attempt_1']` and `attempt['attempt_2']` to CSV format using `convert_to_csv_format`.
#      - Combines both attempts into `combined_output` and writes to the CSV file.
#    - If there are multiple attempts, iterates through each attempt (`enumerate(attempts)`) and formats `output_id` accordingly.
#
# 6. **Completion Message**
#    - Prints a completion message (`"✅ Sample submission CSV file created at {output_csv_path}"`) indicating that the CSV file has been successfully created.
#
# ### Usage
#
# - Ensure that `json_file_path` points to the correct location and format of the input JSON file (`sample_submission.json` in this case).
# - After execution, `sample_submission.csv` will contain the processed submission data in the specified format suitable for submission.
#
# ---
#

# %% [code] {"execution": {"iopub.execute_input": "2024-06-13T14:56:16.383963Z", "iopub.status.busy": "2024-06-13T14:56:16.383566Z", "iopub.status.idle": "2024-06-13T14:56:16.404391Z", "shell.execute_reply": "2024-06-13T14:56:16.402072Z"}, "papermill": {"duration": 0.028758, "end_time": "2024-06-13T14:56:16.407393", "exception": false, "start_time": "2024-06-13T14:56:16.378635", "status": "completed"}, "tags": []}
json_file_path = '/kaggle/input/arc-prize-2024/sample_submission.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

# Define the output CSV file path
output_csv_path = '/kaggle/working/sample_submission.csv'

# Function to convert a 2D list into the required string format
def convert_to_csv_format(matrix):
    return '|' + '|'.join(''.join(map(str, row)) for row in matrix) + '|'

# Process and write to the CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['output_id', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for task_id, attempts in data.items():
        if len(attempts) == 1:
            attempt = attempts[0]
            output_id = f"{task_id}_0"
            attempt_1 = convert_to_csv_format(attempt['attempt_1'])
            attempt_2 = convert_to_csv_format(attempt['attempt_2'])
            combined_output = f"{attempt_1} {attempt_2}"
            writer.writerow({'output_id': output_id, 'output': combined_output})
        else:
            for i, attempt in enumerate(attempts):
                output_id = f"{task_id}_{i}"
                attempt_1 = convert_to_csv_format(attempt['attempt_1'])
                attempt_2 = convert_to_csv_format(attempt['attempt_2'])
                combined_output = f"{attempt_1} {attempt_2}"
                writer.writerow({'output_id': output_id, 'output': combined_output})

print(f"✅ Sample submission CSV file created at {output_csv_path}")


# %% [markdown]
# # 🔄 Function to translate csv to json
#

# %% [markdown]
#
# ### Overview
#
# This function converts an old submission format stored in a CSV file to a new JSON format required for the ARC Prize 2024 competition. It processes each line of the CSV, extracts relevant information, and structures it into a JSON format suitable for submission.
#
# ### Steps and Functionality
#
# 1. **Reading and Parsing CSV File**
#    - The function begins by opening and reading the CSV file specified by `file_path`.
#    - It reads all lines except the header line (`lines[1:]`) to skip the header information.
#
# 2. **Processing Each Line**
#    - For each line in the CSV file:
#      - Extract `output_id` and `output` by splitting the line on `,`.
#      - Split `output_id` into `task_id` and `output_idx` to identify the task and attempt index.
#      - Split `output` based on spaces `' '` to retrieve individual predictions.
#      - It considers only the first two predictions if more than two are present.
#      - Converts each prediction into a matrix format (`pred_matrix`) by parsing and mapping integers from '|' delimited string lines (`pred_lines`).
#
# 3. **Organizing Attempts**
#    - Structures each attempt (`attempt_dict`) as a dictionary with `"attempt_1"` and `"attempt_2"` keys representing matrices for the first and second attempts respectively.
#    - Inserts the attempt into `submission_dict` under the corresponding `task_id`. Inserts at the beginning (`insert(0, ...)`) if it's the first attempt (`output_idx == '0'`), otherwise appends it.
#
# 4. **Writing to JSON File**
#    - Finally, writes the structured `submission_dict` to a new JSON file named `'submission.json'`.
#    - Uses `json.dump()` with `indent=4` for readability, formatting the JSON with indentation.
#    
# 5. **Confirmation Message**
#    - Prints a confirmation message (`"✅ Translated submission saved as 'submission.json'"`) indicating successful completion of the translation process.
#
# ### Usage
#
# - Ensure the CSV file (`file_path`) contains the correct format as expected by the function.
# - After execution, `'submission.json'` will contain the transformed submission data ready for submission to the ARC Prize 2024 competition.
#
# ---
#
#

# %% [code] {"execution": {"iopub.execute_input": "2024-06-13T14:56:16.416592Z", "iopub.status.busy": "2024-06-13T14:56:16.416176Z", "iopub.status.idle": "2024-06-13T14:56:16.429063Z", "shell.execute_reply": "2024-06-13T14:56:16.427843Z"}, "papermill": {"duration": 0.020558, "end_time": "2024-06-13T14:56:16.431688", "exception": false, "start_time": "2024-06-13T14:56:16.41113", "status": "completed"}, "tags": []}
# 🔄 Function to translate from old submission format (csv) to new one (json)

def translate_submission(file_path):
    # Read the original submission file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # Skip the header line
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # Split predictions based on ' '
        
        # Take only the first two predictions
        if len(predictions) > 2:
            predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # Check if pred is not an empty string
                pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_1 = processed_predictions[0] if len(processed_predictions) > 0 else []  # Attempt 1 matrix
        attempt_2 = processed_predictions[1] if len(processed_predictions) > 1 else []  # Attempt 2 matrix

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        attempt_dict = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)  # Insert first attempt
        else:
            submission_dict[task_id].append(attempt_dict)  # Append subsequent attempts
    
    # Write to the new json file
    with open('submission.json', 'w') as file:
        json.dump(submission_dict, file, indent=4)  # Save JSON with indentation for readability

    print(f"✅ Translated submission saved as 'submission.json'")  # Confirmation message after successful translation



# %% [markdown]
# # Task Submission and Prediction Script🧩

# %% [markdown]
#
#
#
# #### 1. File Pointers and Paths
#
# ```python
# data_path = Path('/kaggle/working/')  # 📁 Path to working directory
# test_path = data_path / 'test'         # 📁 Path to test directory
# sample_submission_path = '/kaggle/working/sample_submission.csv'  # 📄 Path to sample submission CSV
# ```
#
# - **Explanation**: 
#   - `data_path` and `test_path`: Define the directories where input data (tasks) and test files are stored respectively.
#   - `sample_submission_path`: Points to the sample submission CSV file where predictions will be formatted and written.
#
# #### 2. Checking Output Against Candidates
#
# ```python
# def check_output_in_candidates(output, candidates):
#     """
#     Checks if the given output matches any of the candidate outputs.
#     """
#     output_is_candidate = False
#
#     for candidate in candidates:
#         if output.shape == candidate.shape:
#             if (output == candidate).all():
#                 output_is_candidate = True
#                 break
#     return output_is_candidate
# ```
#
# - **Explanation**: 
#   - `check_output_in_candidates`: Function to verify if a predicted output matches any of the candidate solutions. This is crucial for validation during prediction.
#
# #### 3. Prediction Functions
#
# ```python
# def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):
#     """
#     Predicts solutions for tasks based on input-output pairs using candidate generation.
#     """
#     # Implementation details as explained in the code above...
# ```
#
# - **Explanation**: 
#   - `predict_part`: Core function for predicting solutions. It iterates through input-output pairs, generates candidates, and verifies against expected outputs using `check_output_in_candidates`.
#
# #### 4. Object Manipulation Functions
#
# ```python
# def get_cropped_object(array, object_map, bg_color=None):
#     """
#     Retrieves a cropped portion of the array based on the given object map.
#     """
#     # Implementation details as explained in the code above...
# ```
#
# - **Explanation**: 
#   - Functions like `get_cropped_object`, `keep_one_object`, `get_cropped_objects`, and `get_inputs_with_one_object` are utility functions for manipulating arrays to extract objects based on provided object maps.
#
# #### 5. Prediction Types and Strategies
#
# ```python
# def predict_part_types(task):
#     """
#     Predicts types of solutions for tasks based on input characteristics and candidate generation.
#     """
#     # Implementation details as explained in the code above...
# ```
#
# - **Explanation**: 
#   - `predict_part_types`: Uses different strategies (`get_object_map_funcs`) to predict types of solutions based on input characteristics and candidate generation methods.
#
# #### 6. Submission and Result Handling
#
# ```python
# def submit(predict):
#     """
#     Submits predictions for test tasks and creates a new submission CSV file.
#     """
#     # Implementation details as explained in the code above...
# ```
#
# - **Explanation**: 
#   - `submit`: Reads test tasks, predicts outputs using `predict`, formats them according to the competition requirements, and outputs them to `old_submission.csv`, which is then translated into the final submission format using `translate_submission`.
#
# ### Execution Flow
#
# - **Main Execution**: The `main()` function is called when the script runs, which in turn calls `submit(predict_part_types)`.
# - **Submission Process**: 
#   - Reads the sample submission CSV.
#   - Iterates through test files, predicts outputs using `predict_part_types`.
#   - Formats predictions and saves them to `old_submission.csv`.
#   - Translates the CSV submission to the required JSON format for ARC Prize 2024.
#
#
#

# %% [code] {"execution": {"iopub.execute_input": "2024-06-13T14:56:16.441016Z", "iopub.status.busy": "2024-06-13T14:56:16.440555Z", "iopub.status.idle": "2024-06-13T14:56:54.530407Z", "shell.execute_reply": "2024-06-13T14:56:54.52903Z"}, "papermill": {"duration": 38.097796, "end_time": "2024-06-13T14:56:54.533141", "exception": false, "start_time": "2024-06-13T14:56:16.435345", "status": "completed"}, "tags": []}
# 📂 Add pointers to files and sample submission CSV created above from ARC Prize JSONs

data_path = Path('/kaggle/working/')  # 📁 Path to working directory
test_path = data_path / 'test'  # 📁 Path to test directory

sample_submission_path = '/kaggle/working/sample_submission.csv'  # 📄 Path to sample submission CSV

# Function to check if output matches any candidate
def check_output_in_candidates(output, candidates):
    """
    Checks if the given output matches any of the candidate outputs.

    Args:
    - output (numpy.ndarray): The output matrix to compare.
    - candidates (list of numpy.ndarray): List of candidate matrices to compare against.

    Returns:
    - output_is_candidate (bool): True if output matches any candidate, False otherwise.
    """
    output_is_candidate = False

    for candidate in candidates:
        if output.shape == candidate.shape:
            if (output == candidate).all():
                output_is_candidate = True
                break
    return output_is_candidate


# Function to predict based on task inputs and candidates
def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):
    """
    Predicts solutions for tasks based on input-output pairs using candidate generation.

    Args:
    - task (Task): Task object containing input-output pairs and test inputs.
    - get_candidates (function): Function to generate candidates based on input and object maps.
    - train_object_maps (list of numpy.ndarray, optional): Object maps for training inputs.
    - train_bg_colors (list, optional): Background colors for training inputs.

    Returns:
    - all_input_predictions (list): List of predictions for each test input if the task is fully solvable, otherwise an empty list.
    """
    part_task = True
    for i, (input, output) in enumerate(task.pairs):
        input = np.array(input)
        output = np.array(output)

        candidates = get_candidates(input, object_maps=train_object_maps[i], bg_color=train_bg_colors[i])

        if candidates:
            if not check_output_in_candidates(output, candidates):
                part_task = False
                break
        else:
            part_task = False
            break

    all_input_predictions = []
    if part_task:
        for input in task.test_inputs:
            test_candidates = get_candidates(input)
            predictions = test_candidates
            predictions = unique_arrays(predictions)
            predictions = sorted(predictions, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            all_input_predictions.append(predictions)

    else:
        all_input_predictions = []

    return all_input_predictions


# Function to get cropped object based on object map
def get_cropped_object(array, object_map, bg_color=None):
    """
    Retrieves a cropped portion of the array based on the given object map.

    Args:
    - array (numpy.ndarray): The input array from which to retrieve the cropped object.
    - object_map (numpy.ndarray): Binary map indicating the location of the object.
    - bg_color (int or float, optional): Background color to fill outside the object boundaries.

    Returns:
    - cropped_object (numpy.ndarray): Cropped object from the input array.
    """
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    return array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]


# Function to keep one object based on object map
def keep_one_object(array, object_map, bg_color=None):
    """
    Extracts and retains only one object from the input array based on the given object map.

    Args:
    - array (numpy.ndarray): The input array containing multiple objects.
    - object_map (numpy.ndarray): Binary map indicating the location of the object to keep.
    - bg_color (int or float, optional): Background color to fill outside the object boundaries.

    Returns:
    - retained_object (numpy.ndarray): Array with only the specified object retained, with background filled if specified.
    """
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    if bg_color is None:
        bg_color = detect_bg_(array)
    output_ = np.full_like(array, bg_color)
    output_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]
    return output_


# Function to get cropped objects based on object maps
def get_cropped_objects(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    """
    Retrieves cropped objects from the input array based on the detected object maps.

    Args:
    - array (numpy.ndarray): The input array from which to retrieve objects.
    - get_object_maps (function, optional): Function to detect object maps in the array.
    - object_maps (list of numpy.ndarray, optional): Pre-detected object maps.
    - augment (function, optional): Function to apply augmentation to cropped objects.
    - bg_color (int or float, optional): Background color to fill outside the object boundaries.

    Returns:
    - objects (list of numpy.ndarray): List of cropped objects extracted from the input array.
    """
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(get_cropped_object(array, object_map)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


# Function to get inputs with one object based on object maps
def get_inputs_with_one_object(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    """
    Retrieves inputs containing exactly one object based on detected object maps.

    Args:
    - array (numpy.ndarray): The input array from which to retrieve inputs with one object.
    - get_object_maps (function, optional): Function to detect object maps in the array.
    - object_maps (list of numpy.ndarray, optional): Pre-detected object maps.
    - augment (function, optional): Function to apply augmentation to retained objects.
    - bg_color (int or float, optional): Background color to fill outside the object boundaries.

    Returns:
    - objects (list of numpy.ndarray): List of inputs with exactly one object retained.
    """
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(keep_one_object(array, object_map, bg_color=bg_color)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


# List of functions to get object maps
get_object_map_funcs = [get_objects_by_connectivity_, partial(get_objects_by_connectivity_, touch='corner'),
                        get_objects_by_color_and_connectivity_, partial(get_objects_by_color_and_connectivity_, touch='corner'), get_objects_by_color_,
                        get_objects_rectangles, partial(get_objects_rectangles, direction='horisontal'), get_objects_rectangles_without_noise, get_objects_rectangles_without_noise_without_padding]


# Function to predict part types based on task inputs
def predict_part_types(task):
    """
    Predicts types of solutions for tasks based on input characteristics and candidate generation.

    Args:
    - task (Task): Task object containing inputs to predict solutions for.

    Returns:
    - predictions (list): List of predicted solutions for each input if solvable, otherwise an empty list.
    """
    predictions = []

    if check_output_color_from_input(task):
        bg_colors = [detect_bg_(input_) for input_ in task.inputs]
        for i, get_object_maps in enumerate(get_object_map_funcs):

            object_maps_list = [get_object_maps(input_, bg_color=bg_color) for input_, bg_color in zip(task.inputs, bg_colors)]
            for get_object_func in (get_cropped_objects, get_inputs_with_one_object):
                for augment in simple_output_process_options:
                    get_candidates = partial(get_object_func, get_object_maps=get_object_maps, augment=augment)
                    predictions = predict_part(task, get_candidates=get_candidates, train_object_maps=object_maps_list, train_bg_colors=bg_colors)
                    if predictions:
                        break
                if predictions:
                    break
            if predictions:
                break

    return predictions


# Function to submit predictions and create a new submission CSV file
def submit(predict):
    """
    Submits predictions for test tasks and creates a new submission CSV file.

    Args:
    - predict (function): Function to predict solutions for tasks.

    Returns:
    - None
    """
    submission = pd.read_csv(sample_submission_path, index_col='output_id')
    submission['output'] = ''
    test_fns = sorted(os.listdir(test_path))
    count = 0
    for fn in test_fns:
        fp = test_path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f))
            all_input_preds = predict(task)
            if all_input_preds:
                print(fn)
                count += 1

            for i, preds in enumerate(all_input_preds):
                output_id = str(fn.split('.')[-2]) + '_' + str(i)
                string_preds = [get_string(pred) for pred in preds[:3]]
                pred = ' '.join(string_preds)
                submission.loc[output_id, 'output'] = pred

    print(count)
    submission.to_csv('old_submission.csv')  # Save submission to CSV
    # Create a submission file that follows the ARC Prize 2024 guidelines
    translate_submission('/kaggle/working/old_submission.csv')



# %% [code] {"papermill": {"duration": 0.004121, "end_time": "2024-06-13T14:56:54.541778", "exception": false, "start_time": "2024-06-13T14:56:54.537657", "status": "completed"}, "tags": []}
# Main function to execute submission process
def main():
    submit(predict_part_types)


if __name__ == '__main__':
    main()


# %% [markdown]
# ## 🌟 Keep Exploring! 🌟
#
# Thanks a bunch for diving into this notebook! If you had a blast or learned something new, why not dive into more of my captivating projects and contributions on my profile?
#
# 👉 [Let's Explore More!](https://www.kaggle.com/zulqarnainalipk) 👈
#
# [GitHub](https://github.com/zulqarnainalipk) |
# [LinkedIn](https://www.linkedin.com/in/zulqarnainalipk/)
#
# ## 💬 Share Your Thoughts! 💡
#
# Your feedback is like treasure to us! Your brilliant ideas and insights fuel our ongoing improvement. Got something to say, ask, or suggest? Don't hold back!
#
# 📬 Drop me a line via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
#
# Huge thanks for your time and engagement. Your support is like rocket fuel propelling me to create even more epic content.
# Keep coding joyfully and wishing you stellar success in your data science adventures! 🚀
#
