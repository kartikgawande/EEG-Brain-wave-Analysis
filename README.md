# Conflicting-Competency

This branch is contributed by **Kartik Satish Gawande (MT2023045)**.

## Directory Details

### **classification/16th to 30th participant/**
This directory contains earlier classification attempts using **Random Forest** and **KNN** on data from the 16th to 30th participants. These attempts were conducted:
- **Participant Agnostically** 
- **Participant Non-Agnostically**

**Key File:**
- `Results.jpg` - Contains the results of the training.

---

### **pre-processing/Kartik/Video timestamps/**
This directory includes Python scripts used to output the finalized training timestamps.

**Key Files:**
- `Synced_Times_All.csv` - Contains all the offsets for each participant's video timestamps to synchronize them. 
- Other intermediate files are also present, which were used to derive the final synced file.

---

#### **pre-processing/Kartik/Video timestamps/stamps/**
This subdirectory contains all the interaction timestamps for each participant. These timestamps were:
- **Semi-automatically extracted** using `script.py` from the `pre-processing/Kartik/Video timestamps` directory.

---

### **pre-processing/Kartik/**
This directory contains Python notebooks that were used to calculate the **start** and **end timestamps** for each activity type and file for each participant. The activity types are:
- **1, 2, and 3** (representing different activity types)
- Files are specified as **1, 2, 3, and 4** (representing the first, second, third, and fourth files respectively).

**Key File:**
- `exp_timestamps_refined.csv` - The final file containing timestamps synced with dataset timestamps. These are the **final training timestamps**.

---

**Note:** All timestamps in `exp_timestamps_refined.csv` are synced with the dataset timestamps.
### **kedro/**
This directory has the entire codebase (including from all the other directories) shifted to kedro framework for better handling of the workflow. Please use this folder using kedro. 

```bash
pip install kedro
```

## Kedro Project Structure Overview

### Directories and Their Contents

- **`notebooks`**: Contains all the scripts and Jupyter notebooks used across different parts of the project.
- **`src/kartik_egg/pipelines`**: Each pipeline directory has two main files:
  - `nodes.py`: Contains all functions for data manipulation and model training.
  - `pipeline.py`: Specifies the execution order of nodes and the data flow (inputs and outputs) between them.
    - Note: Many pipeline variables are instantiated here, but only one can be used at a time when running a pipeline.

### Running a Pipeline

To run a specific pipeline, use the following command:

```bash
kedro run --pipeline=<pipeline_name>
```

### Configuration Files in `conf/base/`
- The directory contains all the configurations used by the nodes during runtime.
  - **`parameters_training.yml`**: Directly specify model parameters like the number of estimators for the RandomForest model.
  - **`parameters_preprocessing.yml`**: Describes how files in the dataset are labeled and other preprocessing parameters.

### Data Directory Structure in `data/`
- Well-organized according to their role in the pipeline workflows:
  - **`00_scattered`**: Initial dataset files that are totally scattered.
  - **`01_raw`**: Dataset files appended into a single file.
  - **`02_intermediate/video_data`**: Intermediate data that cannot be directly used in training.
  - **`03_feature`**: Contains final training stamps with mentioned label columns.
  - **`04_primary`**: Holds the training-ready dataset.
  - **`05_model_input`**: Training and testing split files.
  - **`06_models`**: Saved models.
  - **`08_reporting`**: Necessary reports from each trained model in JSON format.

### Additional Resources
- For a better understanding of Kedro, please watch this 5-minute video:
  [![Watch the video](https://img.youtube.com/vi/PdNkECqvI58/0.jpg)](https://youtu.be/PdNkECqvI58?si=7OmYVW3vyYQCI0qk)

