## xedata - Documentation

This package provides tools for processing, saving, merging, and performing various operations on data. It supports batch job submission using SLURM.

### Installation

To install the package in user-develop mode, follow these steps:

1. Clone the repository (in /dali or /project2):

   ```shell
   cd /dali/$USER/
   git clone https://github.com/cfuselli/xedata.git
   ```

2. Navigate to the repository directory:

   ```shell
   cd xedata
   ```

3. Install the package:

   ```shell
   pip install --user -e .
   ```

### Usage

The package provides the following modes of operation:

- `process`: Processes the runs with the specified targets in `process_data.py`.
- `save`: Saves data to dataframes.
- `merge`: Merges data from multiple sources.
- `doall`: Performs all the operations above.

To run the package, use the following command:

```shell
xedata <mode> <label> [options]
```

#### Available Options

- `mode`: Mode of operation. Choose one of the following:
  - `process`: Processes the runs with the specified targets in `process_data.py`.
  - `save`: Saves data to dataframes.
  - `merge`: Merges data from multiple sources.
  - `doall`: Performs all the operations above.

- `label`: Name of the sbatch submission.

- `--targets`, `-t`: Strax data type name(s) that should be produced with live processing. Multiple targets can be specified by separating them with spaces. Default: `event_info`.

- `--n_per_job`, `-n`: Number of runs per job. Default: 100.

- `--runs`, `-r`: File (txt) in `/runs_selection` to source (without extension). Default: `nton_official_sr0_none`.

- `--mem_per_cpu`, `-m`: Memory per CPU in megabytes. Default: 10000.

- `--container`, `-c`: Versioned container to use. Default: `2022.06.3`.

- `--context`, `-ct`: Cutax context to use. Default: `xenonnt_v8`.

- `--partition`, `-p`: SLURM partition to use. See `utilix.batchq` for available options. Default: `dali`.

- `--qos`, `-q`: SLURM quality of service to use. See `utilix.batchq` for available options. Default: `dali`.

- `--selection_str`, `-s`: Selection string to load data. See `strax.get_array` for more information. Default: empty.

### Before Running

Before running the command, make sure to modify the paths and configurations according to your setup. Adjust the paths and options in the `submitter.py` file to match your environment.

### Examples

1. Process runs with live processing:

   ```shell
   xedata process bkg_events --runs nton_official_sr0_none --target event_info --n_per_job 50
   ```

   This command processes the runs with the `event_info` and `event_basics` targets, with 50 runs per job.

2. Save data to dataframes:

   ```shell
   xedata save bkg_events --runs nton_official_sr0_none --target event_info --n_per_job 50 --container 2022.06.3
   ```

   This command saves the data to dataframes using the specified runs and container.

3. Merge data from multiple sources:

   ```shell
   xedata merge bkg_events --runs nton_official_sr0_none --target event_info --n_per_job 50 --selection_str 'time>0'
   ```

   This command merges data from two sources, filtering based on the provided selection string.

4. Perform all operations:

   ```shell
   xedata doall bkg_events
   ```

   This command performs all operations: processing, saving, and merging.

```
```

Please note that the above documentation assumes that you have set up your environment and dependencies correctly. Adjustments may be needed based on your specific configuration.