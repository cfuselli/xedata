
# xedata

xedata is a package for processing and saving XENONnT data.

## Installation

To install xedata, you can use pip with the user develop option:

```shell
pip install --user -e .
```

This will install the package in development mode, allowing you to make changes to the code without reinstalling.

Note: Make sure you have the required dependencies installed before proceeding with the installation.

## Usage

After installing xedata, you can use the `xedata` command to process and save XENONnT data. Here are the available options:

```shell
xedata process -n <runs_per_job> -o <output_folder> -f <runs_file> -m <mode> -s <source>
```

- `-n, --num_jobs`: The number of runs per job.
- `-o, --output_folder`: The output folder for the processed data.
- `-f, --runs_file`: The file containing the list of run IDs.
- `-m, --mode`: The processing mode.
- `-s, --source`: The source for processing.

Before running the command, make sure to modify the paths accordingly in the configuration files or as command-line arguments.

## Examples

Here are some examples of how to use xedata:

- Process data using 10 parallel jobs:
  ```shell
  xedata process -n 10 -o /path/to/output -f runs.txt -m mode1 -s source1
  ```

- Process data in a different mode and source:
  ```shell
  xedata process -n 5 -o /path/to/output -f runs.txt -m mode2 -s source2
  ```

Please refer to the documentation for more details on the available processing modes and sources.

## Contributing

Contributions to xedata are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
```

Feel free to modify and customize this README to fit your specific needs.