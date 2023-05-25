# xedata

xedata is a Python package designed for processing and saving XENONnT data. It provides a command-line interface (CLI) for convenient data processing and offers various options for customization. This document serves as a guide to installing xedata, understanding its functionalities, and using it effectively.

## Installation

To install xedata, follow these steps:

1. Clone the xedata repository 
(Please note: clone the repository in a directory where you have space, like on /dali or /project2, not in /home !):

   ```shell
   git clone https://github.com/cfuselli/xedata.git
   ```

2. Navigate to the cloned directory:

   ```shell
   cd xedata
   ```

3. Install the package using pip with the user develop option:

   ```shell
   pip install --user -e .
   ```

   This will install xedata in development mode, allowing you to make changes to the code without reinstalling.

   Note: Make sure you have the required dependencies installed before proceeding with the installation.

## Usage

The xedata package provides a CLI with multiple commands to process and manage XENONnT data. Here's an overview of the available commands:

### `xedata process`

The `process` command is used to process XENONnT data. It performs data processing operations based on the specified parameters.

```shell
xedata process [options]
```

Available options:

- `-n, --num_jobs`: The number of parallel jobs to run. Default is 1.
- `-o, --output_folder`: The output folder for the processed data. Default is "./output".
- `-f, --runs_file`: The file containing the list of run IDs. Default is "./runs.txt".
- `-m, --mode`: The processing mode. Default is "mode1".
- `-s, --source`: The data source. Default is "source1".

The processing mode and data source can be selected based on your specific requirements. Please refer to the documentation or the source code for detailed information about available modes and sources.



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