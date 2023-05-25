from argparse import ArgumentParser
import os
import getopt
import time
import numpy as np
from itertools import zip_longest
import os, glob

from xedata.defaults import MY_PATH, XEDATA_PATH , OUTPUT_FOLDER

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=['process', 'save', 'merge', 'doall'],
                        help='process - will process the runs with the specified targets in process_data.py'
                             'is_stored - check if targets are stored '
                             'save - saves data to dataframes')

    parser.add_argument('label', type=str,
                        help='Name of sbatch submission. Example: rn220_events')

    parser.add_argument('--targets', '-t', nargs='*', default='event_info'.split(),
        help="Strax data type name(s) that should be produced with live processing. Separated by a space")

    parser.add_argument('--n_per_job','-n', type=int, default=35,
                        help='how many runs per job')

    parser.add_argument('--runs', '-r', type=str, default='runfile_test',
                        help='file (txt) in /runs_selection to source (without extension)')

    parser.add_argument('--mem_per_cpu', '-m', type=int, default=10000,
                        help='mb per cpu')

    parser.add_argument('--container', '-c', type=str, default='2022.06.3',
                        help='Versioned container - default for SR0')
    
    parser.add_argument('-ct', '--context', default='xenonnt_v8', type=str, 
                        help='cutax context to use. For example, ')

    parser.add_argument('--partition', '-p', type=str, default='dali',
                        help='partition, see utilix.batchq')

    parser.add_argument('--qos', '-q', type=str, default='dali',
                        help='qos, see utilix.batchq')

    parser.add_argument('--selection_str', '-s', type=str, default='',
                        help='selection str to load data, see strax.get_array')

    args = parser.parse_args()

    return args

def main():


    welcome = """
    
    Thanks for using XEDATA 
        - the tool to load and save XENONnT data

    Carlo Fuselli (cfuselli@nikhef.nl)
    
    """
    print(welcome)

    args = parse_args()

    mode = args.mode
    label = args.label
    targets = args.targets
    n_per_job = args.n_per_job
    runs = args.runs
    mem_per_cpu = args.mem_per_cpu
    container = args.container
    context = args.context
    partition = args.partition
    qos = args.qos

    if mode == 'merge':
        n_per_job = 999999


    if (mode == 'process') | (mode == 'save') | (mode == 'merge'):
        submit(
        mode=mode, 
        label=label, 
        n_per_job=n_per_job,
        targets=targets, 
        runs=runs, 
        mem_per_cpu=mem_per_cpu, 
        container=container, 
        context=context,
        partition=partition, 
        qos=qos
        )
    
    elif mode == 'doall':
        submitted_jobs = submit(
        mode='process', 
        label=label, 
        n_per_job=n_per_job,
        targets=targets, 
        runs=runs, 
        mem_per_cpu=mem_per_cpu, 
        container=container, 
        partition=partition, 
        context=context,
        qos=qos)
        
        if not None in submitted_jobs:
            save_jobs = submit(
                mode='save',
                label=label, 
                n_per_job=int(n_per_job*5),
                targets=targets, 
                runs=runs, 
                mem_per_cpu=mem_per_cpu,  
                container=container, 
                partition=partition, 
                qos=qos,
                context=context,
                dependency=submitted_jobs
                )

        if (not None in save_jobs):
            merge_jobs = submit(
                mode='merge',
                label=label, 
                n_per_job=9999999,
                targets=targets, 
                runs=runs, 
                mem_per_cpu=int(mem_per_cpu*2),  
                container=container, 
                partition=partition, 
                qos=qos,
                context=context,
                dependency=save_jobs
                )

        else:
            print("Something failed in the submission of the first batch of jobs")

    return


def submit(mode, n_per_job, targets, runs, mem_per_cpu, container, partition, qos, label, context, **kwargs):

    from utilix.batchq import submit_job

    container_file = f'xenonnt-{container}.simg'
    output_folder = OUTPUT_FOLDER

    runs_filename = os.path.join(XEDATA_PATH, f'run_selection/{runs}.txt')
    log_dir =       os.path.join(XEDATA_PATH,'logs/')
    process_data_file =        os.path.join(XEDATA_PATH,'xedata/process_data.py')

    with open(runs_filename) as file:
        run_ids = file.readlines()
        run_ids = [line.rstrip() for line in run_ids]

    if mode == 'save':
        for f in glob.glob(os.path.join(XEDATA_PATH, f"dataframes_tmp/{label}-*.npy")):
            os.remove(f)


    total_n_jobs = len(list(zip_longest(*(iter(run_ids),) * n_per_job)))

    status = f"""
-------------------------------------------------------------
    - Mode: {mode}
    - Submitting {total_n_jobs} jobs
    - Container: {container}
    - Output Folder: {output_folder}
    - Runs Filename: {runs_filename}
    - PythonFile: {process_data_file}
    - Logs: {XEDATA_PATH}/logs
-------------------------------------------------------------

    """
    print(status)

    submitted_jobs = []

    print('Preparing to process:')
    for i in range(total_n_jobs):
        log = os.path.join(log_dir, f'log_{mode}_{label}_{i}.sh')
        jobname = f'{mode}_{label}_{i}'

        # set -e is necessary to tell bash to raise errors, fundamental for dependencies
        # TODO: move this command to utilix template
        jobstring = f"""
set -e

echo "Starting process_data"
echo `date`
echo "{i} {n_per_job} {output_folder}"

python {process_data_file} \
    --index {i} \
    --n_per_job {n_per_job} \
    --mode {mode} \
    --label {label} \
    --context {context} \
    --targets {" ".join(targets)} \
    --runs_filename {runs_filename} \

"""


        print('Submitting %i/%i' % (i, total_n_jobs-1), jobname)
        # Utilix function to submit jobs
        job_id = submit_job(jobstring,
                          log=log,
                          jobname=jobname,
                          mem_per_cpu=mem_per_cpu,
                          container=container_file,
                          partition=partition,
                          qos=qos,
                          exclude_nodes='dali001,dali003',
                          **kwargs
                          )
        
        submitted_jobs.append(job_id)


    print('Finished')

    return submitted_jobs


if __name__ == "__main__":
    main()
