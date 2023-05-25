from argparse import ArgumentParser
import os
import getopt
import time
import numpy as np
from itertools import zip_longest
import os, glob

from defaults import MY_PATH, XEDATA_PATH , OUTPUT_FOLDER

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=['process', 'is_stored', 'save', 'save-dep'],
                        help='process - will process the runs with the specified targets in process_data.py'
                             'is_stored - check if targets are stored '
                             'save - saves data to dataframes')
    parser.add_argument('--n_per_job','-n', type=int, default=20,
                        help='how many runs per job')
    parser.add_argument('--runs', '-r', type=str, default='none',
                        help='file (txt) in /runs_selection to source (without extension)')
    parser.add_argument('--mem_per_cpu', '-m', type=int, default=10000,
                        help='mb per cpu')
    parser.add_argument('--container', '-c', type=str, default='2022.06.3',
                        help='Versioned container - default for SR0')
    parser.add_argument('--partition', '-p', type=str, default='dali',
                        help='partition, see utilix.batchq')
    parser.add_argument('--qos', '-q', type=str, default='dali',
                        help='qos, see utilix.batchq')
    
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
    n_per_job = args.n_per_job
    runs = args.runs
    mem_per_cpu = args.mem_per_cpu
    container = args.container
    partition = args.partition
    qos = args.qos

    if (mode == 'process') | (mode == 'save'):
        submit_jobs(
            mode=mode,
            n_per_job=n_per_job,
            runs=runs,
            mem_per_cpu=mem_per_cpu, 
            container=container,
            qos=qos,
            partition=partition
            )
    
    elif mode == 'save-dep':
        submitted_jobs = submit_jobs(
            mode='process',
            n_per_job=n_per_job,
            runs=runs,
            mem_per_cpu=mem_per_cpu, 
            container=container,
            qos=qos,
            partition=partition
            )
        
        if not None in submitted_jobs:

            dependency_jobs = submit_jobs(
                mode='save',
                n_per_job=int(n_per_job*20),
                runs=runs,
                mem_per_cpu=int(mem_per_cpu/2), 
                container=container,
                qos=qos,
                partition=partition,
                dependency=submitted_jobs
                )

        else:
            print("Something failed in the submission of the first batch of jobs")

    return

def submit_jobs(mode, n_per_job, runs, mem_per_cpu, container, partition, qos, **kwargs):

    from utilix import batchq

    container_file = f'xenonnt-{container}.simg'
    output_folder = OUTPUT_FOLDER

    runs_filename = os.path.join(XEDATA_PATH, f'run_selection/{runs}.txt')
    log_dir =       os.path.join(XEDATA_PATH,'logs/')
    pyfile =        os.path.join(XEDATA_PATH,'xedata/process_data.py')

    with open(runs_filename) as file:
        run_ids = file.readlines()
        run_ids = [line.rstrip() for line in run_ids]

    if mode == 'save':
        for f in glob.glob(os.path.join(XEDATA_PATH, f"dataframes_tmp/*{ #TODO }*.npy")):
            os.remove(f)


    list_of_groups = list(zip_longest(*(iter(run_ids),) * n_per_job))

    status = f"""
-------------------------------------------------------------
    - Mode: {mode}
    - Submitting {len(list_of_groups)} jobs
    - Container: {container}
    - Output Folder: {output_folder}
    - Runs Filename: {runs_filename}
    - PythonFile: {pyfile}
    - Logs: {XEDATA_PATH}/logs
-------------------------------------------------------------

    """
    print(status)

    submitted_jobs = []

    print('Preparing to process:')
    for i in range(len(list_of_groups)):
        log = log_dir + ('log_'+mode+'_'+{runs}+'_%i.sh' % i)
        jobname = ('job_'+mode+'_'+runs+'_%i' % i)

        jobstring = f"""
        echo "Starting process_data"
        echo "{i} {n_per_job} {output_folder}"
        python {pyfile} -i {i} -n {n_per_job} -r {runs_filename} -m {mode} -r {runs_filename}
        echo "Script complete, bye byeee!"
        echo `date`
        """

        print('Submitting %i/%i' % (i, len(list_of_groups)-1), jobname)
        job_id = batchq.submit_job(jobstring,
                          log=log,
                          jobname=jobname,
                          mem_per_cpu=mem_per_cpu,
                          container=container_file,
                          partition=partition,
                          qos=qos,
                          **kwargs
                          )
        
        submitted_jobs.append(job_id)


    print('Finished')

    return submitted_jobs


if __name__ == "__main__":
    main()
