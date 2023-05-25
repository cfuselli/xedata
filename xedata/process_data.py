import numpy as np
import sys
from argparse import ArgumentParser
from pathlib import Path

from xedata.defaults import MY_PATH, XEDATA_PATH , OUTPUT_FOLDER

"""
This file is called in submitter.py with:

        python {process_data_file} \
            --index {i} \
            --n_per_job {n_per_job} \
            --mode {mode} \
            --label {label} \
            --targets {targets} \
            --runs_filename {runs_filename}
"""

def parse_args():


    parser = ArgumentParser()
    parser.add_argument('-i', '--index', type=int, help='index of the current job')
    parser.add_argument('-n', '--n_per_job', '--n-per-job', type=int, help='how many runs per job')
    parser.add_argument('-l', '--label', type=str, help='Name for the job batch, like rn220_events')
    parser.add_argument('-m', '--mode', type=str, help='processing mode')
    parser.add_argument('--targets', '-t', nargs='*', help="Strax data type name(s) that should be produced with live processing.")
    parser.add_argument('-r', '--runs-filename', '--runs_filename', type=str, help='file containing the list of run IDs')
    parser.add_argument('-c', '--context', default='xenonnt_v8', type=str, help='cutax context to use. For example, ')
    
    args = parser.parse_args()

    return args



def get_context(context):

    import strax
    import straxen
    import cutax

    straxen.print_versions()
    st = getattr(cutax.contexts, context)(output_folder=OUTPUT_FOLDER)

    import extra_plugins
    st.register_all(extra_plugins)

    print(st._plugin_class_registry)
    
    return st

def process_data(
            index ,
            n_per_job,
            label,
            mode,
            targets,
            run_ids,
            context
            ):

    import strax
    import straxen
    import cutax
    import time

    print('Started processing')
    print('Total runs:', len(run_ids))
    print(run_ids)

    st = get_context(context)

    # Add your data processing logic here

    exceptions = False

    for run_id in run_ids:
        for t in targets:
            print('\n\nBuilidng' + run_id + '  ' + t)
            start_time = time.time()
            try:
                st.make(run_id,t,save=t)
                print(t, '%.2f' % (time.time() - start_time))
            except Exception as e:
                failures = True
                print('Exception: ', e)


    print('\n\nFinished')
    
    if failures:
        raise
    else:
        return

def save_data(
            index ,
            n_per_job,
            label,
            mode,
            targets,
            run_ids,
            context
            ):

    import strax
    import straxen
    import cutax
    import time
    
    print('\n\nLoading...')            
    start_time = time.time()

    st = get_context(context)

    df = st.get_array(run_ids, 
                save_targets, 
                add_run_id_field=True,
                selection_str=selection_str,
                )

    time.sleep(2)

    with open(f'{XEDATA_PATH}/hdf5/{label}-{i}.npy', 'wb') as f:
        np.save(f, df)

    print('\nLoaded and saved!')
    print('%.2f' % (time.time() - start_time))


def main():
 

    args = parse_args()

    index = args.index
    n_per_job = args.n_per_job
    label = args.label
    mode = args.mode
    targets = args.targets
    runs_filename = args.runs_filename
    context = args.context

    with open(runs_filename) as file:
        run_ids = file.readlines()
        run_ids = [line.rstrip() for line in run_ids]
    run_ids = run_ids[index * n_per_job: (index + 1) * n_per_job]

    if mode == 'process':
        process_data(    
            index = index,
            n_per_job = n_per_job,
            label = label,
            mode = mode,
            targets = targets,
            run_ids = run_ids,
            context=context
        )


    elif mode == 'save':
        save_data(
            index = index,
            n_per_job = n_per_job,
            label = label,
            mode = mode,
            targets = targets,
            run_ids = run_ids,
            context=context
        )

if __name__ == "__main__":
    main()