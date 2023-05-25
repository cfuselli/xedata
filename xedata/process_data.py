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
    parser.add_argument('-s', '--selection_str', default='', type=str, help='load data selection string, see strax and straxen get_array')
    
    args = parser.parse_args()

    return args


def main():
 

    args = parse_args()

    index = args.index
    n_per_job = args.n_per_job
    label = args.label
    mode = args.mode
    targets = args.targets
    runs_filename = args.runs_filename
    context = args.context
    selection_str = args.selection_str

    print(f"Running in mode: {mode}" )

    with open(runs_filename) as file:
        _run_ids = file.readlines()
        _run_ids = [line.rstrip() for line in _run_ids]
    run_ids = _run_ids[index * n_per_job: (index + 1) * n_per_job]

    only_one_job = ( len(run_ids) == len(_run_ids) )

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
            context=context,
            selection_str=selection_str,
            only_one_job=only_one_job,
        )
        

    elif mode == 'merge':

        merge_data(
            index = index,
            n_per_job = n_per_job,
            label = label,
            mode = mode,
            targets = targets,
            run_ids = run_ids,
            context=context
        )
    
    else:

        print("How did you arrive here? Something is off.")

        


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

    failures = False

    for run_id in run_ids:
        for t in targets:
            print('\n\nBuilidng' + run_id + '  ' + t)
            start_time = time.time()
            try:
                st.make(run_id,t)
                print(t, '%.2f' % (time.time() - start_time))
            except Exception as e:
                failures = True
                print('Exception: ', e)


    print('\n\nFinished')
    
    if failures:
        raise
    else:
        return 0

def save_data(
            index ,
            n_per_job,
            label,
            mode,
            targets,
            run_ids,
            context,
            selection_str,
            only_one_job
            ):

    import strax
    import straxen
    import cutax
    import time
    
    print('\n\nLoading...')            
    start_time = time.time()

    st = get_context(context)

    df = st.get_array(run_ids, 
                targets, 
                add_run_id_field=True,
                selection_str=selection_str,
                )

    time.sleep(2)

    if not only_one_job:
        filename = f'{XEDATA_PATH}/hdf5/{label}-{i}.npy'
    else:
        filename = f'{XEDATA_PATH}/hdf5/{label}.npy'

    with open(filename, 'wb') as f:
        np.save(f, df)

    print('\nLoaded and saved: {filename}')
    print('%.2f' % (time.time() - start_time))


def merge_data():

    path =   os.path.join(XEDATA_PATH, 'hdf5')
    outname = os.path.join(path, f'{label}.npy')

    all_files = glob.glob(os.path.join(path, f'{label}-*.npy'))
    all_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    print(all_files)

    for i,f in enumerate(all_files):
        x = np.load(f)
        if i == 0:
            new = x
        else:
            new = np.concatenate([new,x])
            
    np.save(outname, new)


    for fdel in all_files:
        print(f"Deleting part file {fdel}")
        os.remove(fdel)

    print("Finished!")


if __name__ == "__main__":
    main()