import numpy as np
import sys
from argparse import ArgumentParser

from defaults import MY_PATH, XEDATA_PATH , OUTPUT_FOLDER

def parse_args():


    parser = ArgumentParser()
    parser.add_argument('-i', '--index', type=int, help='index of the current job')
    parser.add_argument('-n', '--n_per_job', type=int, help='how many runs per job')
    parser.add_argument('-r', '--runs-filename', type=str, help='file containing the list of run IDs')
    parser.add_argument('-m', '--mode', type=str, help='processing mode')
    
    args = parser.parse_args()

    return args


targets = [
    'event_info',
]

save_targets = [
    'event_info',
]

selection_str = ""


def get_context():

    import strax
    import straxen
    import cutax

    straxen.print_versions()
    st = cutax.contexts.xenonnt_v8(output_folder=OUTPUT_FOLDER)

    import extra_plugins
    st.register_all(extra_plugins)

    return st

def process_data(run_ids, index, n_per_job, runs_filename, mode, source):

    import strax
    import straxen
    import cutax
    import time

    print('Started processing')
    print('Total runs:', len(run_ids))
    print(run_ids)

    st = get_context()

    # Add your data processing logic here

    failures = False

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
        raise Exception(e)
    else:
        return

def save_data(run_ids, index, n_per_job, runs_filename, mode, source, df_output):

    import strax
    import straxen
    import cutax
    import time
    
    print('\n\nLoading...')            
    start_time = time.time()

    st = get_context()

    df = st.get_array(run_ids, 
                save_targets, 
                add_run_id_field=True,
                selection_str=selection_str,
                )

    time.sleep(2)
    with open(f'{XEDATA_PATH}/hdf5/df_{source}_{i}.npy', 'wb') as f:
        np.save(f, df)

    print('\nLoaded and saved!')
    print('%.2f' % (time.time() - start_time))


def main():
 

    args = parse_args()

    index = args.index
    n_per_job = args.n_per_job
    runs_filename = args.runs_filename
    mode = args.mode
    source = args.source
    df_output = args.df_output

    with open(runs_filename) as file:
        run_ids = file.readlines()
        run_ids = [line.rstrip() for line in run_ids]
    run_ids = run_ids[index * n_per_job: (index + 1) * n_per_job]

    if mode == 'process':
        process_data(run_ids, index, n_per_job, runs_filename, mode, source)

    elif mode == 'save':
        save_data(run_ids, index, n_per_job, runs_filename, mode, source, df_output)

if __name__ == "__main__":
    main()