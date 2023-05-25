import numpy as np
import sys

targets = [
    'event_info',
]

save_targets = [
    'event_info',
]

selection_str = ""


def get_context(output_folder):

    import strax
    import straxen
    import cutax

    straxen.print_versions()
    st = cutax.contexts.xenonnt_v8(output_folder=output_folder) # 23 June 2022

    return st

def process_data(run_ids, index, npergroup, output_folder, runs_filename, mode, source):

    import strax
    import straxen
    import cutax
    import time

    print('Started processing')
    print('Total runs:', len(run_ids))
    print(run_ids)

    st = get_context(output_folder)

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

def save_data(run_ids, index, npergroup, output_folder, runs_filename, mode, source, df_output):

    import strax
    import straxen
    import cutax
    import time
    
    print('\n\nLoading...')            
    start_time = time.time()

    st = get_context(output_folder)

    df = st.get_array(run_ids, 
                save_targets, 
                add_run_id_field=True,
                selection_str=selection_str,
                )

    time.sleep(2)
    with open(f'/dali/lgrandi/cfuselli/jobs_new/hdf5/df_{source}_{i}.npy', 'wb') as f:
        np.save(f, df)

    print('\nLoaded and saved!')
    print('%.2f' % (time.time() - start_time))


def main():
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument('-i', '--index', type=int, help='index of the current job')
    parser.add_argument('-n', '--npergroup', type=int, help='how many runs per job')
    parser.add_argument('-o', '--output-folder', type=str, help='output folder for processed data')
    parser.add_argument('-dfo', '--df-output', type=str, help='output folder for processed data')
    parser.add_argument('-r', '--runs-filename', type=str, help='file containing the list of run IDs')
    parser.add_argument('-m', '--mode', type=str, help='processing mode')
    parser.add_argument('-s', '--source', type=str, help='source for processing')
    args = parser.parse_args()

    index = args.index
    npergroup = args.npergroup
    output_folder = args.output_folder
    runs_filename = args.runs_filename
    mode = args.mode
    source = args.source
    df_output = args.df_output

    with open(runs_filename) as file:
        run_ids = file.readlines()
        run_ids = [line.rstrip() for line in run_ids]
    run_ids = run_ids[index * npergroup: (index + 1) * npergroup]

    if mode == 'process':
        process_data(run_ids, index, npergroup, output_folder, runs_filename, mode, source)

    elif mode == 'save':
        save_data(run_ids, index, npergroup, output_folder, runs_filename, mode, source, df_output)

if __name__ == "__main__":
    main()