import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import math

sys.path.append("/usr/lib/python3/dist-packages/")
#sys.path.append("/home/pietro/.local/bin")
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io.raw_info import read_raw_info
from metavision_core.event_io import EventsIterator

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Rendering of labels extracted from RGB video.")

    parser.add_argument("--input","-i", type=str, required=True,
                        help="The .raw file containing data")

    parser.add_argument("--output","-o", type=str, required=True,
                        help="The path of resulting .hdf5 output file.")
    
    parser.add_argument('--avoid-support-indexes', dest='store_support_indexes', action='store_false',
                        help="If support indexes are enabled, in the output file, we will store also a second dataset, \
                        which will act as mapping from (quantized time from the start of events) to (index \
                        of the first event after that period). For example, if period value is 100us, in \
                        this vector index 1 will contain the index of the first event after 100uS, while the \
                        fifth element will be the index of the first event after 500uS from the start, and so on. \
                        As downside, it slows down conversion to hdf5, and stores some tens of MB.")
    parser.set_defaults(store_support_indexes=True)

    parser.add_argument("--support-period-us", type=int, default=100,
                        help="how often to store a support index? Values in microSeconds (uS)")
   
    return parser.parse_args()


def main():

    args = get_arguments()
    input_path = str(args.input)
    output_path = str(args.output)

    use_support_indexes=bool(args.store_support_indexes)
    if use_support_indexes:
        support_timestep = int(args.support_period_us)

    assert input_path.endswith(".raw")
    assert output_path.endswith(".hdf5")
    assert os.path.isfile(input_path)

    if not os.path.isdir(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))


    first_event_ts = -1
    last_event_ts = -1
    total_events_number = 0
    print("Start checking events..")
    for ev in EventsIterator(input_path, delta_t=100000):
        if ev.size == 0:
            continue
        if first_event_ts < 0:
            first_event_ts = ev[0]['t']
        last_event_ts = ev[-1]['t']
        total_events_number += ev.size
        last_ev = ev[-1]

    total_time = last_event_ts - first_event_ts
    print("Events: {}. Total time: {}".format(total_events_number, total_time))

    out_file = h5py.File(output_path,'w')
    dataset = out_file.create_dataset('events', (total_events_number,), dtype=last_ev.dtype,
                                      compression="gzip", compression_opts=4)
    dataset.attrs['name'] = 'PROPHESEE_EVK4'
    dataset.attrs['width'] = 1280
    dataset.attrs['height'] = 720
    dataset.attrs['total_events'] = total_events_number
    dataset.attrs['total_time_uS'] = total_time

    if use_support_indexes:
        # Let's create also a support index table, which contains indexes of events every T time
        # This is not necessary, but can safe A LOT OF TIME when processing temporal batches of data
        # timestamps are integer expressed in uSeconds

        indexes = total_time // support_timestep + 1

        support_dataset_name = 'support_indexes'
        print("Storing {} indexes in dataset \'{}\'. Timestep = {} uS".format(indexes, support_dataset_name, support_timestep))
        
        support_idx_ds = out_file.create_dataset(support_dataset_name, (indexes,), dtype=np.uint64,
                                                 compression="gzip", compression_opts=4)
        #of course, the first polarity event is the one that goes at idx 0 --> because 0uS are passed from the start.
        support_idx_ds[0] = 0
        next_support_idx = 1
        support_idx_ds.attrs['timestep_uS'] = support_timestep


    curr_out_idx = 0
    window_size_us = 10000

    for ev in tqdm(EventsIterator(input_path, delta_t=window_size_us), total=math.ceil(total_time / window_size_us)):
        n_pols = ev.size
        if n_pols == 0:
            print("skip")
            continue

        dataset[curr_out_idx : curr_out_idx + n_pols] = ev
        if use_support_indexes:
            next_ts_threshold = first_event_ts + next_support_idx * support_timestep
            max_ts = ev[-1]['t']
            while max_ts >= next_ts_threshold:
                # if we are here, the next event which we aim to index is in this batch. Let's find it.
                batch_idx = np.argmax(ev['t']>=next_ts_threshold)
                support_idx_ds[next_support_idx] = curr_out_idx + batch_idx
                next_support_idx += 1
                next_ts_threshold = first_event_ts + next_support_idx * support_timestep

        curr_out_idx += n_pols



    return


if __name__ == "__main__":
    main()
