import time
import numpy as np
import atexit

cuda_timers = {}
timers = {}

class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')

    # are we sure about this?  maybe they should belong to different categories.
    all_timers = [*cuda_timers.items(), *timers.items()]

    sorted_dict = {}

    total_time_measured = 0
    for idx, (timer_name, timing_values) in enumerate(all_timers):
        tot_time = float(np.sum(np.array(timing_values)))
        total_time_measured += tot_time
        sorted_dict[idx] = tot_time
    
    sorted_dict = dict(sorted(sorted_dict.items(), key=lambda x:x[1], reverse=True))


    for idx in sorted_dict.keys():
        timer_name, timing_values = all_timers[idx]
        avg_iteration_time = np.mean(np.array(timing_values))
        n_iterations = len(timing_values)
        total_time = sorted_dict[idx]
        out_str = '  +  '

        out_str += 'AVG ITERATION TIME: '
        if avg_iteration_time < 1000.0:
            out_str += '{:.2f} ms'.format(avg_iteration_time)
        else:
            out_str += '{:.2f} s'.format(avg_iteration_time / 1000.0)
        
        out_str += ' \tITERATIONS: {}'.format(n_iterations)

        out_str += ' \tTOTAL TIME: '
        if total_time < 1000.0:
            out_str += '{:.2f} ms'.format(total_time)
        else:
            out_str += '{:.2f} s'.format(total_time / 1000.0)
        
        usage = 100.0 * total_time/total_time_measured
        if usage < 10.0:
            out_str += '\t{:.2f}%'.format(usage)
        else:
            out_str += '\t{:.1f}%'.format(usage)

        out_str += '\t-->\t{}'.format(timer_name)
        print(out_str)


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
