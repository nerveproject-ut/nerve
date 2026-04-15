import os
import sys
import cv2
from datetime import timedelta


def get_time_hh_mm_ss(sec):
    td_str = str(timedelta(seconds=sec))
    x = td_str.split(':')
    res = ""
    if int(x[0]) > 0:
        res += "{} hours, ".format(x[0])
    if int(x[1]) > 0 or int(x[0]) > 0:
        res += "{} minutes, ".format(x[1])

    res += '{} seconds'.format(x[2])
    return res


if __name__ == '__main__':
    root_dir = '/media/neuro-gpu/DAIS_DATASET/processed_sessions'

    days = [os.path.join(root_dir, o) for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o))]
    
    total_data = {}

    for d in days:
        sessions = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        daily_sessions = {}
        for s in sessions:
            video = cv2.VideoCapture(os.path.join(s, 'L515_depth.mp4'))
            vid_fps = video.get(cv2.CAP_PROP_FPS)
            vid_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_seconds = int(round(vid_total_frames / vid_fps))
            daily_sessions[os.path.basename(s)] = vid_seconds
            #print(get_time_hh_mm_ss(vid_seconds))
        total_data[os.path.basename(d)] = dict(sorted(daily_sessions.items()))

    total_data = dict(sorted(total_data.items()))
    output = ""
    total_total = 0
    total_sessions = 0
    for d in total_data:
        total_s = 0
        for s in total_data[d]:
            total_s +=  total_data[d][s]
            total_sessions += 1
        total_total += total_s
        output += "\nDay {} --> {} ; ({}) sessions \n".format(d, get_time_hh_mm_ss(total_s), len(total_data[d]))
        for s in total_data[d]:
            output += '> {} --> {}\n'.format(s, get_time_hh_mm_ss(total_data[d][s]))
        #output += '\n'
    print(output)
    print("-----------------")
    print('Total time: {}'.format(get_time_hh_mm_ss(total_total)))
    print('Total sessions: {}\n'.format(total_sessions))
    print("-----------------")

