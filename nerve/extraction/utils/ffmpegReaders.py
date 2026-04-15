import ffmpeg
import numpy as np
import os
import gc
import time


class VideoReader_x264:
    def __init__(self, video_path:str, in_pix_fmt='rgb24', out_pix_fmt=np.uint8, channels=3, frames_buffer=30*60) -> None:
        assert video_path.endswith(".mp4")
        assert os.path.isfile(video_path)

        try:
            video_info = next(s for s in ffmpeg.probe(video_path)['streams'] if s['codec_type'] == 'video')
        except FileNotFoundError:
            video_info = next(s for s in ffmpeg.probe(video_path, cmd="ffmpeg.ffprobe")['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.totalFrames = int(video_info['nb_frames'])
        self.fps = int(video_info['r_frame_rate'].split('/')[0])
        self.total_time = self.totalFrames/self.fps
        self.frame_period_ms = 1e3/self.fps
        self._video_path = video_path
        self._in_pix_fmt = in_pix_fmt
        self._out_pix_fmt = out_pix_fmt
        self._n_channels = channels
        self._frames_buffer = frames_buffer
        self._current_batch = None

        #just in case you want to iterate over this object
        self._current_frame = 0

        print("Reading video {} --> fps: {}; totalFrames: {}; time: {}s".format(video_path, self.fps, self.totalFrames, self.total_time))
        self._load_batch(0)
        
    def _load_batch(self, batch_idx:int):
        
        if self._current_batch is not None:
            del self._current_batch
            gc.collect() #Let's make sure to free memory

        out, _ = (
            ffmpeg
            .input(self._video_path)
            .filter_('select', 'gte(n,{})'.format(batch_idx * self._frames_buffer))
            .output('pipe:', format='rawvideo', pix_fmt=self._in_pix_fmt, vframes=self._frames_buffer , loglevel="quiet")
            .run(capture_stdout=True, capture_stderr=True)
        )

        if self._n_channels == 1:
            shape = (-1, self.height, self.width)
        else:
            shape = (-1, self.height, self.width, self._n_channels)

        batch = (
            np
            .frombuffer(out, self._out_pix_fmt)
            .reshape(*shape)
        )
        self._current_batch_idx = batch_idx
        self._current_batch = batch
        #print("batchIdx: {} -> #frames={}".format(batch_idx, len(batch)))
        return self._current_batch
    
    def GetFrameFromIndex(self, frame_index:int):
        """
        Retrieve the frame at the given index.

        Args:
            frame_index (int): index of the frame you wan tot extract

        Returns:
            (np.ndarray, float): if the given index is valid, then it is returned a tuple containing
            the frame as np.ndarray, and a float which is the time of frame since the start of the video, in millisenconds.
            If index is not valid, the it is returned a None value.
        """
        if frame_index < 0 or frame_index >= self.totalFrames:
            return None

        batch_idx = frame_index // self._frames_buffer
        rel_idx = frame_index % self._frames_buffer

        frame_time_ms = (frame_index+1)*self.frame_period_ms

        # NOTE: here we are returning a copy of ndarray for the following two reasons:
        #       1) otherwise, those ndarray would be readonly, and this could be cause troubles in an eventual next step, depending by the application
        #       2) in this way, there are not (undirect) references to the big internal buffer, so that memory can be freed before occupying the same memory amount with a new buffer instance.
        
        if batch_idx == self._current_batch_idx:
            return self._current_batch[rel_idx].copy(), frame_time_ms
        else:
            return self._load_batch(batch_idx)[rel_idx].copy(), frame_time_ms
        

    def __len__(self):
        return self.totalFrames
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._current_frame >= self.totalFrames:
            raise StopIteration
        self._current_frame += 1
        return self.GetFrameFromIndex(self._current_frame - 1)


if __name__ == "__main__":
    import cv2
    
    video_path_rgb = "/home/pietro/test_sessions/2023-10-25_14-26-59/test_rgb.mp4"
    reader = VideoReader_x264(video_path_rgb, 'rgb24', np.uint8, 3, 60)
    for i in range(reader.totalFrames):
        cv2.imshow("rgb-frame", reader.GetFrameFromIndex(i)[0])
        cv2.waitKey(1)

    assert reader.GetFrameFromIndex(-1) is None
    assert reader.GetFrameFromIndex(reader.totalFrames) is None

    video_path_depth = "/home/pietro/test_sessions/2023-10-25_14-26-59/test_depth.mp4"
    reader = VideoReader_x264(video_path_depth, 'gray16le', np.int16, 1, 60)
    for i in range(reader.totalFrames):
        cv2.imshow("depth-frame", reader.GetFrameFromIndex(i)[0])
        cv2.waitKey(1)

    video_path_conf = "/home/pietro/test_sessions/2023-10-25_14-26-59/test_conf.mp4"
    reader = VideoReader_x264(video_path_conf, 'gray', np.uint8, 1, 60)
    for i in range(reader.totalFrames):
        cv2.imshow("conf-frame", reader.GetFrameFromIndex(i)[0])
        cv2.waitKey(1)