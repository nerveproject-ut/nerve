import ffmpeg
import numpy as np
    

class VideoWriter_x264:
    def __init__(self, output_name, fps=60.0, in_pix_fmt='bgr24', out_pix_fmt='yuv420p', input_args=None, output_args={'loglevel':'quiet'}):
        self.fn = output_name
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args['framerate'] = fps
        self.input_args['pix_fmt'] = in_pix_fmt
        self.output_args['pix_fmt'] = out_pix_fmt
        self.output_args['vcodec'] = 'libx264'
    
    def write(self, frame:np.ndarray):
        if self.process is None:
            h,w = frame.shape[:2]
            self.process = (
                ffmpeg
                    .input('pipe:', format='rawvideo', s='{}x{}'.format(w, h), **self.input_args)
                    .output(self.fn, **self.output_args)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )

    def release(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


class VideoWriter_nvenc:
    """
    Ffmpeg encoder with Nvidia hardware accelleration.
    """
    def __init__(self, output_name, fps=60, in_pix_fmt='bgr24', out_pix_fmt='yuv420p', input_args={'vsync': '0'}, output_args={'loglevel':'quiet'}):
        self.fn = output_name
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args['framerate'] = fps
        self.input_args['pix_fmt'] = in_pix_fmt
        self.output_args['pix_fmt'] = out_pix_fmt
        self.output_args['vcodec'] = 'h264_nvenc'
    
    def write(self, frame):
        if self.process is None:
            h,w = frame.shape[:2]
            self.process = (
                ffmpeg
                    .input('pipe:', format='rawvideo', s='{}x{}'.format(w, h), **self.input_args)
                    .output(self.fn, **self.output_args)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )

    def release(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()