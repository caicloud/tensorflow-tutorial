import os
import tempfile

class RunConfig(object):
    def __init__(self):
        self.is_chief = True
        self.use_gpu = False
        self.sync = False
        self.num_workers = 1
        self.num_ps = 0
        self.task_type = "worker"
        self.task_index = 0
        self.max_steps = int(os.getenv("TF_MAX_STEPS", "1"))
        self.logdir = os.getenv("TF_LOGDIR", None)
        if self.logdir is None:
            self.logdir = tempfile.mkdtemp()
            print("WARNING: Using temporary folder as log directory: {}".format(self.logdir))
        self.save_checkpoints_secs = int(os.getenv("TF_SAVE_CHECKPOINTS_SECS", "600"))
        self.save_summaries_steps = int(os.getenv("TF_SAVE_SUMMARIES_STEPS", "100"))

cfg = RunConfig()
