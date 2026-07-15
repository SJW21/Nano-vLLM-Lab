from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


class Profiler:
    def __init__(self, config, rank: int = 0):
        self._enabled = config.enable_profiler
        self._output = f"{config.profiler_output}/rank_{rank}"
        self._warmup_steps = config.profiler_warmup_steps
        self._active_steps = config.profiler_active_steps
        self._prof = None

    def __enter__(self):
        if not self._enabled:
            return self
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        kwargs = {
            "activities": activities,
            "on_trace_ready": tensorboard_trace_handler(self._output),
        }
        if self._active_steps > 0:
            kwargs["schedule"] = schedule(
                wait=0,
                warmup=self._warmup_steps,
                active=self._active_steps,
                repeat=1, # If need recurrent profiling, turn up.
            )
        self._prof = profile(**kwargs)
        self._prof.__enter__()
        return self

    def __exit__(self, *args):
        if self._prof is not None:
            self._prof.__exit__(*args)
            self._prof = None

    def step(self):
        if self._prof is not None:
            self._prof.step()
