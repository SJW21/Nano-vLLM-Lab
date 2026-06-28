from nanovllm.engine.llm_engine import LLMEngine, DPLLMEngine


class LLM:
    def __init__(self, model, **kwargs):
        dp_size = kwargs.get("data_parallel_size", 1)
        if dp_size > 1:
            self._engine = DPLLMEngine(model, **kwargs)
        else:
            self._engine = LLMEngine(model, **kwargs)

    def generate(self, *args, **kwargs):
        return self._engine.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)
