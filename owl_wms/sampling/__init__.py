def get_sampler_cls(sampler_id):
    if sampler_id == "av_window":
        """
        Most basic Audio+Video sampler with CFG
        """
        from .av_window import AVWindowSampler
        return AVWindowSampler
    elif sampler_id == "t3_action_caching":
        """
        Tekken sampler with KV caching
        """
        from .t3_action import TekkenCachingActionSampler
        return TekkenCachingActionSampler
    elif sampler_id == "t3_caching":
        """
        Tekken sampler with KV caching
        """
        from .t3_caching import TekkenCachingSampler
        return TekkenCachingSampler
    
    elif sampler_id == "t3_sampler":
        """
        TekkenRFT model sampler with sliding window attention.
        """
        from .t3_sampler import TekkenSampler
        return TekkenSampler
    elif sampler_id == 't3_caching_cfg':
        """
        TekkenRFT model sampler with sliding window attention and CFG.
        """
        from .t3_caching import TekkenCachingSamplerWithCFG
        return TekkenCachingSamplerWithCFG
    elif sampler_id == "t3_simple":
        """
        Simple Tekken sampler without caching.
        """
        from .t3_caching import SimpleTekkenSampler
        return SimpleTekkenSampler
    elif sampler_id == "t3_caching_v2":
        """
        Advanced Tekken sampler with proper sliding context logic (v2).
        """
        from .tekken_caching_v2 import TekkenCachingSamplerV2
        return TekkenCachingSamplerV2
    elif sampler_id == "t3_simple_v2":
        """
        Simplified Tekken sampler v2 for debugging.
        """
        from .tekken_caching_v2 import SimpleTekkenSamplerV2
        return SimpleTekkenSamplerV2
    elif sampler_id == "av_caching":
        """
        Audio+Video sampler with KV caching.
        """
        #from .av_caching import AVCachingSampler
        #return AVCachingSampler
        from .av_caching_v2 import AVCachingSamplerV2
        return AVCachingSamplerV2
    elif sampler_id == "av_causal":
        """
        Audio+Video sampler with causal sampling, caches noisy history on first diffusion step.
        """
        from .av_window import CausalAVWindowSampler
        return CausalAVWindowSampler
    elif sampler_id == "av_causal_no_cfg":
        """
        Identical to av causal but skips cfg (ideal for distilled models)
        """
        from .av_window import CausalAVWindowSamplerNoCFG
        return CausalAVWindowSamplerNoCFG
    # elif sampler_id == "av_caching_one_step":
    #     """
    #     Identical to av_caching but with a hard assumption for one step to simplify
    #     """
    #     from .av_caching import AVCachingOneStepSampler
    #     return AVCachingOneStepSampler
