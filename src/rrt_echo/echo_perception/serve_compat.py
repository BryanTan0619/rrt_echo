"""Run legacy vLLM with explicit RRT video metadata. No installed package edits.
Usage: python -m rrt_echo.echo_perception.serve_compat serve MODEL [original vLLM flags]
"""

import copy
import json
import os
import time

from .video_transport import MIME, unpack


def install():
    import numpy as np
    from vllm.multimodal.video import VideoMediaIO

    original = VideoMediaIO.load_base64

    def load(self, media_type, data):
        if media_type.lower() != MIME:
            return original(self, media_type, data)
        frames, metadata = unpack(data)
        arrays = [np.asarray(self.image_io.load_base64("image/jpeg", f)) for f in frames]
        return np.stack(arrays), metadata

    VideoMediaIO.load_base64 = load
    audit_path = os.environ.get("RRT_PROCESSOR_AUDIT")
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    call = Qwen3VLMultiModalProcessor._call_hf_processor

    def audited(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        videos = []
        working = dict(mm_data)
        if working.get("videos"):
            # HF pads odd-length timestamp lists in place. Preserve source provenance.
            working["videos"] = [(array, copy.deepcopy(meta)) for array, meta in working["videos"]]
            for array, meta in mm_data["videos"]:
                videos.append({"decoded_frames": len(array), "metadata": copy.deepcopy(meta)})
        result = call(self, prompt, working, mm_kwargs, tok_kwargs)
        if videos and audit_path:
            grid = result.get("video_grid_thw")
            row = {
                "time": time.time(),
                "videos": videos,
                "video_grid_thw": grid.tolist() if grid is not None else None,
                "do_sample_frames": mm_kwargs.get("do_sample_frames"),
            }
            with open(audit_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return result

    Qwen3VLMultiModalProcessor._call_hf_processor = audited


if __name__ == "__main__":
    install()
    from vllm.entrypoints.cli.main import main

    main()
