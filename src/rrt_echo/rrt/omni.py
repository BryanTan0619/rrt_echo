"""Local M3-Agent Omni weights as an actual audio/video binding reviewer.

This is a model-based review, not a calibrated active-speaker detector or a
human media audit. No API key, remote upload, or implicit download is required.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
import traceback
from pathlib import Path

from ..schema import digest
from ..storage import atomic_json
from .audio import file_hash

INSTRUCTION = """Inspect the actual synchronized TARGET video WITH AUDIO and candidate reference
images. Associate ONLY the specified utterance with the visible person producing
that speech. Reference images identify candidate appearance; they do not prove
speaking. Allow unknown/offscreen and overlapping voices. A name spoken is NOT
the speaker identity. A reaction shot, central position or generic open mouth is
not sufficient. Compare timing, visible speaking motion, actual heard utterance,
and competing people. Do not invent identity from story or role expectations.
Return only a JSON object:
{"local_instance": "one supplied exact face alias such as face_0, or null",
 "verdict": "supported|unresolved|contradicted",
 "observed_speaking": true or false, "story_used_as_evidence": false,
 "reason": "specific audio/video observation or reason unresolved",
 "claims": ["at most 2 claims explicitly made in the utterance, not proven world facts"]}.
If the supplied ASR text does not match the heard speech, return unresolved.
"""


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("omni_output_not_json")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if value.get("verdict") not in {"supported", "unresolved", "contradicted"}:
        raise ValueError("invalid_omni_verdict")
    return value


class OmniBindingReviewer:
    def __init__(self, checkpoint):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.checkpoint = str(checkpoint)
        self.calls = []
        attention = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=attention,
            local_files_only=True,
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint, local_files_only=True)

    def complete(self, messages, out, stage):
        import torch
        from qwen_omni_utils import process_mm_info

        started = time.monotonic()
        request_id = digest({"messages": messages, "checkpoint": self.checkpoint, "stage": stage})
        record = {
            "request_id": request_id,
            "stage": stage,
            "model": self.checkpoint,
            "used_modalities": ["audio", "video"],
            "use_audio_in_video": True,
        }
        self.calls.append(record)
        atomic_json(out / f"{stage}.request.json", {"messages": messages, **record})
        try:
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            if audios is None or not len(audios) or videos is None or not len(videos):
                raise ValueError("omni_did_not_receive_audio_and_video")
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                fps=8,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            if "input_features" not in inputs or "feature_attention_mask" not in inputs:
                raise ValueError("processor_discarded_audio_features")
            record["audio_feature_shape"] = list(inputs["input_features"].shape)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            record["input_tokens"] = int(inputs.input_ids.shape[-1])
            record["audio_arrays"] = len(audios)
            with torch.inference_mode():
                ids = self.model.generate(
                    **inputs, use_audio_in_video=True, max_new_tokens=512, do_sample=False
                )
            raw = self.processor.batch_decode(
                ids[:, inputs.input_ids.size(1) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            record["output_tokens"] = int(ids.shape[-1] - inputs.input_ids.shape[-1])
            record["raw_output"] = raw
            result = parse_json(raw)
            result.update(model=self.checkpoint, request_id=request_id)
            return result
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            record["traceback"] = traceback.format_exc()
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            atomic_json(out / f"{stage}.response.json", record)

    def review(self, video, task, out):
        import av
        from PIL import Image

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        clip = out / "target.mp4"
        # Decode exact input seek to bounded clip; preserve synchronized AV in one
        # container. Retain source offset so model-relative timestamps remain traceable.
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                str(task["start"]),
                "-i",
                str(video),
                "-t",
                str(task["end"] - task["start"]),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                "scale=640:-2",
                "-r",
                "8",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                str(clip),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        with av.open(str(clip)) as c:
            frames = sum(1 for _ in c.decode(video=0))
        with av.open(str(clip)) as c:
            samples = sum(f.samples for f in c.decode(audio=0))
        refs = []
        aliases = {}
        for n, candidate in enumerate(task["candidates"]):
            source = Path(candidate["media"]["uri"])
            if file_hash(source) != candidate["media"]["sha256"]:
                raise ValueError("visual_reference_hash_mismatch")
            with Image.open(source) as im:
                box = candidate["region"]["box"]
                crop = im.crop(
                    tuple(v * (im.width if i % 2 == 0 else im.height) for i, v in enumerate(box))
                )
                crop.thumbnail((280, 280))
                path = out / f"candidate_{n}.jpg"
                crop.convert("RGB").save(path)
            aliases[f"face_{n}"] = candidate["local_instance"]
            refs.extend(
                [
                    {"type": "text", "text": f"Candidate local_instance=face_{n}"},
                    {"type": "image", "image": str(path)},
                ]
            )
        evidence = {
            "evidence_id": "av:" + task["task_id"],
            "source_sha256": file_hash(video),
            "uri": str(clip),
            "clip_sha256": file_hash(clip),
            "start": task["start"],
            "end": task["end"],
            "modalities": ["audio", "video"],
            "audio_samples": samples,
            "video_frames": frames,
            "clip_zero_source_seconds": task["start"],
            "candidate_instances": [c["local_instance"] for c in task["candidates"]],
            "reference_media_ids": [c["region"]["media_id"] for c in task["candidates"]],
            "temporal_precision": "transcoded_8fps_model_review_not_phoneme_alignment",
        }
        atomic_json(out / "evidence.json", evidence)
        u = task["utterance"]
        target = {
            "text": u["text"],
            "start_in_clip": u["start"] - task["start"],
            "end_in_clip": u["end"] - task["start"],
            "utterance_id": u["utterance_id"],
        }
        content = [
            {
                "type": "video",
                "video": str(clip),
                "fps": 8,
                "min_pixels": 224 * 224,
                "max_pixels": 320 * 320,
            },
            *refs,
        ]
        prompt = INSTRUCTION + "\nTARGET_UTTERANCE\n" + json.dumps(target)
        first = self.complete(
            [
                {
                    "role": "user",
                    "content": [
                        *content,
                        {
                            "type": "text",
                            "text": prompt
                            + "\nSTORY_CONTEXT (not evidence)\n"
                            + json.dumps(task["story_context"]),
                        },
                    ],
                }
            ],
            out,
            "propose",
        )
        first["local_instance"] = aliases.get(
            str(first.get("local_instance")).strip("<>"), first.get("local_instance")
        )
        second = None
        if first.get("local_instance") in evidence["candidate_instances"]:
            # Blind to story AND the proposed answer; reverse reference order to expose
            # presentation bias. Repeating a model is not an independent media audit.
            second_content = [content[0]]
            pairs = [refs[i : i + 2] for i in range(0, len(refs), 2)]
            for pair in reversed(pairs):
                second_content.extend(pair)
            second = self.complete(
                [
                    {
                        "role": "user",
                        "content": [
                            *second_content,
                            {
                                "type": "text",
                                "text": prompt
                                + "\nRecheck competing speakers. Prefer unresolved when audio/video is ambiguous.",
                            },
                        ],
                    }
                ],
                out,
                "verify",
            )
        if second:
            second["local_instance"] = aliases.get(
                str(second.get("local_instance")).strip("<>"), second.get("local_instance")
            )
        atomic_json(out / "alias_map.json", aliases)
        return evidence, first, second
