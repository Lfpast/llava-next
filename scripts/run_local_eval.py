#!/usr/bin/env python3
"""
LLaVA-Video 本地评测脚本
兼容 SigLIP 和 VJEPA2 两种视觉塔。
修复了 vocab_size 不匹配导致的 embedding size error。

评测任务:
  1. 视频开放式问答  (eval_video_qa.json)
  2. 视频描述        (eval_video_cap.json)
  3. 视频多选题      (eval_video_mc.json)
  4. 图文问答        (eval_image_qa.json)

用法:
  python scripts/run_local_eval.py --model_path <PATH> [--max_frames 16] [--tasks all]
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from functools import partial, reduce

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-Video Local Evaluation v2")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--eval_dir", type=str,
                        default="/home/user/liujian/jiayusheng/LLaVA-NeXT/data/eval_local")
    parser.add_argument("--video_dir", type=str,
                        default="/home/user/liujian/jiayusheng/LLaVA-NeXT/data/test/video_media")
    parser.add_argument("--image_dir", type=str,
                        default="/home/user/liujian/jiayusheng/LLaVA-NeXT/data/test/image_media")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save results; defaults to <model_path>/eval")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated: video_qa,video_cap,video_mc,image_qa or 'all'")
    return parser.parse_args()



# ============================================================
# 视频加载
# ============================================================
def load_video(video_path, max_frames_num, force_sample=True):
    """均匀采样视频帧，返回 numpy ndarray (T, H, W, C) 和时间戳字符串"""
    from decord import VideoReader, cpu

    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3), dtype=np.uint8), "0.00s", 0.0

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frames / avg_fps if avg_fps > 0 else 0.0

    if total_frames <= max_frames_num or force_sample:
        frame_idx = np.linspace(0, total_frames - 1, max_frames_num, dtype=int).tolist()
    else:
        step = max(1, round(avg_fps))
        frame_idx = list(range(0, total_frames, step))[:max_frames_num]

    frame_time = ",".join([f"{i / avg_fps:.2f}s" for i in frame_idx])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time, video_time


# ============================================================
# 模型加载 —— 专门处理 vocab_size 不匹配
# ============================================================
def load_model(model_path, device="cuda"):
    """
    加载 LLaVA 模型。
    修复：若 config.json 缺少 vocab_size，自动从 safetensors 推断并写入，
    避免 from_pretrained 用默认 151936 初始化时与保存的 152064 embedding 不匹配。
    """
    import json as _json
    from llava.model.builder import load_pretrained_model

    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        cfg = _json.load(open(config_path))
        if "vocab_size" not in cfg:
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                try:
                    from safetensors import safe_open
                    idx = _json.load(open(index_path))
                    shard = idx["weight_map"].get("model.embed_tokens.weight")
                    if shard:
                        with safe_open(os.path.join(model_path, shard), framework="pt") as f:
                            embed_shape = f.get_tensor("model.embed_tokens.weight").shape
                        cfg["vocab_size"] = embed_shape[0]
                        _json.dump(cfg, open(config_path, "w"), indent=2)
                        print(f"[INFO] Auto-fixed config.json: vocab_size={embed_shape[0]}")
                except Exception as e:
                    print(f"[WARN] Could not auto-fix vocab_size: {e}")

    print(f"Loading model: {model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base=None,
        model_name="llava_qwen",
        torch_dtype="float16",      # V100 不支持 bfloat16
        device_map="auto" if device == "cuda" else device,
        attn_implementation="sdpa", # V100 兼容
    )
    model.eval()

    embed_size = model.get_input_embeddings().weight.shape[0]
    is_vjepa2 = getattr(model.config, "mm_vision_converter", False)
    print(f"[OK] embed_tokens={embed_size} | tokenizer={len(tokenizer)} | "
          f"type={'VJEPA2' if is_vjepa2 else 'SigLIP'}")
    return tokenizer, model, image_processor


# ============================================================
# 推理辅助：构建 prompt
# ============================================================
def build_prompt(question: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    if "<image>" in question:
        question = question.replace("<image>", DEFAULT_IMAGE_TOKEN)
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ============================================================
# 视频推理
# ============================================================
def run_video_inference(model, tokenizer, image_processor, video_path,
                        question, max_frames=16, device="cuda"):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    frames, frame_time, video_time = load_video(video_path, max_frames)

    # 与训练时 add_time_instruction=True 保持一致
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are "
        f"uniformly sampled from it. These frames are located at {frame_time}. "
        "Please answer the following questions related to this video."
    )
    full_question = time_instruction + "\n" + question

    pixel_values = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    first_device = next(model.parameters()).device
    pixel_values = pixel_values.to(first_device, dtype=torch.float16)
    video_tensor = [pixel_values]

    prompt = build_prompt(full_question)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(first_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    # LLaVA generate() returns only newly generated tokens (not input),
    # so decode the full output_ids[0] directly.
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ============================================================
# 图片推理
# ============================================================
def run_image_inference(model, tokenizer, image_processor, image_path,
                        question, device="cuda"):
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    first_device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(first_device, dtype=torch.float16)
    image_tensors = [image_tensor]

    prompt = build_prompt(question)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(first_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            modalities=["image"],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            use_cache=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ============================================================
# 评分工具
# ============================================================
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_choice(text: str) -> str:
    """从模型输出中提取 A/B/C/D"""
    text = text.strip()
    m = re.match(r"^([A-D])", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option)\s*(?:is\s*)?([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\.", text)
    if m:
        return m.group(1)
    return text[:1].upper() if text else ""


def compute_text_match(pred: str, gt: str) -> float:
    p, g = normalize_text(pred), normalize_text(gt)
    if g == p:
        return 1.0
    if g in p:
        return 1.0
    if p in g and len(p) > 2:
        return 1.0
    pw, gw = p.split(), g.split()
    if gw and pw and gw[0] == pw[0] and len(gw[0]) > 1:
        return 0.5
    return 0.0


def _ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu4(reference: str, hypothesis: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = normalize_text(reference).split()
        hyp = normalize_text(hypothesis).split()
        if not ref or not hyp:
            return 0.0
        return float(
            sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=SmoothingFunction().method1)
        )
    except Exception:
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_cider_single(reference: str, hypothesis: str) -> float:
    ref_tok = normalize_text(reference).split()
    hyp_tok = normalize_text(hypothesis).split()
    if not ref_tok or not hyp_tok:
        return 0.0
    scores = []
    for n in range(1, 5):
        rcnt = Counter(_ngrams(ref_tok, n))
        hcnt = Counter(_ngrams(hyp_tok, n))
        if not rcnt or not hcnt:
            scores.append(0.0)
            continue
        common = set(rcnt) & set(hcnt)
        dot = sum(rcnt[k] * hcnt[k] for k in common)
        nr = math.sqrt(sum(v * v for v in rcnt.values()))
        nh = math.sqrt(sum(v * v for v in hcnt.values()))
        scores.append(dot / (nr * nh) if nr and nh else 0.0)
    return sum(scores) / len(scores) * 10.0


# ============================================================
# 评测任务
# ============================================================
def eval_video_qa(model, tokenizer, image_processor, eval_dir, video_dir,
                  max_frames, device):
    data_path = os.path.join(eval_dir, "eval_video_qa.json")
    if not os.path.exists(data_path):
        print("  [SKIP] eval_video_qa.json not found")
        return None

    data = json.load(open(data_path))
    correct = 0.0
    total = 0
    skipped = 0
    results = []

    for item in tqdm(data, desc="video_qa"):
        video_path = os.path.join(video_dir, item["video"])
        if not os.path.exists(video_path):
            skipped += 1
            continue

        convs = item["conversations"]
        qa_pairs = [(convs[j]["value"], convs[j + 1]["value"])
                    for j in range(0, len(convs) - 1, 2)]

        item_results = []
        for question, gt in qa_pairs[:2]:
            try:
                pred = run_video_inference(
                    model, tokenizer, image_processor,
                    video_path, question, max_frames, device
                )
                score = compute_text_match(pred, gt)
                correct += score
                total += 1
                item_results.append({"q": question[:80], "gt": gt[:120], "pred": pred[:120], "score": score})
            except Exception as e:
                print(f"  [ERR] {item['id']} : {e}")
                total += 1

        results.append({"id": item["id"], "qa": item_results})

    if skipped:
        print(f"  [WARN] {skipped}/{len(data)} videos not found, skipped")

    acc = correct / total if total > 0 else 0.0
    print(f"  video_qa  accuracy = {acc:.4f}  ({correct:.0f}/{total})")
    return {"task": "video_qa", "accuracy": acc, "correct": correct, "total": total, "details": results}


def eval_video_mc(model, tokenizer, image_processor, eval_dir, video_dir,
                  max_frames, device):
    data_path = os.path.join(eval_dir, "eval_video_mc.json")
    if not os.path.exists(data_path):
        print("  [SKIP] eval_video_mc.json not found")
        return None

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    correct = 0
    total = 0
    skipped = 0
    results = []

    for item in tqdm(data, desc="video_mc"):
        video_path = os.path.join(video_dir, item["video"])
        if not os.path.exists(video_path):
            skipped += 1
            continue

        convs = item["conversations"]
        qa_pairs = [(convs[j]["value"], convs[j + 1]["value"])
                    for j in range(0, len(convs) - 1, 2)]

        item_results = []
        for question, gt in qa_pairs[:2]:
            try:
                pred = run_video_inference(
                    model, tokenizer, image_processor,
                    video_path, question, max_frames, device
                )
                pred_c = extract_choice(pred)
                gt_c   = extract_choice(gt)
                ok = int(pred_c.upper() == gt_c.upper())
                correct += ok
                total += 1
                item_results.append({"gt": gt_c, "pred": pred_c, "pred_full": pred[:80], "ok": bool(ok)})
            except Exception as e:
                print(f"  [ERR] {item['id']} : {e}")
                total += 1

        results.append({"id": item["id"], "mc": item_results})

    if skipped:
        print(f"  [WARN] {skipped}/{len(data)} videos not found, skipped")

    acc = correct / total if total > 0 else 0.0
    print(f"  video_mc  accuracy = {acc:.4f}  ({correct}/{total})")
    return {"task": "video_mc", "accuracy": acc, "correct": correct, "total": total, "details": results}


def eval_video_caption(model, tokenizer, image_processor, eval_dir, video_dir,
                       max_frames, device):
    data_path = os.path.join(eval_dir, "eval_video_cap.json")
    if not os.path.exists(data_path):
        print("  [SKIP] eval_video_cap.json not found")
        return None

    data = json.load(open(data_path))
    bleu4_list, rouge_l_list, cider_list, overlap_list = [], [], [], []
    skipped = 0
    results = []

    for item in tqdm(data, desc="video_cap"):
        video_path = os.path.join(video_dir, item["video"])
        if not os.path.exists(video_path):
            skipped += 1
            continue

        convs = item["conversations"]
        question   = convs[0]["value"]
        gt_caption = convs[1]["value"]

        try:
            pred = run_video_inference(
                model, tokenizer, image_processor,
                video_path, question, max_frames, device
            )
            bleu4   = compute_bleu4(gt_caption, pred)
            rouge_l = compute_rouge_l(gt_caption, pred)
            cider   = compute_cider_single(gt_caption, pred)
            gt_w    = set(normalize_text(gt_caption).split())
            pred_w  = set(normalize_text(pred).split())
            overlap = len(gt_w & pred_w) / max(len(gt_w), 1)

            bleu4_list.append(bleu4)
            rouge_l_list.append(rouge_l)
            cider_list.append(cider)
            overlap_list.append(overlap)
            results.append({
                "id": item["id"],
                "bleu4": bleu4, "rouge_l": rouge_l, "cider": cider, "overlap": overlap,
                "pred": pred[:200], "gt": gt_caption[:200],
            })
        except Exception as e:
            print(f"  [ERR] {item['id']} : {e}")

    if skipped:
        print(f"  [WARN] {skipped}/{len(data)} videos not found, skipped")

    n = len(results)
    avg_bleu4   = float(np.mean(bleu4_list))   if bleu4_list   else 0.0
    avg_rouge_l = float(np.mean(rouge_l_list)) if rouge_l_list else 0.0
    avg_cider   = float(np.mean(cider_list))   if cider_list   else 0.0
    avg_overlap = float(np.mean(overlap_list)) if overlap_list else 0.0
    print(f"  video_cap  BLEU4={avg_bleu4:.4f}  ROUGE-L={avg_rouge_l:.4f}  "
          f"CIDEr={avg_cider:.4f}  Overlap={avg_overlap:.4f}  ({n} samples)")
    return {"task": "video_caption",
            "avg_bleu4": avg_bleu4, "avg_rouge_l": avg_rouge_l,
            "avg_cider": avg_cider, "avg_word_overlap": avg_overlap,
            "num_samples": n, "details": results}


def eval_image_qa(model, tokenizer, image_processor, eval_dir, image_dir, device):
    data_path = os.path.join(eval_dir, "eval_image_qa.json")
    if not os.path.exists(data_path):
        print("  [SKIP] eval_image_qa.json not found")
        return None

    data = json.load(open(data_path))
    correct = 0.0
    total = 0
    skipped = 0
    results = []

    for item in tqdm(data, desc="image_qa"):
        image_path = os.path.join(image_dir, item["image"])
        if not os.path.exists(image_path):
            skipped += 1
            continue

        convs = item["conversations"]
        qa_pairs = [(convs[j]["value"], convs[j + 1]["value"])
                    for j in range(0, len(convs) - 1, 2)]

        item_results = []
        for question, gt in qa_pairs[:2]:
            try:
                pred = run_image_inference(
                    model, tokenizer, image_processor,
                    image_path, question, device
                )
                is_mc = bool(re.search(r"\b[A-D]\.", question))
                if is_mc:
                    pred_c = extract_choice(pred)
                    gt_c   = extract_choice(gt)
                    score = float(pred_c.upper() == gt_c.upper())
                else:
                    score = compute_text_match(pred, gt)

                correct += score
                total += 1
                item_results.append({"gt": gt[:80], "pred": pred[:80], "score": score})
            except Exception as e:
                print(f"  [ERR] {item.get('id','')} : {e}")
                total += 1

        results.append({"id": item.get("id", ""), "qa": item_results})

    if skipped:
        print(f"  [WARN] {skipped}/{len(data)} images not found, skipped")

    acc = correct / total if total > 0 else 0.0
    print(f"  image_qa  accuracy = {acc:.4f}  ({correct:.0f}/{total})")
    return {"task": "image_qa", "accuracy": acc, "correct": correct, "total": total, "details": results}


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, image_processor = load_model(args.model_path, args.device)

    task_list = (
        ["video_qa", "video_mc", "video_cap", "image_qa"]
        if args.tasks == "all"
        else [t.strip() for t in args.tasks.split(",")]
    )

    all_results = {}
    t0 = time.time()

    for task in task_list:
        print(f"\n{'=' * 70}\nTask: {task}\n{'=' * 70}")

        if task == "video_qa":
            r = eval_video_qa(
                model, tokenizer, image_processor,
                args.eval_dir, args.video_dir, args.max_frames, args.device
            )
        elif task == "video_mc":
            r = eval_video_mc(
                model, tokenizer, image_processor,
                args.eval_dir, args.video_dir, args.max_frames, args.device
            )
        elif task in ("video_cap", "video_caption"):
            r = eval_video_caption(
                model, tokenizer, image_processor,
                args.eval_dir, args.video_dir, args.max_frames, args.device
            )
        elif task == "image_qa":
            r = eval_image_qa(
                model, tokenizer, image_processor,
                args.eval_dir, args.image_dir, args.device
            )
        else:
            print(f"  Unknown task: {task}, skipping")
            continue

        if r:
            all_results[task] = r

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY  (elapsed: {elapsed:.1f}s)")
    print(f"{'=' * 70}")
    for task, r in all_results.items():
        if task in ("video_qa", "video_mc", "image_qa"):
            print(f"  {task:20s}: Accuracy = {r['accuracy']:.4f}  "
                  f"({r['correct']:.0f}/{r['total']})")
        elif task == "video_caption":
            print(f"  {task:20s}: BLEU4={r.get('avg_bleu4',0):.4f}  "
                  f"ROUGE-L={r.get('avg_rouge_l',0):.4f}  "
                  f"CIDEr={r.get('avg_cider',0):.4f}  "
                  f"Overlap={r.get('avg_word_overlap',0):.4f}  "
                  f"({r['num_samples']} samples)")
    print(f"{'=' * 70}")

    # 保存结果（去除 details 的简洁版 + 完整版）
    summary = {t: {k: v for k, v in r.items() if k != "details"} for t, r in all_results.items()}
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSummary  → {summary_path}")

    detail_path = os.path.join(args.output_dir, "eval_details.json")
    json.dump(all_results, open(detail_path, "w"), indent=2, ensure_ascii=False)
    print(f"Details  → {detail_path}")


if __name__ == "__main__":
    main()
