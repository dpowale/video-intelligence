import json

import jiwer
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def _ensure_nltk_data() -> None:
    for resource in ("punkt_tab",):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def calculate_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """
    Calculate WER, CER, BLEU-1/2/3/4, and ROUGE-1/2/L between reference and
    hypothesis strings.

    Args:
        references:  List of ground-truth strings (one per sample).
        hypotheses:  List of predicted strings (one per sample, same order).

    Returns:
        Dict with keys WER, CER, BLEU-1..4, ROUGE-1/2/L (precision/recall/F1).
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) must be the same length"
        )

    # ── WER / CER ────────────────────────────────────────────────────────────
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)

    # ── BLEU ─────────────────────────────────────────────────────────────────
    _ensure_nltk_data()
    smooth = SmoothingFunction().method1
    ref_tokens = [[nltk.word_tokenize(r.lower())] for r in references]
    hyp_tokens = [nltk.word_tokenize(h.lower()) for h in hypotheses]

    bleu_scores = {
        f"BLEU-{n}": round(
            corpus_bleu(
                ref_tokens,
                hyp_tokens,
                weights=tuple(1 / n if i < n else 0 for i in range(4)),
                smoothing_function=smooth,
            ),
            4,
        )
        for n in (1, 2, 3, 4)
    }

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: dict[str, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "fmeasure": []}
        for k in ("rouge1", "rouge2", "rougeL")
    }
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key, val in scores.items():
            agg[key]["precision"].append(val.precision)
            agg[key]["recall"].append(val.recall)
            agg[key]["fmeasure"].append(val.fmeasure)

    rouge_scores = {
        key.upper().replace("ROUGE", "ROUGE-"): {
            sub: round(sum(vals) / len(vals), 4)
            for sub, vals in sub_dict.items()
        }
        for key, sub_dict in agg.items()
    }

    return {
        "WER": round(wer, 4),
        "CER": round(cer, 4),
        **bleu_scores,
        **rouge_scores,
    }


if __name__ == "__main__":
    refs = ["this is a test", "transcription evaluates well"]
    hyps = ["this is a text", "transcription evaluates well"]

    res = calculate_metrics(refs, hyps)
    print("Evaluation Metrics:")
    print(json.dumps(res, indent=4))
