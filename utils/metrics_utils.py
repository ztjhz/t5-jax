import evaluate


def calculate_bleu_score(predictions: list[str], references: list[list[str]]):
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results["score"]
