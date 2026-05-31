# src/eval/content.py
_bleurt = None

def bleurt_score(references: list[str], candidates: list[str]) -> list[float]:
    global _bleurt
    if _bleurt is None:
        from evaluate import load
        _bleurt = load("bleurt", module_type="metric")
    return _bleurt.compute(references=references, predictions=candidates)["scores"]
