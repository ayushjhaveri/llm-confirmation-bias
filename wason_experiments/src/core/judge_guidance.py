# core/judge_guidance.py
from typing import Dict

# Canonical, human-readable rule names exactly as in your rules YAML ("name" field).
# Each entry provides lightweight, rule-specific guidance to reduce false matches.
_GUIDE_TXT: Dict[str, str] = {
    # ===== Train (16) =====
    "All even": """\
ACCEPT: "all even", "each number is even".
REJECT: any rule that allows odds, or says "at least one even", or is about divisibility by 4/6 instead of parity-only.""",

    "Each divides next": """\
ACCEPT: "a divides b and b divides c", "divisibility chain a|b|c".
REJECT: mere ordering or arithmetic progression; also reject any rule that allows b%a!=0 or c%b!=0.""",

    "Exactly two equal": """\
ACCEPT: "two numbers are the same and the third is different", "exactly two equal".
REJECT: "at least two equal" (too broad) and "all equal" (too narrow).""",

    "At least one even": """\
ACCEPT: "at least one even number".
REJECT: "all even", "exactly two even", or parity conditions that exclude a single even possibility.""",

    # T2
    "All end with 6": """\
ACCEPT: "last digit 6 for all three", "each ends in 6".
REJECT: any rule that allows other last digits or only requires at least one 6.""",

    "Increasing differences": """\
ACCEPT: "(b-a) and (c-b) both positive with (b-a) < (c-b)", "gaps strictly increase".
REJECT: non-decreasing only, AP (equal gaps), or any rule without strict increase.""",

    "a is min": """\
ACCEPT: "a is the minimum", "a ≤ b and a ≤ c".
REJECT: "a < b < c" (too strict), or "nondecreasing" without stating a is the minimum.""",

    "All distinct": """\
ACCEPT: "pairwise distinct", "all different".
REJECT: equality of any two or all three; ordering claims alone are insufficient.""",

    # T3
    "All divisible by 5": """\
ACCEPT: "each multiple of 5".
REJECT: "at least two multiples of 5" (too broad) or other moduli like 10 only.""",

    "a is max": """\
ACCEPT: "a is the maximum", "a ≥ b and a ≥ c".
REJECT: strict chains like a>b>c (too strict) or max at other positions.""",

    "Non-monotone (middle between ends)": """\
ACCEPT: "(a-b)*(b-c) < 0", "b lies strictly between a and c (one gap up, one down)".
REJECT: monotone sequences, non-strict variants like allowing equality.""",

    "At least two multiples of 5": """\
ACCEPT: "two or three numbers are multiples of 5".
REJECT: "all multiples of 5" (too strict) or "at least one" (too broad).""",

    # T4
    "All divisible by 3": """\
ACCEPT: "each multiple of 3".
REJECT: "at least one multiple of 3" or divisibility by other numbers.""",

    "Alternating parity (ends same)": """\
ACCEPT: "a and c same parity, b opposite parity".
REJECT: any rule that only says 'alternating' without ensuring ends match; reject simple 'all odd/even'.""",

    "Ascending": """\
ACCEPT: "strictly increasing", "a<b<c".
REJECT: "non-decreasing" (allows equals) or AP without strictness guarantee.""",

    "Non-decreasing": """\
ACCEPT: "a ≤ b ≤ c", "non-decreasing".
REJECT: strictly increasing only (too strict).""",

    # ===== OOD Validation (8) =====
    "All end with 9": """\
ACCEPT: "all last digit 9".
REJECT: any that allows other last digits or only requires at least one 9.""",

    "Non-decreasing differences": """\
ACCEPT: "(b-a) ≤ (c-b)", "gaps do not decrease".
REJECT: strict increase required, or unrelated monotonicity only.""",

    "c is max": """\
ACCEPT: "c is the maximum", "c ≥ a and c ≥ b".
REJECT: strict c>a and c>b only if it forbids ties (too strict).""",

    "Arithmetic progression (AP)": """\
ACCEPT: "constant difference", "AP", "linear sequence with fixed step (any step)".
REJECT: "consecutive integers only" (step=1 only), "strictly increasing" without constant step.""",

    "All divisible by 7": """\
ACCEPT: "each multiple of 7".
REJECT: any other modulus or 'at least one multiple of 7'.""",

    "Exactly two even": """\
ACCEPT: "exactly two even, one odd".
REJECT: "at least two even" or "all even/odd".""",

    "At least one multiple of 4": """\
ACCEPT: "at least one number divisible by 4".
REJECT: 'all multiples of 4' or 'exactly one' (too narrow).""",

    "At least two distinct": """\
ACCEPT: "not all equal", "at least two different numbers".
REJECT: "all distinct" (too strict) or equality-only statements.""",

    # ===== OOD Test (16) =====
    "All end with 1": """\
ACCEPT: "all last digit 1".
REJECT: allowing other last digits or 'at least one ends with 1'.""",

    "c is min": """\
ACCEPT: "c is the minimum", "c ≤ a and c ≤ b".
REJECT: strict-only when it excludes ties unnecessarily.""",

    "All negative": """\
ACCEPT: "all three < 0".
REJECT: "at least one negative" or "mixed signs".""",

    "Descending": """\
ACCEPT: "strictly decreasing", "a>b>c".
REJECT: "non-increasing" (allows ties).""",

    "All odd": """\
ACCEPT: "each number odd".
REJECT: "at least one odd", "exactly two odd", parity alternation, etc.""",

    "b is (strict) max": """\
ACCEPT: "b is strictly the maximum", "b>a and b>c".
REJECT: non-strict max or 'b is maximum allowing ties'.""",

    "Mixed signs": """\
ACCEPT: "one or two numbers negative (not all, not none)".
REJECT: "all negative", "all positive". """,

    "At least one multiple of 3": """\
ACCEPT: "at least one divisible by 3".
REJECT: "all divisible by 3" or "exactly one" claims.""",

    "All prime numbers": """\
ACCEPT: "each number prime".
REJECT: "contains a prime" or 'primes OR …' mixes.""",

    "Non-increasing": """\
ACCEPT: "a ≥ b ≥ c", "non-increasing".
REJECT: strictly decreasing only (too strict).""",

    "All positive": """\
ACCEPT: "all > 0".
REJECT: "non-negative" (allows zero), or sign-mixed variants.""",

    "Contains a prime": """\
ACCEPT: "at least one is prime".
REJECT: "all prime" (too strict).""",

    "All cube numbers": """\
ACCEPT: "each a perfect cube".
REJECT: "square numbers" or 'at least one cube'.""",

    "Exactly two odd": """\
ACCEPT: "two odd, one even".
REJECT: 'at least two odd' or 'all odd/even'.""",

    "Decreasing gaps": """\
ACCEPT: "(b-a) > (c-b)", "gaps strictly decrease".
REJECT: non-increasing only, or AP (equal gaps).""",

    "At least one odd": """\
ACCEPT: "at least one odd".
REJECT: "exactly one odd", "all odd", or parity chains.""",
}

def guidance_for_rule(rule_name: str) -> str:
    key = " ".join((rule_name or "").strip())
    # exact match on provided names; else a conservative fallback
    return _GUIDE_TXT.get(rule_name, """\
ACCEPT: phrasing identical in meaning to the ground truth (same set of allowed triples).
REJECT: broader or narrower scopes, or statements from a different rule family.""")
