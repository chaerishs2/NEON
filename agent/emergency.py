import re

EMERGENCY_KEYWORDS = [
    "숨이 안", "숨쉬기 힘", "숨을 못", "호흡곤란", "가슴이 아", "가슴이 답답", "가슴 통증",
    "쓰러", "의식", "정신이 없", "피가 많이", "출혈", "경련", "발작", "심하게 어지", "도와주세요",
]


def squash(text):
    return re.sub(r"\s+", "", text)


def is_emergency(text):
    t = squash(text)
    return any(squash(k) in t for k in EMERGENCY_KEYWORDS)
