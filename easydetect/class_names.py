# Apache-2.0
"""Which of a dataset's classes the COCO weights already know, by name.

A model fine-tuned from the COCO checkpoint gets new class heads. A class
whose name is one of COCO's 80 (``apple``, ``사과``, ``bottles``) starts from
the row that already detects it instead of from nothing: on a small dataset
that is most of what there is to learn.
"""

from __future__ import annotations

import re

#: More names people give COCO's classes: plurals are handled separately.
ALIASES = {
    "person": ["people", "human", "man", "woman", "pedestrian", "사람", "인물"],
    "bicycle": ["bike", "자전거"],
    "car": ["자동차", "승용차"],
    "motorcycle": ["motorbike", "오토바이"],
    "airplane": ["plane", "aeroplane", "비행기"],
    "bus": ["버스"],
    "train": ["기차", "열차"],
    "truck": ["트럭"],
    "boat": ["ship", "보트"],
    "traffic light": ["신호등"],
    "fire hydrant": ["소화전"],
    "stop sign": ["정지 표지판"],
    "parking meter": ["주차 미터기"],
    "bench": ["벤치"],
    "bird": ["새"],
    "cat": ["kitten", "고양이"],
    "dog": ["puppy", "개", "강아지"],
    "horse": ["말"],
    "sheep": ["양"],
    "cow": ["소", "젖소"],
    "elephant": ["코끼리"],
    "bear": ["곰"],
    "zebra": ["얼룩말"],
    "giraffe": ["기린"],
    "backpack": ["배낭", "백팩"],
    "umbrella": ["우산"],
    "handbag": ["핸드백"],
    "tie": ["necktie", "넥타이"],
    "suitcase": ["luggage", "여행가방", "캐리어"],
    "frisbee": ["프리스비"],
    "skis": ["ski", "스키"],
    "snowboard": ["스노보드"],
    "sports ball": ["ball", "soccer ball", "football", "basketball", "baseball",
                    "공", "축구공", "농구공", "야구공"],
    "kite": ["연"],
    "baseball bat": ["야구 배트", "야구방망이"],
    "baseball glove": ["야구 글러브"],
    "skateboard": ["스케이트보드"],
    "surfboard": ["서핑보드"],
    "tennis racket": ["tennis racquet", "테니스 라켓"],
    "bottle": ["병", "물병", "페트병"],
    "wine glass": ["와인잔"],
    "cup": ["mug", "컵", "머그컵"],
    "fork": ["포크"],
    "knife": ["칼"],
    "spoon": ["숟가락"],
    "bowl": ["그릇"],
    "banana": ["바나나"],
    "apple": ["사과"],
    "sandwich": ["샌드위치"],
    "orange": ["오렌지", "귤"],
    "broccoli": ["브로콜리"],
    "carrot": ["당근"],
    "hot dog": ["hotdog", "핫도그"],
    "pizza": ["피자"],
    "donut": ["doughnut", "도넛"],
    "cake": ["케이크"],
    "chair": ["의자"],
    "couch": ["sofa", "소파"],
    "potted plant": ["plant", "houseplant", "화분"],
    "bed": ["침대"],
    "dining table": ["table", "식탁", "테이블"],
    "toilet": ["변기"],
    "tv": ["television", "tv monitor", "monitor", "텔레비전", "티비", "모니터"],
    "laptop": ["노트북"],
    "mouse": ["마우스"],
    "remote": ["remote control", "리모컨"],
    "keyboard": ["키보드"],
    "cell phone": ["phone", "mobile phone", "smartphone", "cellphone",
                   "휴대폰", "핸드폰", "스마트폰"],
    "microwave": ["전자레인지"],
    "oven": ["오븐"],
    "toaster": ["토스터"],
    "sink": ["싱크대", "세면대"],
    "refrigerator": ["fridge", "냉장고"],
    "book": ["책"],
    "clock": ["시계"],
    "vase": ["꽃병"],
    "scissors": ["가위"],
    "teddy bear": ["teddy", "곰인형", "테디베어"],
    "hair drier": ["hair dryer", "드라이기", "헤어드라이어"],
    "toothbrush": ["칫솔"],
}


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(name))).strip().lower()


def _lookup(source: dict[int, str]) -> dict[str, int]:
    """Every spelling of each source class -> its index."""
    table: dict[str, int] = {}
    for i, name in source.items():
        key = _key(name)
        for spelling in [key, *ALIASES.get(key, [])]:
            table.setdefault(_key(spelling), int(i))
    return table


def match_classes(source: dict[int, str], target: dict[int, str]) -> dict[int, int]:
    """``{target index: source index}`` for the target classes the source knows.

    By name, ignoring case, ``_``/``-`` and a plural ``s``, with the common
    English and Korean names of COCO's classes. Each source class is used once.
    """
    table, used, out = _lookup(source), set(), {}
    for j, name in target.items():
        key = _key(name)
        candidates = [key]
        if key.endswith("es"):
            candidates.append(key[:-2])
        if key.endswith("s"):
            candidates.append(key[:-1])
        for c in candidates:
            i = table.get(c)
            if i is not None and i not in used:
                out[int(j)] = i
                used.add(i)
                break
    return out


__all__ = ["ALIASES", "match_classes"]
