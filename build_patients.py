#!/usr/bin/env python3
"""NEON 가상 환자 32명 생성 스크립트.

구성: 주인공 2명(p1, p2: 직접 설계) + CSV 기반 변환 30명(p3~p32).
개인정보 원칙:
  - CSV의 이름/의사/병원/보험/청구액/병실 번호는 사용하지 않는다(이름은 전부 임의 가명).
  - 미성년자(만 19세 미만)는 제외한다.
  - 연락처/주민번호/주소/생년월일 필드는 만들지 않는다(나이만 보관).
  - 모든 환자에 synthetic=True 와 데이터 출처(provenance)를 남긴다.
  - to_llm_view(): 외부 모델로 보낼 최소 정보 뷰(이름, 병실, 간호사/의사 이름, 절대 날짜 제외).

실행:
  python3 -I build_patients.py --csv <healthcare_dataset.csv> --out <출력 폴더> [--today YYYY-MM-DD] [--seed N]
시연 날짜가 바뀌면 --today 를 바꿔 다시 실행한다(일정/수술 경과일이 오늘 기준으로 다시 계산됨).
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "1.0"

CONDITIONS = ["Diabetes", "Hypertension", "Asthma", "Arthritis", "Cancer", "Obesity"]
AGE_BINS = [(19, 39), (40, 54), (55, 64), (65, 74), (75, 89)]
ADMISSION_KO = {"Elective": "예정 입원", "Urgent": "긴급 입원", "Emergency": "응급실 경유 입원"}
TEST_KO = {"Normal": "정상", "Abnormal": "이상 소견", "Inconclusive": "판정 보류"}
GENDER_KO = {"Male": "남", "Female": "여"}

WARD_OF = {
    "Diabetes": "내분비내과 병동", "Obesity": "내분비내과 병동",
    "Hypertension": "순환기내과 병동", "Asthma": "호흡기내과 병동",
    "Arthritis": "정형외과 병동", "Cancer": "종양내과 병동",
}
WARD_INFO = {
    "내분비내과 병동": {"floor": 5, "nurses": ["문하윤", "백소영"], "docs": ["한상민 교수", "윤서현 교수"]},
    "순환기내과 병동": {"floor": 6, "nurses": ["이지은", "정다은"], "docs": ["강태준 교수", "오지민 교수"]},
    "호흡기내과 병동": {"floor": 7, "nurses": ["최예린", "장하늘"], "docs": ["조현석 교수", "신유나 교수"]},
    "정형외과 병동": {"floor": 3, "nurses": ["박서진", "김나연"], "docs": ["임재현 교수", "송미래 교수"]},
    "종양내과 병동": {"floor": 8, "nurses": ["황소민", "안유리"], "docs": ["서동혁 교수", "권수아 교수"]},
}

SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍"]
MALE_GIVEN = ["도현", "시우", "준서", "지훈", "현우", "승민", "재민", "태윤", "건우", "성진", "영호", "대현", "병철", "상우", "경수", "진석", "동원", "민기", "용수", "철민"]
FEMALE_GIVEN = ["서윤", "지우", "하은", "수빈", "유진", "은지", "미영", "정희", "영숙", "순옥", "혜진", "소연", "나래", "다은", "경희", "옥자", "미경", "현주", "보라", "선영"]

ALLERGY_POOL = ["페니실린", "조영제", "라텍스", "갑각류"]
FORBIDDEN_KEYS = {"phone", "tel", "rrn", "ssn", "address", "email", "birth_date", "birthday", "resident_id"}


# ---------------------------------------------------------------- 날짜 유틸
def at(today, day, hh, mm=0):
    d = datetime(today.year, today.month, today.day) + timedelta(days=day, hours=hh, minutes=mm)
    return d.strftime("%Y-%m-%dT%H:%M")


def dstr(today, day):
    return (today + timedelta(days=day)).strftime("%Y-%m-%d")


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def parse_d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def rel_label(days):
    names = {-2: "그저께", -1: "어제", 0: "오늘", 1: "내일", 2: "모레"}
    if days in names:
        return names[days]
    return f"{days}일 후" if days > 0 else f"{-days}일 전"


def rel_when(dt, today):
    return f"{rel_label((dt.date() - today).days)} {dt:%H:%M}"


# ---------------------------------------------------------------- CSV 로딩
def load_rows(path):
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            key = tuple(r.values())
            if key in seen:  # 완전 중복 행 제거
                continue
            seen.add(key)
            try:
                age = int(r["Age"])
                adm = datetime.strptime(r["Date of Admission"], "%Y-%m-%d").date()
                dis = datetime.strptime(r["Discharge Date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                continue
            if not (19 <= age <= 89):  # 미성년자 제외
                continue
            if (r["Medical Condition"] not in CONDITIONS or r["Test Results"] not in TEST_KO
                    or r["Admission Type"] not in ADMISSION_KO or r["Gender"] not in GENDER_KO):
                continue
            r["_idx"], r["_age"], r["_los"] = i, age, max((dis - adm).days, 1)
            rows.append(r)
    return rows


def pick_rows(rows, rng):
    picked = {}
    for c in CONDITIONS:
        pool = [r for r in rows if r["Medical Condition"] == c]
        chosen = []
        for lo, hi in AGE_BINS:
            cand = [r for r in pool if lo <= r["_age"] <= hi]
            if not cand:
                raise SystemExit(f"CSV에 {c} {lo}-{hi}세 후보가 없습니다.")
            chosen.append(rng.choice(cand))
        picked[c] = chosen
    return picked


# ---------------------------------------------------------------- 이름/병실 배정
class NameMaker:
    def __init__(self, rng, reserved):
        self.rng, self.used = rng, set(reserved)

    def make(self, gender_ko):
        given = MALE_GIVEN if gender_ko == "남" else FEMALE_GIVEN
        for _ in range(1000):
            n = self.rng.choice(SURNAMES) + self.rng.choice(given)
            if n not in self.used:
                self.used.add(n)
                return n
        raise SystemExit("가명 생성 실패")


class RoomAllocator:
    def __init__(self, rng, reserved):
        self.pools = {}
        for ward, info in WARD_INFO.items():
            pos = [(info["floor"] * 100 + r, b) for r in range(1, 11) for b in range(1, 5)]
            rng.shuffle(pos)
            self.pools[ward] = [p for p in pos if p not in reserved]

    def take(self, ward):
        room, bed = self.pools[ward].pop()
        return f"{room}호", f"{bed}번 침상"


# ---------------------------------------------------------------- 활력징후
def make_vitals(rng, today, kind, cond):
    out = []
    base_t = {"post_op": 36.9, "chemo": 36.7}.get(kind, 36.6)
    for d, h in [(-1, 6), (-1, 14), (-1, 22), (0, 6)]:
        v = {"time": at(today, d, h), "temp": round(base_t + rng.uniform(-0.2, 0.3), 1),
             "pulse": rng.randint(66, 88)}
        if cond == "Hypertension":
            v["bp_sys"], v["bp_dia"] = rng.randint(132, 150), rng.randint(82, 94)
        else:
            v["bp_sys"], v["bp_dia"] = rng.randint(112, 128), rng.randint(68, 80)
        v["spo2"] = rng.randint(94, 98) if cond == "Asthma" else rng.randint(96, 99)
        if cond == "Diabetes":
            v["glucose"] = rng.randint(110, 185)
        out.append(v)
    return out


def fall_and_mobility(age, surgical, rng):
    if surgical:
        return ("높음" if age >= 60 else "중간"), "보행 보조 필요"
    if age >= 75:
        return "높음", rng.choice(["보행 보조 필요", "독립 보행"])
    if age >= 60:
        return "중간", "독립 보행"
    return "낮음", "독립 보행"


# ---------------------------------------------------------------- 질환별 템플릿
def tmpl_diabetes(rng, today):
    meds_pool = [
        {"name": "메트포르민", "schedule": "아침·저녁 식후", "route": "경구", "purpose": "혈당 조절"},
        {"name": "인슐린(속효성)", "schedule": "식전, 간호사 투여", "route": "주사", "purpose": "혈당 조절"},
        {"name": "인슐린(지속형)", "schedule": "취침 전, 간호사 투여", "route": "주사", "purpose": "혈당 조절"},
        {"name": "글리메피리드", "schedule": "아침 식전", "route": "경구", "purpose": "혈당 조절"},
    ]
    sched_pool = [
        (1, 6, 30, "공복 혈당 채혈", True, "전날 밤 9시 이후 금식 · 금식 중 혈당약/인슐린은 간호사에게 확인"),
        (1, 8, 0, "당화혈색소(HbA1c) 검사", False, "채혈 · 금식 불필요"),
        (2, 10, 0, "안저 검사", False, "눈 확대 점안 후 몇 시간 눈부심 가능"),
        (1, 14, 0, "영양 상담", False, "식사 후 진행"),
        (2, 9, 0, "복부 초음파", True, "검사 전 8시간 금식 필요"),
    ]
    return {
        "diagnosis": rng.choice(["제2형 당뇨병 (혈당 조절 및 합병증 평가)", "제2형 당뇨병 (고혈당 교정)"]),
        "surgery": None, "meds": rng.sample(meds_pool, 2), "sched": rng.sample(sched_pool, 2),
        "care": ["혈당 측정 하루 4회 (식전·취침 전)", "저혈당 증상(식은땀·떨림·어지러움) 시 즉시 호출벨",
                 "발 관리 및 피부 상처 확인"],
        "diet": "당뇨식", "kind": "general",
    }


def tmpl_hypertension(rng, today):
    meds_pool = [
        {"name": "암로디핀", "schedule": "아침 식후", "route": "경구", "purpose": "혈압 조절"},
        {"name": "로사르탄", "schedule": "아침 식후", "route": "경구", "purpose": "혈압 조절"},
        {"name": "하이드로클로로티아지드", "schedule": "아침 식후", "route": "경구", "purpose": "혈압 조절(소변량 증가 가능)"},
    ]
    sched_pool = [
        (1, 9, 0, "24시간 활동혈압 측정", False, "장비를 착용하고 일상 활동"),
        (1, 11, 0, "심전도 검사", False, "금식 불필요"),
        (2, 10, 0, "심장 초음파", False, "금식 불필요"),
        (1, 8, 0, "소변 검사", False, "아침 첫 소변"),
    ]
    return {
        "diagnosis": "고혈압 (혈압 조절 및 약물 조정)", "surgery": None,
        "meds": rng.sample(meds_pool, 2), "sched": rng.sample(sched_pool, 2),
        "care": ["혈압 측정 하루 3회", "저염식 안내", "자리에서 일어날 때 천천히 (어지러움 주의)"],
        "diet": "저염식", "kind": "general",
    }


def tmpl_asthma(rng, today):
    sched_pool = [
        (1, 10, 0, "폐기능 검사", False, "검사 전 흡입기 사용 시점은 간호사 안내"),
        (1, 13, 0, "흉부 X-ray", False, "금식 불필요"),
        (2, 9, 30, "알레르기 검사", False, "피부 반응 검사 · 약 30분 소요"),
    ]
    return {
        "diagnosis": "기관지 천식 (급성 악화 후 관리)", "surgery": None,
        "meds": [
            {"name": "속효성 흡입기", "schedule": "필요 시", "route": "흡입", "purpose": "호흡 불편 완화"},
            {"name": "흡입 스테로이드", "schedule": "하루 2회, 사용 후 입 헹굼", "route": "흡입", "purpose": "염증 조절"},
            {"name": "경구 스테로이드(단기)", "schedule": "아침 식후", "route": "경구", "purpose": "급성 악화 치료"},
        ],
        "sched": rng.sample(sched_pool, 2),
        "care": ["산소포화도 하루 3회 확인", "호흡이 힘들면 즉시 호출벨", "흡입기 사용법 교육", "먼지·찬 공기 노출 주의"],
        "diet": "일반식", "kind": "general",
    }


def tmpl_obesity(rng, today):
    sched_pool = [
        (1, 9, 0, "체성분 검사", False, "금식 불필요"),
        (1, 14, 0, "영양 상담", False, ""),
        (2, 10, 0, "운동 처방 상담", False, "편한 복장"),
        (1, 6, 30, "갑상선 기능 혈액 검사", True, "공복 채혈"),
        (2, 21, 0, "수면 다원 검사", False, "야간 검사 · 저녁 식사 후 입실"),
    ]
    meds = [] if rng.random() < 0.5 else [
        {"name": "암로디핀", "schedule": "아침 식후", "route": "경구", "purpose": "혈압 조절"}]
    return {
        "diagnosis": "고도비만 (대사 평가 및 체중 관리)", "surgery": None, "meds": meds,
        "sched": rng.sample(sched_pool, 2),
        "care": ["식이 기록", "보행 운동 하루 2회", "매일 아침 체중 측정"],
        "diet": "저열량식", "kind": "general",
    }


def tmpl_arthritis(rng, today, bin_idx, gender_ko):
    side = rng.choice(["우측", "좌측"])
    if bin_idx == 2:  # 무릎 인공관절 수술 후 1일
        pod, name = 1, f"{side} 무릎 인공관절 전치환술"
        diag = f"{side} 무릎 골관절염 (인공관절 수술 후 회복)"
    elif bin_idx == 3:  # 고관절 인공관절 수술 후 3일
        pod, name = 3, f"{side} 고관절 인공관절 치환술"
        diag = f"{side} 고관절 골관절염 (인공관절 수술 후 회복)"
    else:
        return {
            "diagnosis": rng.choice(["골관절염 (통증 조절 및 재활)", "류마티스 관절염 (악화기 치료)"]),
            "surgery": None,
            "meds": [
                {"name": "아세트아미노펜", "schedule": "6시간마다", "route": "경구", "purpose": "통증 조절"},
                {"name": "소염진통제", "schedule": "아침·저녁 식후", "route": "경구", "purpose": "염증·통증 조절"},
                {"name": "위장보호제", "schedule": "식전", "route": "경구", "purpose": "위 보호"},
            ],
            "sched": [(1, 10, 0, "물리치료", False, "편한 복장"), (1, 8, 30, "혈액 염증 수치 검사", False, "금식 불필요")],
            "care": ["관절 보호 및 무리한 동작 피하기", "낙상 주의", "온찜질 (간호사 안내에 따라)"],
            "diet": "일반식", "kind": "general",
        }
    return {
        "diagnosis": diag,
        "surgery": {"name": name, "post_op_day": pod},
        "meds": [
            {"name": "아세트아미노펜", "schedule": "6시간마다, 통증이 심하면 간호사에게 요청", "route": "경구", "purpose": "통증 조절"},
            {"name": "항혈전 주사", "schedule": "하루 1회, 간호사 투여", "route": "주사", "purpose": "혈전 예방"},
            {"name": "예방적 항생제", "schedule": "처방 기간 중 정해진 시간", "route": "주사", "purpose": "수술 후 감염 예방"},
        ],
        "sched": [(0, 17, 0, "상처 소독", False, "하루 2회 (오전 10시, 오후 5시)"),
                  (1, 10, 0, "상처 소독", False, ""),
                  (1, 14, 0, "재활치료(보행 훈련)", False, "보조기 사용")],
        "care": ["상처 소독 하루 2회", "보행 시 보조 필요 (낙상 주의)", "수술한 다리를 베개로 높이기",
                 "기침·심호흡 운동", "수술 부위 통증·붓기·진물·열 변화는 간호사에게 알리기"],
        "diet": "일반식", "kind": "post_op",
    }


def tmpl_cancer(rng, today, bin_idx, gender_ko):
    types_m = [("대장암", "복강경 대장 절제술"), ("위암", "위 절제술"), ("폐암", "폐엽 절제술")]
    types_f = [("유방암", "유방 부분절제술"), ("대장암", "복강경 대장 절제술"),
               ("갑상선암", "갑상선 절제술"), ("위암", "위 절제술")]
    ctype, sname = rng.choice(types_m if gender_ko == "남" else types_f)
    if bin_idx in (1, 2):  # 수술 후
        pod = 3 if bin_idx == 1 else 4
        soft = ctype in ("대장암", "위암")
        return {
            "diagnosis": f"{ctype} (수술 후 회복)",
            "surgery": {"name": sname, "post_op_day": pod},
            "meds": [
                {"name": "아세트아미노펜", "schedule": "6시간마다", "route": "경구/주사", "purpose": "통증 조절"},
                {"name": "예방적 항생제", "schedule": "처방 기간 중 정해진 시간", "route": "주사", "purpose": "수술 후 감염 예방"},
            ],
            "sched": [(0, 17, 0, "상처 소독", False, "하루 2회"), (1, 15, 0, "조기 보행 운동", False, "간호사 동행")],
            "care": ["상처 소독 하루 2회", "조기 보행 격려", "기침·심호흡 운동",
                     "수술 부위 통증 악화·진물·열은 간호사에게 알리기"],
            "diet": "단계적 식이 진행(물 → 죽)" if soft else "일반식", "kind": "post_op",
        }
    cycle = rng.randint(1, 6)
    return {
        "diagnosis": f"{ctype} (항암 치료 {cycle}주기)", "surgery": None,
        "meds": [
            {"name": "항구토제", "schedule": "항암 주사 전후", "route": "경구/주사", "purpose": "구역·구토 예방"},
            {"name": "진통제", "schedule": "필요 시", "route": "경구", "purpose": "통증 조절"},
            {"name": "변비약", "schedule": "필요 시", "route": "경구", "purpose": "변비 완화"},
        ],
        "sched": [(1, 8, 0, "항암 전 혈액 검사", False, "금식 불필요"),
                  (1, 10, 0, "항암 주사", False, "수 시간 소요 예상"),
                  (2, 14, 0, "CT 촬영", True, "조영제 사용 · 금식 시간은 간호사에게 확인")][:rng.randint(2, 3)],
        "care": ["손 위생 철저 (감염 예방)", "발열 시 즉시 호출벨", "구강 관리", "식사는 소량씩 자주"],
        "diet": "부드러운 식사", "kind": "chemo",
    }


# ---------------------------------------------------------------- 환자 조립
def finish_patient(pid, name, age, gender_ko, blood, cond, admitted_offset, exp_offset, adm_type,
                   tpl, today, rng, alloc, nurse_idx, test_raw, provenance, allergies=None, vitals=None):
    ward = WARD_OF[cond]
    room, bed = alloc if isinstance(alloc, tuple) else alloc.take(ward)
    info = WARD_INFO[ward]
    surgery = None
    if tpl["surgery"]:
        pod = tpl["surgery"]["post_op_day"]
        surgery = {"name": tpl["surgery"]["name"], "date": dstr(today, -pod), "post_op_day": pod}
    schedule = []
    for (d, h, m, item, fasting, note) in sorted(tpl["sched"], key=lambda x: (x[0], x[1], x[2])):
        schedule.append({"datetime": at(today, d, h, m), "item": item, "fasting": fasting, "note": note})
    surgical = surgery is not None
    fall, mobility = fall_and_mobility(age, surgical, rng)
    if allergies is None:
        allergies = [rng.choice(ALLERGY_POOL)] if rng.random() < 0.3 else []
    if vitals is None:
        vitals = make_vitals(rng, today, tpl["kind"], cond)
        adm_start = datetime.strptime(dstr(today, -admitted_offset), "%Y-%m-%d")
        vitals = [v for v in vitals if parse_dt(v["time"]) >= adm_start]
    return {
        "id": pid, "name": name, "age": age, "gender": gender_ko, "blood_type": blood,
        "ward": ward, "room": room, "bed": bed,
        "admitted_at": dstr(today, -admitted_offset), "expected_discharge": dstr(today, exp_offset),
        "admission_type": ADMISSION_KO[adm_type], "condition": cond, "diagnosis": tpl["diagnosis"],
        "surgery": surgery, "medications": tpl["meds"],
        "allergies": allergies, "allergy_checked": True,
        "schedule": schedule,
        "vitals": vitals,
        "care_notes": tpl["care"], "diet": tpl["diet"], "mobility": mobility, "fall_risk": fall,
        "last_test_result": test_raw,
        "last_test_summary": f"최근 검사 결과: {TEST_KO[test_raw]} (자세한 설명은 담당 의료진이 안내)",
        "watch_flags": [],
        "assigned_nurse": info["nurses"][nurse_idx % 2],
        "attending_physician": info["docs"][nurse_idx % 2],
        "synthetic": True, "provenance": provenance,
    }


def hero_patients(today):
    p1 = {
        "id": "p1", "name": "박정호", "age": 61, "gender": "남", "blood_type": "A+",
        "ward": "내분비내과 병동", "room": "502호", "bed": "2번 침상",
        "admitted_at": dstr(today, -2), "expected_discharge": dstr(today, 4),
        "admission_type": "긴급 입원", "condition": "Diabetes",
        "diagnosis": "제2형 당뇨병 (혈당 조절 및 합병증 평가)", "surgery": None,
        "medications": [
            {"name": "메트포르민", "schedule": "아침·저녁 식후", "route": "경구", "purpose": "혈당 조절"},
            {"name": "인슐린(속효성)", "schedule": "식전, 간호사 투여", "route": "주사", "purpose": "혈당 조절"},
        ],
        "allergies": [], "allergy_checked": True,
        "schedule": [
            {"datetime": at(today, 1, 6, 30), "item": "공복 혈당 채혈", "fasting": True,
             "note": "전날 밤 9시 이후 금식 · 금식 중 혈당약/인슐린 투여는 반드시 간호사에게 확인"},
            {"datetime": at(today, 1, 9, 0), "item": "복부 초음파", "fasting": True,
             "note": "검사 전 8시간 금식 필요"},
            {"datetime": at(today, 1, 14, 0), "item": "영양 상담", "fasting": False, "note": "식사 후 병실 방문 진행"},
        ],
        "vitals": [
            {"time": at(today, -1, 6), "temp": 36.6, "pulse": 78, "bp_sys": 128, "bp_dia": 80, "spo2": 97, "glucose": 158},
            {"time": at(today, -1, 14), "temp": 36.7, "pulse": 76, "bp_sys": 126, "bp_dia": 78, "spo2": 97, "glucose": 142},
            {"time": at(today, -1, 22), "temp": 36.5, "pulse": 74, "bp_sys": 124, "bp_dia": 78, "spo2": 98, "glucose": 171},
            {"time": at(today, 0, 6), "temp": 36.6, "pulse": 75, "bp_sys": 126, "bp_dia": 80, "spo2": 97, "glucose": 149},
        ],
        "care_notes": ["혈당 측정 하루 4회 (식전·취침 전)", "저혈당 증상(식은땀·떨림·어지러움) 시 즉시 호출벨",
                       "검사 금식 중 인슐린/혈당약 조정은 의료진 확인 후 진행", "발 관리 및 피부 상처 확인"],
        "diet": "당뇨식 (검사 전 금식 예정)", "mobility": "독립 보행", "fall_risk": "낮음",
        "last_test_result": "Abnormal", "last_test_summary": "최근 검사 결과: 이상 소견 (자세한 설명은 담당 의료진이 안내)",
        "watch_flags": [], "assigned_nurse": "문하윤", "attending_physician": "한상민 교수",
        "synthetic": True, "provenance": {"method": "handcrafted", "csv_row": None, "csv_fields_used": []},
    }
    p2 = {
        "id": "p2", "name": "이순자", "age": 67, "gender": "여", "blood_type": "B+",
        "ward": "정형외과 병동", "room": "301호", "bed": "2번 침상",
        "admitted_at": dstr(today, -3), "expected_discharge": dstr(today, 6),
        "admission_type": "예정 입원", "condition": "Arthritis",
        "diagnosis": "우측 무릎 골관절염 (인공관절 수술 후 회복)",
        "surgery": {"name": "우측 무릎 인공관절 전치환술", "date": dstr(today, -2), "post_op_day": 2},
        "medications": [
            {"name": "아세트아미노펜", "schedule": "6시간마다, 통증이 심하면 간호사에게 요청", "route": "경구", "purpose": "통증 조절"},
            {"name": "항혈전 주사", "schedule": "하루 1회, 간호사 투여", "route": "주사", "purpose": "혈전 예방"},
            {"name": "예방적 항생제", "schedule": "처방 기간 중 정해진 시간", "route": "주사", "purpose": "수술 후 감염 예방"},
            {"name": "위장보호제", "schedule": "아침 식전", "route": "경구", "purpose": "위 보호"},
        ],
        "allergies": ["라텍스"], "allergy_checked": True,
        "schedule": [
            {"datetime": at(today, 0, 17, 0), "item": "상처 소독", "fasting": False, "note": "하루 2회 (오전 10시, 오후 5시)"},
            {"datetime": at(today, 1, 9, 0), "item": "X-ray 촬영 (수술 부위 확인)", "fasting": False,
             "note": "금식 필요 없음 · 휠체어로 이동"},
            {"datetime": at(today, 1, 10, 0), "item": "상처 소독", "fasting": False, "note": ""},
            {"datetime": at(today, 1, 14, 0), "item": "재활치료(보행 훈련)", "fasting": False, "note": "보조기 사용"},
            {"datetime": at(today, 1, 17, 0), "item": "상처 소독", "fasting": False, "note": ""},
        ],
        "vitals": [
            {"time": at(today, -1, 6), "temp": 37.0, "pulse": 82, "bp_sys": 122, "bp_dia": 76, "spo2": 97},
            {"time": at(today, -1, 14), "temp": 37.2, "pulse": 84, "bp_sys": 124, "bp_dia": 76, "spo2": 97},
            {"time": at(today, -1, 22), "temp": 37.4, "pulse": 88, "bp_sys": 126, "bp_dia": 78, "spo2": 96},
            {"time": at(today, 0, 6), "temp": 37.6, "pulse": 90, "bp_sys": 128, "bp_dia": 80, "spo2": 96},
        ],
        "care_notes": ["상처 소독 하루 2회", "보행 시 보조 필요 (낙상 주의)", "수술한 다리를 베개로 높이기",
                       "기침·심호흡 운동", "수술 부위 통증·붓기·진물·열 변화는 간호사에게 알리기"],
        "diet": "일반식", "mobility": "보행 보조 필요", "fall_risk": "높음",
        "last_test_result": "Normal", "last_test_summary": "최근 검사 결과: 정상 (자세한 설명은 담당 의료진이 안내)",
        "watch_flags": ["수술 후 체온 상승 추세 (37.0 → 37.6℃)"],
        "assigned_nurse": "박서진", "attending_physician": "임재현 교수",
        "synthetic": True, "provenance": {"method": "handcrafted", "csv_row": None, "csv_fields_used": []},
    }
    return [p1, p2]


def apply_flags(patients, rng):
    def pick(cond_fn):
        c = [p for p in patients if p["id"] not in ("p1", "p2") and cond_fn(p)]
        return rng.choice(c) if c else None

    p = pick(lambda x: x["condition"] == "Cancer" and not x["surgery"])
    if p:
        p["vitals"][-1]["temp"] = 37.8
        p["watch_flags"].append("최근 체온 37.8℃ (항암 치료 중)")
    p = pick(lambda x: x["condition"] == "Asthma")
    if p:
        p["vitals"][-1]["spo2"] = 93
        p["watch_flags"].append("산소포화도 낮은 편 (93%)")
    p = pick(lambda x: x["condition"] == "Hypertension")
    if p:
        p["vitals"][-1]["bp_sys"], p["vitals"][-1]["bp_dia"] = 158, 96
        p["watch_flags"].append("혈압 높은 편 (158/96)")
    p = pick(lambda x: x["condition"] == "Diabetes")
    if p:
        p["vitals"][-2 if len(p["vitals"]) > 1 else -1]["glucose"] = 64
        p["watch_flags"].append("저혈당 범위 수치 기록 (64 mg/dL)")


def build(rows, today, seed):
    rng = random.Random(seed)
    heroes = hero_patients(today)
    reserved_names = {p["name"] for p in heroes}
    for info in WARD_INFO.values():
        reserved_names.update(info["nurses"])
        reserved_names.update(d.split()[0] for d in info["docs"])
    names = NameMaker(rng, reserved_names)
    alloc = RoomAllocator(rng, {(502, 2), (301, 2)})
    picked = pick_rows(rows, rng)
    patients = list(heroes)
    n = 3
    for cond in CONDITIONS:
        for bin_idx, r in enumerate(picked[cond]):
            gender_ko = GENDER_KO[r["Gender"]]
            if cond == "Diabetes":
                tpl = tmpl_diabetes(rng, today)
            elif cond == "Hypertension":
                tpl = tmpl_hypertension(rng, today)
            elif cond == "Asthma":
                tpl = tmpl_asthma(rng, today)
            elif cond == "Obesity":
                tpl = tmpl_obesity(rng, today)
            elif cond == "Arthritis":
                tpl = tmpl_arthritis(rng, today, bin_idx, gender_ko)
            else:
                tpl = tmpl_cancer(rng, today, bin_idx, gender_ko)
            pod = tpl["surgery"]["post_op_day"] if tpl["surgery"] else 0
            admitted_offset = (pod + rng.randint(1, 2)) if pod else rng.randint(0, 5)
            exp_offset = max(r["_los"] - admitted_offset, 0)
            exp_offset = min(max(exp_offset, 1), 10) if exp_offset else rng.randint(1, 4)
            patients.append(finish_patient(
                f"p{n}", names.make(gender_ko), r["_age"], gender_ko, r["Blood Type"], cond,
                admitted_offset, exp_offset, r["Admission Type"], tpl, today, rng, alloc, n,
                r["Test Results"],
                {"method": "csv_derived", "csv_row": r["_idx"],
                 "csv_fields_used": ["Age", "Gender", "Blood Type", "Medical Condition", "Admission Type", "Test Results"]},
            ))
            n += 1
    apply_flags(patients, rng)
    return patients


# ---------------------------------------------------------------- 외부 모델용 최소 정보 뷰
def to_llm_view(p, today):
    """이름, 병실/침상, 간호사·의사 이름, 절대 날짜를 제외한 최소 정보 뷰."""
    adm = parse_d(p["admitted_at"])
    sched = []
    for s in p["schedule"]:
        dt = parse_dt(s["datetime"])
        if dt.date() >= today:
            sched.append({"when": rel_when(dt, today), "item": s["item"], "fasting": s["fasting"], "note": s["note"]})
    vit = []
    for v in p["vitals"][-3:]:
        item = {k: v[k] for k in v if k != "time"}
        item["when"] = rel_when(parse_dt(v["time"]), today)
        vit.append(item)
    return {
        "patient_ref": p["id"], "age_band": f"{p['age'] // 10 * 10}대", "gender": p["gender"],
        "hospital_day": (today - adm).days + 1,
        "ward": p["ward"], "admission_type": p["admission_type"], "diagnosis": p["diagnosis"],
        "surgery": ({"name": p["surgery"]["name"], "post_op_day": p["surgery"]["post_op_day"]}
                    if p["surgery"] else None),
        "medications": [{k: m[k] for k in ("name", "schedule", "purpose")} for m in p["medications"]],
        "allergies": p["allergies"] if p["allergy_checked"] else "미확인",
        "upcoming_schedule": sched, "recent_vitals": vit,
        "care_notes": p["care_notes"], "diet": p["diet"], "mobility": p["mobility"], "fall_risk": p["fall_risk"],
        "watch_flags": p["watch_flags"], "last_test_summary": p["last_test_summary"],
        "days_until_expected_discharge": (parse_d(p["expected_discharge"]) - today).days,
    }


# ---------------------------------------------------------------- 검증
def validate(patients, today):
    errs = []
    if len(patients) != 32:
        errs.append(f"환자 수 {len(patients)} != 32")
    if len({p["id"] for p in patients}) != len(patients):
        errs.append("id 중복")
    if len({p["name"] for p in patients}) != len(patients):
        errs.append("이름 중복")
    if len({(p["ward"], p["room"], p["bed"]) for p in patients}) != len(patients):
        errs.append("병실/침상 중복")
    for p in patients:
        if p["age"] < 19:
            errs.append(f"{p['id']} 미성년자")
        if p.get("synthetic") is not True:
            errs.append(f"{p['id']} synthetic 표시 없음")
        if FORBIDDEN_KEYS & set(p.keys()):
            errs.append(f"{p['id']} 금지 필드 존재")
        if parse_d(p["expected_discharge"]) <= today:
            errs.append(f"{p['id']} 퇴원 예정일이 오늘 이전")
        if parse_d(p["admitted_at"]) > today:
            errs.append(f"{p['id']} 입원일이 미래")
        if not p["vitals"] or any(parse_dt(v["time"]).date() < parse_d(p["admitted_at"]) for v in p["vitals"]):
            errs.append(f"{p['id']} 활력징후가 없거나 입원 이전 시각")
        if p["surgery"] and parse_d(p["surgery"]["date"]) < parse_d(p["admitted_at"]):
            errs.append(f"{p['id']} 수술일이 입원일보다 빠름")
        view = json.dumps(to_llm_view(p, today), ensure_ascii=False)
        for secret in (p["name"], p["room"], p["assigned_nurse"], p["attending_physician"], p["admitted_at"]):
            if secret in view:
                errs.append(f"{p['id']} LLM 뷰에 식별 정보 포함: {secret}")
    by = {p["id"]: p for p in patients}
    p1, p2 = by["p1"], by["p2"]
    if not any(s["fasting"] and parse_dt(s["datetime"]).date() == today + timedelta(days=1) for s in p1["schedule"]):
        errs.append("p1 내일 금식 검사 없음")
    if not (p2["surgery"] and p2["surgery"]["post_op_day"] == 2 and p2["vitals"][-1]["temp"] == 37.6):
        errs.append("p2 시나리오 설정 불일치")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (기본: 한국 시간 오늘)")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()
    today = parse_d(a.today) if a.today else datetime.now(KST).date()
    rows = load_rows(a.csv)
    patients = build(rows, today, a.seed)
    errs = validate(patients, today)
    if errs:
        print("검증 실패:\n - " + "\n - ".join(errs))
        sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "patients.json"), "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)
    counts = {}
    for p in patients:
        counts[p["condition"]] = counts.get(p["condition"], 0) + 1
    meta = {
        "schema_version": SCHEMA_VERSION, "base_date": today.isoformat(), "seed": a.seed,
        "count": len(patients), "by_condition": counts,
        "source": {"file_name": os.path.basename(a.csv), "usable_rows_after_cleaning": len(rows),
                   "note": "원본 CSV는 저장소에 포함하지 않는다. 출처와 이용 조건은 팀이 별도로 확인해 기록할 것."},
        "privacy": [
            "실제 환자 정보를 사용하지 않았다. 모든 레코드는 합성(synthetic) 데이터다.",
            "원본 CSV의 이름/의사/병원/보험/청구액/병실 번호/약물은 사용하지 않았다(이름은 임의 가명).",
            "CSV에서 가져온 값은 나이, 성별, 혈액형, 질환 분류, 입원 유형, 검사 결과 라벨뿐이다.",
            "만 19세 미만은 제외했다. 연락처/주민번호/주소/생년월일 필드는 만들지 않았다.",
            "병동/병실/수술/약물/일정/활력징후/케어 사항은 규칙 기반으로 합성한 값이며 임상 검수를 받지 않았다.",
            "외부 모델에는 to_llm_view()로 만든 최소 정보 뷰만 전달한다(이름, 병실, 담당자 이름, 절대 날짜 제외).",
        ],
        "regenerate": "python3 -I build_patients.py --csv <healthcare_dataset.csv> --out <폴더> --today YYYY-MM-DD",
    }
    with open(os.path.join(a.out, "patients.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"OK: {len(patients)}명 생성 (기준일 {today}) → {a.out}")
    print("질환별:", counts)
    print("수술 후 환자:", [(p["id"], p["surgery"]["name"], p["surgery"]["post_op_day"]) for p in patients if p["surgery"]])
    print("주의 플래그:", [(p["id"], p["watch_flags"]) for p in patients if p["watch_flags"]])


if __name__ == "__main__":
    main()
