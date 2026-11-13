"""Indian state/UT metadata and tweet-to-state matching for Lok Sabha 2019."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns use word boundaries; order matters for overlapping regions (e.g. Hyderabad → Telangana).
STATE_PATTERNS: dict[str, list[str]] = {
    "Andhra Pradesh": [
        r"\bandhra pradesh\b",
        r"\bandhra\b",
        r"\bamaravati\b",
        r"\bvisakhapatnam\b",
        r"\bvijayawada\b",
        r"\bguntur\b",
        r"\bkurnool\b",
        r"\btirupati\b",
        r"\bnellore\b",
        r"\bchittoor\b",
        r"\bkadapa\b",
        r"\banantapur\b",
        r"\bysrcp\b",
        r"\bjagan\b",
        r"\bjaganmohan\b",
        r"\bchandrababu\b",
        r"\bnaidu\b",
        r"\btdp\b",
        r"\bpawan kalyan\b",
        r"\bjanasena\b",
    ],
    "Telangana": [
        r"\btelangana\b",
        r"\bhyderabad\b",
        r"\bsecunderabad\b",
        r"\bwarangal\b",
        r"\bnizamabad\b",
        r"\bkhammam\b",
        r"\bkarimnagar\b",
        r"\bkcr\b",
        r"\btrs\b",
        r"\bk chandrashekar\b",
        r"\bkchandrashekhar\b",
        r"\bkavitha\b",
        r"\baimim\b",
        r"\basaduddin\b",
        r"\bowaisi\b",
    ],
    "Karnataka": [
        r"\bkarnataka\b",
        r"\bbengaluru\b",
        r"\bbangalore\b",
        r"\bmysuru\b",
        r"\bmysore\b",
        r"\bhubli\b",
        r"\bdharwad\b",
        r"\bmangaluru\b",
        r"\bbelagavi\b",
        r"\bbjp karnataka\b",
        r"\bjds\b",
        r"\bkumaraswamy\b",
        r"\byediyurappa\b",
        r"\bsiddaramaiah\b",
        r"\blingayat\b",
    ],
    "Tamil Nadu": [
        r"\btamil nadu\b",
        r"\btamilnadu\b",
        r"\bchennai\b",
        r"\bmadurai\b",
        r"\bcoimbatore\b",
        r"\btrichy\b",
        r"\btiruchirappalli\b",
        r"\bthoothukudi\b",
        r"\btuthukudi\b",
        r"\bmk stalin\b",
        r"\bstalin\b",
        r"\bdmk\b",
        r"\baiadmk\b",
        r"\bjayalalitha\b",
        r"\bops\b",
        r"\beps\b",
        r"\brajinikanth\b",
    ],
    "Kerala": [
        r"\bkerala\b",
        r"\bthiruvananthapuram\b",
        r"\bkochi\b",
        r"\bkozhikode\b",
        r"\bldf\b",
        r"\budf\b",
        r"\bcpim\b",
        r"\bpinarayi\b",
        r"\brahul gandhi wayanad\b",
        r"\bwayanad\b",
    ],
    "Maharashtra": [
        r"\bmaharashtra\b",
        r"\bmumbai\b",
        r"\bpune\b",
        r"\bnagpur\b",
        r"\bnashik\b",
        r"\bthane\b",
        r"\bshiv sena\b",
        r"\buddhav thackeray\b",
        r"\braj thackeray\b",
        r"\bmns\b",
        r"\bncp\b",
        r"\bsharad pawar\b",
        r"\bajit pawar\b",
        r"\bfadnavis\b",
        r"\bdevendra fadnavis\b",
    ],
    "Gujarat": [
        r"\bgujarat\b",
        r"\bahmedabad\b",
        r"\bsurat\b",
        r"\bvadodara\b",
        r"\brajkot\b",
        r"\bhardik patel\b",
        r"\bvijay rupani\b",
    ],
    "Rajasthan": [
        r"\brajasthan\b",
        r"\bjaipur\b",
        r"\bjodhpur\b",
        r"\budaipur\b",
        r"\bkota\b",
        r"\bashok gehlot\b",
        r"\bsachin pilot\b",
        r"\bvasundhara\b",
    ],
    "Uttar Pradesh": [
        r"\buttar pradesh\b",
        r"\buttarpradesh\b",
        r"\blucknow\b",
        r"\bkanpur\b",
        r"\bvaranasi\b",
        r"\bprayagraj\b",
        r"\ballahabad\b",
        r"\bgorakhpur\b",
        r"\bmeerut\b",
        r"\bayodhya\b",
        r"\byogi\b",
        r"\badityanath\b",
        r"\bakhilesh\b",
        r"\bmayawati\b",
        r"\bbsp\b",
        r"\bsamajwadi\b",
        r"\bsultanpur\b",
    ],
    "Bihar": [
        r"\bbihar\b",
        r"\bpatna\b",
        r"\bgaya\b",
        r"\bmuzaffarpur\b",
        r"\bnitish\b",
        r"\bkumar\b",
        r"\bjdu\b",
        r"\blalu\b",
        r"\btejashwi\b",
        r"\brjd\b",
        r"\bpaswan\b",
    ],
    "West Bengal": [
        r"\bwest bengal\b",
        r"\bbengal\b",
        r"\bkolkata\b",
        r"\bhowrah\b",
        r"\bdarjeeling\b",
        r"\bmamata\b",
        r"\bbanerjee\b",
        r"\btmc\b",
        r"\btrinamool\b",
        r"\bdidi\b",
    ],
    "Odisha": [
        r"\bodisha\b",
        r"\borissa\b",
        r"\bbhubaneswar\b",
        r"\bcuttack\b",
        r"\bnaveen patnaik\b",
        r"\bbjd\b",
    ],
    "Punjab": [
        r"\bpunjab\b",
        r"\bamritsar\b",
        r"\bludhiana\b",
        r"\bjalandhar\b",
        r"\bchandigarh punjab\b",
        r"\bamarinder\b",
        r"\bbadal\b",
        r"\baap punjab\b",
    ],
    "Haryana": [
        r"\bharyana\b",
        r"\bgurgaon\b",
        r"\bgurugram\b",
        r"\bfaridabad\b",
        r"\brohtak\b",
        r"\bhooda\b",
        r"\bkhattar\b",
    ],
    "Madhya Pradesh": [
        r"\bmadhya pradesh\b",
        r"\bbhopal\b",
        r"\bindore\b",
        r"\bjabalpur\b",
        r"\bgwalior\b",
        r"\bkamal nath\b",
        r"\bshivraj\b",
        r"\bchouhan\b",
        r"\bscindia\b",
    ],
    "Chhattisgarh": [
        r"\bchhattisgarh\b",
        r"\braipur\b",
        r"\bbhilai\b",
        r"\bbhupesh\b",
        r"\bbaghel\b",
    ],
    "Jharkhand": [
        r"\bjharkhand\b",
        r"\branchi\b",
        r"\bjamshedpur\b",
        r"\bdhanbad\b",
        r"\bhemant soren\b",
        r"\bjmm\b",
    ],
    "Assam": [
        r"\bassam\b",
        r"\bguwahati\b",
        r"\bdibrugarh\b",
        r"\bsarbananda\b",
        r"\bsonowal\b",
        r"\bajf\b",
        r"\bulf\b",
    ],
    "Delhi": [
        r"\bdelhi\b",
        r"\bnew delhi\b",
        r"\bkejriwal\b",
        r"\baap\b",
        r"\barvind\b",
        r"\bmanish sisodia\b",
        r"\blutyens\b",
        r"\bseven seats delhi\b",
    ],
    "Goa": [
        r"\bgoa\b",
        r"\bpanaji\b",
        r"\bpanjim\b",
    ],
    "Himachal Pradesh": [
        r"\bhimachal\b",
        r"\bshimla\b",
        r"\bdharamshala\b",
        r"\bjairam thakur\b",
    ],
    "Uttarakhand": [
        r"\buttarakhand\b",
        r"\buttaranchal\b",
        r"\bdehradun\b",
        r"\bharidwar\b",
        r"\bnainital\b",
    ],
    "Jammu and Kashmir": [
        r"\bjammu and kashmir\b",
        r"\bjammu\b",
        r"\bkashmir\b",
        r"\bsrinagar\b",
        r"\bmehbooba\b",
        r"\bfarooq abdullah\b",
        r"\bnc\b",
        r"\bpdp\b",
    ],
    "Ladakh": [
        r"\bladakh\b",
        r"\bleh\b",
        r"\bkargil\b",
    ],
    "Manipur": [
        r"\bmanipur\b",
        r"\bimphal\b",
    ],
    "Meghalaya": [
        r"\bmeghalaya\b",
        r"\bshillong\b",
    ],
    "Mizoram": [
        r"\bmizoram\b",
        r"\baizawl\b",
    ],
    "Nagaland": [
        r"\bnagaland\b",
        r"\bkohima\b",
    ],
    "Tripura": [
        r"\btripura\b",
        r"\bagartala\b",
        r"\bbiplab deb\b",
    ],
    "Arunachal Pradesh": [
        r"\barunachal\b",
        r"\bitanagar\b",
    ],
    "Sikkim": [
        r"\bsikkim\b",
        r"\bgangtok\b",
    ],
    "Puducherry": [
        r"\bpuducherry\b",
        r"\bpondicherry\b",
    ],
    "Chandigarh": [
        r"\bchandigarh\b",
    ],
    "Andaman and Nicobar": [
        r"\bandaman\b",
        r"\bnicobar\b",
    ],
    "Lakshadweep": [
        r"\blakshadweep\b",
    ],
    "Dadra and Nagar Haveli": [
        r"\bdadra\b",
        r"\bnagar haveli\b",
    ],
    "Daman and Diu": [
        r"\bdaman\b",
        r"\bdiu\b",
    ],
}

# Compiled patterns for speed
_COMPILED = {
    state: [re.compile(p, re.IGNORECASE) for p in patterns]
    for state, patterns in STATE_PATTERNS.items()
}

STATE_ORDER = list(STATE_PATTERNS.keys())


@dataclass(frozen=True)
class StateElectionInfo:
    name: str
    region: str
    lok_sabha_seats: int
    election_phase: str
    major_parties: str
    key_leaders: str
    result_2019: str
    notes: str


ELECTION_INFO: dict[str, StateElectionInfo] = {
    "Andhra Pradesh": StateElectionInfo(
        "Andhra Pradesh", "South", 25, "Phase 1 (11 Apr)",
        "YSRCP, TDP, BJP, Janasena",
        "Jagan Mohan Reddy, Chandrababu Naidu, Pawan Kalyan",
        "YSRCP won 22/25 seats (landslide)",
        "Highly competitive state; TDP broke alliance with BJP before polls.",
    ),
    "Telangana": StateElectionInfo(
        "Telangana", "South", 17, "Phase 1 (11 Apr)",
        "TRS, BJP, Congress, AIMIM",
        "KCR (K Chandrashekar Rao), Asaduddin Owaisi",
        "TRS won 9/17; BJP gained seats",
        "Regional party TRS dominated; Hyderabad seat saw AIMIM hold.",
    ),
    "Karnataka": StateElectionInfo(
        "Karnataka", "South", 28, "Phase 2 & 7",
        "BJP, Congress, JD(S)",
        "Yediyurappa, Siddaramaiah, Kumaraswamy",
        "BJP won 25/28 seats",
        "Strong Modi wave; Lingayat community support key.",
    ),
    "Tamil Nadu": StateElectionInfo(
        "Tamil Nadu", "South", 39, "Phase 2 (18 Apr)",
        "DMK alliance, AIADMK+BJP",
        "MK Stalin, EPS, OPS",
        "DMK-led alliance won 38/39",
        "State parties dominated; national parties in alliances.",
    ),
    "Kerala": StateElectionInfo(
        "Kerala", "South", 20, "Phase 3 (23 Apr)",
        "LDF, UDF",
        "Pinarayi Vijayan, Rahul Gandhi (Wayanad)",
        "UDF won 19/20; Rahul won Wayanad",
        "Rahul Gandhi contested Wayanad alongside Amethi.",
    ),
    "Maharashtra": StateElectionInfo(
        "Maharashtra", "West", 48, "Phase 3 & 4",
        "BJP-Shiv Sena, Congress-NCP",
        "Modi, Uddhav Thackeray, Sharad Pawar",
        "BJP+Shiv Sena won 41/48",
        "Largest LS contributor; coalition arithmetic crucial.",
    ),
    "Gujarat": StateElectionInfo(
        "Gujarat", "West", 26, "Phase 3 (23 Apr)",
        "BJP, Congress",
        "Modi (from Gujarat), Vijay Rupani",
        "BJP swept 26/26",
        "Modi's home state; clean sweep for BJP.",
    ),
    "Rajasthan": StateElectionInfo(
        "Rajasthan", "North", 25, "Phase 4 & 5",
        "BJP, Congress",
        "Ashok Gehlot, Sachin Pilot, Vasundhara Raje",
        "BJP won 24/25",
        "Congress lost state assembly in 2018 but LS went BJP.",
    ),
    "Uttar Pradesh": StateElectionInfo(
        "Uttar Pradesh", "North", 80, "Phase 1–7",
        "BJP, SP-BSP alliance, Congress",
        "Modi (Varanasi), Yogi, Akhilesh, Mayawati",
        "BJP+allies won 64/80",
        "Most seats; Varanasi was PM's constituency.",
    ),
    "Bihar": StateElectionInfo(
        "Bihar", "East", 40, "Phase 1–7",
        "NDA (BJP+JDU), Mahagathbandhan",
        "Nitish Kumar, Tejashwi Yadav, Lalu Prasad",
        "NDA won 39/40",
        "Nitish Kumar with BJP; near clean sweep.",
    ),
    "West Bengal": StateElectionInfo(
        "West Bengal", "East", 42, "Phase 4–7",
        "TMC, BJP, Left, Congress",
        "Mamata Banerjee, Modi",
        "TMC won 22; BJP surged to 18",
        "Fierce TMC vs BJP battle; BJP became main opposition.",
    ),
    "Odisha": StateElectionInfo(
        "Odisha", "East", 21, "Phase 1–4",
        "BJD, BJP, Congress",
        "Naveen Patnaik",
        "BJD won 12; BJP won 8",
        "Regional party BJD held ground.",
    ),
    "Punjab": StateElectionInfo(
        "Punjab", "North", 13, "Phase 7 (19 May)",
        "Congress, SAD-BJP, AAP",
        "Amarinder Singh, Badals",
        "Congress won 8/13",
        "Congress stronghold in 2019 LS polls.",
    ),
    "Haryana": StateElectionInfo(
        "Haryana", "North", 10, "Phase 6 (12 May)",
        "BJP, Congress, JJP",
        "Khattar, Hooda",
        "BJP won 10/10",
        "Clean sweep for BJP.",
    ),
    "Madhya Pradesh": StateElectionInfo(
        "Madhya Pradesh", "Central", 29, "Phase 4–6",
        "BJP, Congress",
        "Shivraj Chouhan, Kamal Nath",
        "BJP won 28/29",
        "Congress had won assembly 2018 but lost LS badly.",
    ),
    "Chhattisgarh": StateElectionInfo(
        "Chhattisgarh", "Central", 11, "Phase 2 & 3",
        "BJP, Congress",
        "Bhupesh Baghel, Raman Singh",
        "BJP won 9/11",
        "Congress state govt but BJP won most LS seats.",
    ),
    "Jharkhand": StateElectionInfo(
        "Jharkhand", "East", 14, "Phase 4–5",
        "BJP, JMM-Congress",
        "Hemant Soren",
        "BJP won 11/14",
        "BJP dominated tribal belt seats.",
    ),
    "Assam": StateElectionInfo(
        "Assam", "Northeast", 14, "Phase 1–3",
        "BJP, Congress, AIUDF",
        "Sarbananda Sonowal",
        "BJP won 9/14",
        "Northeast gateway state for BJP.",
    ),
    "Delhi": StateElectionInfo(
        "Delhi", "North", 7, "Phase 6 (12 May)",
        "BJP, AAP, Congress",
        "Kejriwal, Modi",
        "BJP won 7/7",
        "AAP did not contest LS; BJP sweep.",
    ),
    "Goa": StateElectionInfo(
        "Goa", "West", 2, "Phase 3 (23 Apr)",
        "BJP, Congress, GFP",
        "Manohar Parrikar legacy",
        "BJP won 2/2",
        "Small state, high national visibility.",
    ),
    "Himachal Pradesh": StateElectionInfo(
        "Himachal Pradesh", "North", 4, "Phase 7 (19 May)",
        "BJP, Congress",
        "Jairam Thakur",
        "BJP won 4/4",
        "Hill state; BJP sweep.",
    ),
    "Uttarakhand": StateElectionInfo(
        "Uttarakhand", "North", 5, "Phase 1 (11 Apr)",
        "BJP, Congress",
        "Trivendra Rawat",
        "BJP won 5/5",
        "Modi rally state; BJP sweep.",
    ),
    "Jammu and Kashmir": StateElectionInfo(
        "Jammu and Kashmir", "North", 6, "Phase 1 (11 Apr)",
        "NC, PDP, BJP, Congress",
        "Mehbooba Mufti, Farooq Abdullah",
        "BJP won 2; NC 3; PDP 0",
        "Low turnout; Article 370 debate dominated.",
    ),
    "Ladakh": StateElectionInfo(
        "Ladakh", "North", 1, "Phase 5 (6 May)",
        "BJP, Congress",
        "Independent / BJP",
        "BJP won 1/1",
        "New UT; single constituency.",
    ),
    "Manipur": StateElectionInfo("Manipur", "Northeast", 2, "Phase 1", "BJP, Congress", "N Biren Singh", "BJP won 1; Congress 1", ""),
    "Meghalaya": StateElectionInfo("Meghalaya", "Northeast", 2, "Phase 1", "NDP, Congress", "Conrad Sangma", "Split 1-1", ""),
    "Mizoram": StateElectionInfo("Mizoram", "Northeast", 1, "Phase 1", "MNF, Congress", "Zoramthanga", "MNF won 1/1", ""),
    "Nagaland": StateElectionInfo("Nagaland", "Northeast", 1, "Phase 1", "NDPP-BJP", "Neiphiu Rio", "NDPP won 1/1", ""),
    "Tripura": StateElectionInfo("Tripura", "Northeast", 2, "Phase 1", "BJP, Left", "Biplab Deb", "BJP won 2/2", "BJP won state in 2018."),
    "Arunachal Pradesh": StateElectionInfo("Arunachal Pradesh", "Northeast", 2, "Phase 1", "BJP", "Pema Khandu", "BJP won 2/2", ""),
    "Sikkim": StateElectionInfo("Sikkim", "Northeast", 1, "Phase 1", "SKM, SDF", "Prem Singh Tamang", "SKM won 1/1", ""),
    "Puducherry": StateElectionInfo("Puducherry", "South", 1, "Phase 2", "Congress, AINRC", "Narayanasamy", "Congress won 1/1", ""),
    "Chandigarh": StateElectionInfo("Chandigarh", "North", 1, "Phase 7", "BJP, Congress", "Kirron Kher", "BJP won 1/1", "Union Territory."),
    "Andaman and Nicobar": StateElectionInfo("Andaman and Nicobar", "Islands", 1, "Phase 1", "BJP, Congress", "—", "BJP won 1/1", ""),
    "Lakshadweep": StateElectionInfo("Lakshadweep", "Islands", 1, "Phase 1", "Congress, NCP", "—", "Congress won 1/1", ""),
    "Dadra and Nagar Haveli": StateElectionInfo("Dadra and Nagar Haveli", "West", 1, "Phase 3", "BJP", "—", "BJP won 1/1", "Now merged with Daman & Diu."),
    "Daman and Diu": StateElectionInfo("Daman and Diu", "West", 1, "Phase 3", "BJP", "—", "BJP won 1/1", ""),
}


def detect_states(text: str) -> list[str]:
    """Return all states/UTs mentioned in tweet text."""
    if not text or not str(text).strip():
        return []
    found = []
    for state in STATE_ORDER:
        if any(p.search(text) for p in _COMPILED[state]):
            found.append(state)
    return found


def primary_state(states: list[str]) -> str | None:
    return states[0] if states else None
