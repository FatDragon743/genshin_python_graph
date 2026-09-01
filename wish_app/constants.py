"""卡池与路径常量。"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# 展示用 Sheet 名 -> 官方 gacha_type（301/400 合并为角色活动）
SHEET_TO_GACHA_TYPES = {
    "角色活动祈愿": (301, 400),
    "武器活动祈愿": (302,),
    "常驻祈愿": (200,),
    "集录祈愿": (500,),
    "新手祈愿": (100,),
}

GACHA_TYPE_TO_SHEET = {}
for sheet, types in SHEET_TO_GACHA_TYPES.items():
    for t in types:
        GACHA_TYPE_TO_SHEET[t] = sheet

SHEET_ORDER = list(SHEET_TO_GACHA_TYPES.keys())

# 主表列（与现有导出兼容）
MAIN_COLUMNS = ["时间", "名称", "类别", "星级", "总次数", "保底内", "备注"]

# 内部 raw 表列
RAW_COLUMNS = [
    "id",
    "uid",
    "gacha_type",
    "time",
    "name",
    "item_type",
    "rank_type",
    "lang",
]

RAW_SHEET = "__raw"
META_SHEET = "__meta"

STANDARD_CHARS = {"刻晴", "迪卢克", "莫娜", "琴", "七七", "提纳里", "迪希雅"}

# 常驻五星武器（武器池/常驻池用来区分限定）
STANDARD_WEAPONS = {
    "风鹰剑",
    "天空之刃",
    "天空之卷",
    "天空之翼",
    "天空之脊",
    "天空之傲",
    "和璞鸢",
    "四风原典",
    "狼的末路",
    "阿莫斯之弓",
}

# 欧非分析展示的卡池
LUCK_POOLS = [
    {"key": "character", "title": "角色池", "types": (301, 400), "track_5050": True},
    {"key": "weapon", "title": "武器池", "types": (302,), "track_5050": True},
    {"key": "permanent", "title": "常驻池", "types": (200,), "track_5050": False},
    {"key": "chronicled", "title": "集录池", "types": (500,), "track_5050": True},
]

# 各池五星出金规则（社区数据拟合：软保底起点与每抽增量）
# 角色/常驻/集录：基础 0.6%，约 74 抽起软保，90 硬保
# 武器：基础 0.7%，约 63 抽起软保，80 硬保
POOL_PITY_RULES = {
    "character": {
        "title": "角色活动祈愿",
        "base_rate": 0.006,
        "soft_pity": 74,
        "hard_pity": 90,
        "rate_increase": 0.06,
        "track_5050": True,
        "featured_note": "小保底 50% UP；歪后下次大保底必出 UP（另有捕获明光，此处未建模）",
        "early_cutoff": 60,
    },
    "weapon": {
        "title": "武器活动祈愿",
        "base_rate": 0.007,
        "soft_pity": 63,
        "hard_pity": 80,
        "rate_increase": 0.07,
        "track_5050": True,
        "featured_note": "出金约 75% 为当期 UP 武器；神铸定轨命定值满后必出定轨（5.0 后满值多为 1）",
        "early_cutoff": 50,
    },
    "permanent": {
        "title": "常驻祈愿",
        "base_rate": 0.006,
        "soft_pity": 74,
        "hard_pity": 90,
        "rate_increase": 0.06,
        "track_5050": False,
        "featured_note": "无 UP / 无大小保底，五星在常驻角色与武器中随机",
        "early_cutoff": 60,
    },
    "chronicled": {
        "title": "集录祈愿",
        "base_rate": 0.006,
        "soft_pity": 74,
        "hard_pity": 90,
        "rate_increase": 0.06,
        "track_5050": True,
        "featured_note": "指定目标后 50% 出指定；未中得 1 命定值，满 1 点下次必出指定（本期有效）",
        "early_cutoff": 60,
    },
}

# 四星（或以上）十抽保底：各池通用结构，基础率武器略高
# 社区拟合：1–8 抽基础率，第 9 抽抬升，第 10 抽必出四星及以上
FOUR_STAR_RULES = {
    "character": {
        "base_rate": 0.051,
        "soft_pity": 9,
        "soft_rate": 0.561,
        "hard_pity": 10,
        "note": "每 10 抽必出四星及以上；活动池四星另有 50% UP / 歪后下次必 UP",
    },
    "weapon": {
        "base_rate": 0.06,
        "soft_pity": 9,
        "soft_rate": 0.561,
        "hard_pity": 10,
        "note": "武器池四星基础约 6%；十抽保底结构同角色池",
    },
    "permanent": {
        "base_rate": 0.051,
        "soft_pity": 9,
        "soft_rate": 0.561,
        "hard_pity": 10,
        "note": "常驻池四星无 UP，十抽必出四星及以上",
    },
    "chronicled": {
        "base_rate": 0.051,
        "soft_pity": 9,
        "soft_rate": 0.561,
        "hard_pity": 10,
        "note": "集录池四星规则同角色活动池（含 UP 机制）",
    },
}

# 近期运势：最近 N 个五星
RECENT_FIVE_STAR_WINDOW = 10

# 国服抽卡 API（biuuu/genshin-wish-export、TeyvatGuide 现行域名）
GACHA_API = "https://public-operation-hk4e.mihoyo.com/gacha_info/api/getGachaLog"
GEN_AUTHKEY_URL = "https://api-takumi.mihoyo.com/binding/api/genAuthKey"
GAME_ROLES_URL = "https://api-takumi.mihoyo.com/binding/api/getUserGameRolesByCookie"
MULTI_TOKEN_URL = "https://api-takumi.mihoyo.com/auth/api/getMultiTokenByLoginTicket"  # + query
LOGIN_BY_COOKIE_URL = "https://webapi.account.mihoyo.com/Api/login_by_cookie"

MIHOYO_LOGIN_URL = "https://user.mihoyo.com/#/login/password"
SESSION_PATH = DATA_DIR / "session.json"

# 拉取顺序
FETCH_GACHA_TYPES = [301, 400, 302, 200, 500, 100]
