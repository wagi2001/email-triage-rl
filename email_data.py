# email_data.py
from __future__ import annotations

import random

# Triage actions (scored by verifier). "open" reveals sender in env step.
TRIAGE_ACTIONS = ["delete", "archive", "label:newsletter", "skip"]
OPEN_ACTION   = "open"
EXPAND_ACTION = "expand"    # reveal thread messages after opening
PAGE_ACTION   = "next_page" # advance inbox to the next page
ACTION_NAMES  = TRIAGE_ACTIONS + [OPEN_ACTION, EXPAND_ACTION, PAGE_ACTION]

# Snippets used when generating thread messages
THREAD_OPENER_SNIPPETS = [
    "Please see the details below.",
    "Wanted to bring this to your attention.",
    "Quick update on the above.",
    "Following up from our last conversation.",
    "Please review at your earliest convenience.",
    "Sharing for your reference.",
    "Action required — see below.",
    "FYI — no action needed.",
    "Looping you in on this.",
]

THREAD_REPLY_SNIPPETS = [
    "Thanks, I'll follow up on this.",
    "Sounds good, I'll take a look.",
    "Got it — will handle by EOD.",
    "Agreed, let's proceed.",
    "Let me check and get back to you.",
    "Any updates on this?",
    "Following up on the below.",
    "Can we discuss in our next sync?",
    "Just saw this, on it now.",
    "Makes sense, thanks for the context.",
]

SENDERS = [
    ("Alice Chen", "alice@work.com"),
    ("Bob Martinez", "bob@work.com"),
    ("Newsletter", "news@techdigest.com"),
    ("Promo Bot", "promo@sale.com"),
    ("GitHub", "noreply@github.com"),
    ("Spam King", "deals@spammy.net"),
    ("Manager Dave", "dave@work.com"),
    ("LinkedIn", "notifications@linkedin.com"),
]

# Per-target lookalikes: addresses that look almost identical to each rule target.
# They share the same subject pool so the only distinguishing feature is the
# sender address — exactly the precision test that trips up LLMs.
LOOKALIKES: dict[str, list[tuple[str, str]]] = {
    "promo@sale.com": [
        ("Promo Team",   "promo@sale.net"),
        ("Sale Alerts",  "promo@sales.com"),
        ("Promos Inc",   "promos@sale.com"),
        ("Sale Promo",   "sale@promo.com"),
        ("Promo Offers", "offers@sale.com"),
        ("Promo Shop",   "promo@sale.org"),
        ("SaleBot",      "promo@mysale.com"),
    ],
    "noreply@github.com": [
        ("GitHub Actions",   "notifications@github.com"),
        ("GitHub Bot",       "github-noreply@github.com"),
        ("GitHub No-Reply",  "no-reply@github.com"),
        ("GitHub Alerts",    "noreply@github.io"),
        ("GitHub Team",      "team@github.com"),
        ("GitHub Mail",      "noreply@github.net"),
        ("GitHub Support",   "support@github.com"),
    ],
    "deals@spammy.net": [
        ("Deals Daily",  "deals@spammy.org"),
        ("Spammy Deals", "deals@spammy.com"),
        ("Mail Deals",   "mail@spammy.net"),
        ("Daily Deals",  "daily@spammy.net"),
        ("Deals Bot",    "deals@spammy.io"),
        ("Spammy Bot",   "noreply@spammy.net"),
    ],
    "news@techdigest.com": [
        ("Tech Digest",    "digest@techdigest.com"),
        ("TechDigest Net", "news@techdigest.net"),
        ("Tech News",      "news@tech-digest.com"),
        ("Digest Daily",   "daily@techdigest.com"),
        ("Tech Weekly",    "weekly@techdigest.com"),
    ],
    "notifications@linkedin.com": [
        ("LinkedIn Mail",    "mail@linkedin.com"),
        ("LinkedIn Alerts",  "alerts@linkedin.com"),
        ("LinkedIn Team",    "team@linkedin.com"),
        ("LinkedIn Updates", "updates@linkedin.com"),
        ("LinkedIn News",    "news@linkedin.com"),
    ],
    "alice@work.com": [
        ("Alice C.",       "alice@works.com"),
        ("A. Chen",        "a.chen@work.com"),
        ("Alice Work",     "alicec@work.com"),
        ("Alicia Chen",    "alicia@work.com"),
    ],
    "bob@work.com": [
        ("Bob M.",         "bob@works.com"),
        ("R. Martinez",    "robert@work.com"),
        ("B. Martinez",    "b.martinez@work.com"),
        ("Bob Work",       "bobm@work.com"),
    ],
    "dave@work.com": [
        ("Dave M.",        "dave@works.com"),
        ("D. Manager",     "dmanager@work.com"),
        ("Dave Work",      "dave@work.net"),
        ("Manager D.",     "manager@work.com"),
    ],
}

# Flat list of all decoy senders (used for fallback / non-expert difficulty)
DECOY_SENDERS = [entry for entries in LOOKALIKES.values() for entry in entries]

# Subject pool: lookalikes share subjects with their real counterpart so that
# the only distinguishing feature is the sender address itself.
_LOOKALIKE_SUBJECTS: dict[str, list[str]] = {}
_TARGET_SUBJECTS: dict[str, list[str]] = {
    "alice@work.com":  ["Q3 report ready", "Quick question", "Meeting notes",
                        "Urgent follow-up", "Data request", "Can you review this?"],
    "bob@work.com":    ["PR review request", "Lunch tomorrow?", "Bug in prod",
                        "Review needed", "Sprint update", "Lunch meeting?"],
    "dave@work.com":   ["All hands Friday", "Performance review", "Team update",
                        "Policy update", "Monthly review", "Friday recap"],
    "promo@sale.com":  ["50% OFF today only!", "Flash sale ends tonight", "You won!",
                        "Last chance deal", "Exclusive members offer", "Members-only sale"],
    "noreply@github.com": ["New issue opened", "PR merged", "CI failed",
                           "Build succeeded", "Review requested", "Dependabot alert"],
    "deals@spammy.net":   ["Congratulations!", "Claim your prize", "Urgent offer",
                           "Special announcement", "You've been selected", "Act now"],
    "news@techdigest.com":["This week in AI", "Top 10 tools", "Dev digest #42",
                           "Weekly roundup", "Tech bytes", "Monthly digest"],
    "notifications@linkedin.com": ["You have 3 new connections", "Someone viewed your profile",
                                   "New message from a connection", "Job alert for you",
                                   "People you may know"],
}
for _target, _entries in LOOKALIKES.items():
    for _, _addr in _entries:
        _LOOKALIKE_SUBJECTS[_addr] = _TARGET_SUBJECTS.get(_target, ["(no subject)"])

SUBJECTS = {
    **_TARGET_SUBJECTS,
    **_LOOKALIKE_SUBJECTS,
}

RULES = [
    {
        "description": "delete all emails from promo@sale.com",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
    },
    {
        "description": "archive all emails from noreply@github.com",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
    },
    {
        "description": "label all emails from news@techdigest.com as 'newsletter'",
        "action": "label:newsletter",
        "filter_key": "sender_email",
        "filter_value": "news@techdigest.com",
    },
    {
        "description": "delete all emails from deals@spammy.net",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "deals@spammy.net",
    },
    {
        "description": "archive all emails from notifications@linkedin.com",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "notifications@linkedin.com",
    },
    {
        "description": "archive all emails from bob@work.com",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "bob@work.com",
    },
    {
        "description": "archive all emails from alice@work.com",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "alice@work.com",
    },
    {
        "description": "label all emails from dave@work.com as newsletter",
        "action": "label:newsletter",
        "filter_key": "sender_email",
        "filter_value": "dave@work.com",
    },
]

# Hard mode: subject / keyword rules (partial match)
HARD_RULES = [
    {
        "description": "delete all emails whose subject contains 'OFF'",
        "action": "delete",
        "filter_key": "subject",
        "filter_value": "OFF",
        "filter_op": "contains",
    },
    {
        "description": "archive all emails whose subject contains 'PR'",
        "action": "archive",
        "filter_key": "subject",
        "filter_value": "PR",
        "filter_op": "contains",
    },
    {
        "description": "label all emails whose subject contains 'digest' as 'newsletter'",
        "action": "label:newsletter",
        "filter_key": "subject",
        "filter_value": "digest",
        "filter_op": "contains",
    },
    {
        "description": "archive all emails whose subject contains 'update'",
        "action": "archive",
        "filter_key": "subject",
        "filter_value": "update",
        "filter_op": "contains",
    },
    {
        "description": "delete all emails whose subject contains 'sale'",
        "action": "delete",
        "filter_key": "subject",
        "filter_value": "sale",
        "filter_op": "contains",
    },
    {
        "description": "archive all emails whose subject contains 'review'",
        "action": "archive",
        "filter_key": "subject",
        "filter_value": "review",
        "filter_op": "contains",
    },
    {
        "description": "label all emails whose subject contains 'report' as newsletter",
        "action": "label:newsletter",
        "filter_key": "subject",
        "filter_value": "report",
        "filter_op": "contains",
    },
]

# Expert mode: compound AND / EXCEPT rules.
# Each generates trap emails that match only the primary condition,
# forcing a model to apply both conditions precisely.
COMPOUND_RULES = [
    # --- AND rules: action only when BOTH conditions are true ---
    {
        "description": "archive emails from noreply@github.com only if subject contains 'PR'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "also_requires": {
            "filter_key": "subject",
            "filter_value": "PR",
            "filter_op": "contains",
        },
    },
    {
        "description": "delete emails from promo@sale.com only if subject contains 'OFF'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "also_requires": {
            "filter_key": "subject",
            "filter_value": "OFF",
            "filter_op": "contains",
        },
    },
    {
        "description": "label emails from news@techdigest.com as newsletter only if subject contains 'digest'",
        "action": "label:newsletter",
        "filter_key": "sender_email",
        "filter_value": "news@techdigest.com",
        "also_requires": {
            "filter_key": "subject",
            "filter_value": "digest",
            "filter_op": "contains",
        },
    },
    # --- EXCEPT rules: action unless exception condition is true ---
    {
        "description": "delete emails from promo@sale.com except if subject contains 'won'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "except_if": {
            "filter_key": "subject",
            "filter_value": "won",
            "filter_op": "contains",
        },
    },
    {
        "description": "archive emails from noreply@github.com except if subject contains 'failed'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "except_if": {
            "filter_key": "subject",
            "filter_value": "failed",
            "filter_op": "contains",
        },
    },
    {
        "description": "delete emails from deals@spammy.net except if subject contains 'prize'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "deals@spammy.net",
        "except_if": {
            "filter_key": "subject",
            "filter_value": "prize",
            "filter_op": "contains",
        },
    },
    # --- Negated AND rules: action only if secondary condition is NOT true ---
    # Counter-intuitive: skip the "obvious" emails, act on the boring ones.
    {
        "description": "archive emails from noreply@github.com only if subject does NOT contain 'merged'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "also_requires": {
            "filter_key": "subject",
            "filter_value": "merged",
            "filter_op": "contains",
            "negate": True,
        },
    },
    {
        "description": "delete emails from promo@sale.com only if subject does NOT contain 'OFF'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "also_requires": {
            "filter_key": "subject",
            "filter_value": "OFF",
            "filter_op": "contains",
            "negate": True,
        },
    },
    # --- Except-if-any rules: multiple exception keywords ---
    {
        "description": "archive emails from noreply@github.com except if subject contains 'failed' or 'issue'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "except_if_any": [
            {"filter_key": "subject", "filter_value": "failed", "filter_op": "contains"},
            {"filter_key": "subject", "filter_value": "issue", "filter_op": "contains"},
        ],
    },
    {
        "description": "delete emails from deals@spammy.net except if subject contains 'prize' or 'congratulations'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "deals@spammy.net",
        "except_if_any": [
            {"filter_key": "subject", "filter_value": "prize", "filter_op": "contains"},
            {"filter_key": "subject", "filter_value": "congratulations", "filter_op": "contains"},
        ],
    },
    # --- Compound rules for work senders ---
    {
        "description": "archive emails from bob@work.com only if subject contains 'review'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "bob@work.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "review", "filter_op": "contains",
        },
    },
    {
        "description": "archive emails from alice@work.com only if subject contains 'report'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "alice@work.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "report", "filter_op": "contains",
        },
    },
    {
        "description": "archive emails from bob@work.com except if subject contains 'lunch'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "bob@work.com",
        "except_if": {
            "filter_key": "subject", "filter_value": "lunch", "filter_op": "contains",
        },
    },
    {
        "description": "archive emails from dave@work.com except if subject contains 'Friday'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "dave@work.com",
        "except_if": {
            "filter_key": "subject", "filter_value": "Friday", "filter_op": "contains",
        },
    },
    # --- Additional negated AND rules ---
    {
        "description": "archive emails from noreply@github.com only if subject does NOT contain 'succeeded'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "succeeded",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "delete emails from deals@spammy.net only if subject does NOT contain 'announcement'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "deals@spammy.net",
        "also_requires": {
            "filter_key": "subject", "filter_value": "announcement",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "archive emails from noreply@github.com only if subject does NOT contain 'Build'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "Build",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "label emails from news@techdigest.com as newsletter only if subject does NOT contain 'digest'",
        "action": "label:newsletter",
        "filter_key": "sender_email",
        "filter_value": "news@techdigest.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "digest",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "archive emails from notifications@linkedin.com only if subject does NOT contain 'viewed'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "notifications@linkedin.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "viewed",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "archive emails from bob@work.com only if subject does NOT contain 'lunch'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "bob@work.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "lunch",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "archive emails from dave@work.com only if subject does NOT contain 'update'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "dave@work.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "update",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "archive emails from alice@work.com only if subject does NOT contain 'question'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "alice@work.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "question",
            "filter_op": "contains", "negate": True,
        },
    },
    {
        "description": "delete emails from promo@sale.com only if subject does NOT contain 'won'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "also_requires": {
            "filter_key": "subject", "filter_value": "won",
            "filter_op": "contains", "negate": True,
        },
    },
    # --- Except-if-any with negated phrasing (double-negation confusion) ---
    {
        "description": "archive emails from noreply@github.com except if subject contains 'merged' or 'succeeded'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "noreply@github.com",
        "except_if_any": [
            {"filter_key": "subject", "filter_value": "merged",    "filter_op": "contains"},
            {"filter_key": "subject", "filter_value": "succeeded", "filter_op": "contains"},
        ],
    },
    {
        "description": "delete emails from promo@sale.com except if subject contains 'OFF' or 'won'",
        "action": "delete",
        "filter_key": "sender_email",
        "filter_value": "promo@sale.com",
        "except_if_any": [
            {"filter_key": "subject", "filter_value": "OFF", "filter_op": "contains"},
            {"filter_key": "subject", "filter_value": "won", "filter_op": "contains"},
        ],
    },
    {
        "description": "archive emails from bob@work.com except if subject contains 'lunch' or 'Bug'",
        "action": "archive",
        "filter_key": "sender_email",
        "filter_value": "bob@work.com",
        "except_if_any": [
            {"filter_key": "subject", "filter_value": "lunch", "filter_op": "contains"},
            {"filter_key": "subject", "filter_value": "Bug",   "filter_op": "contains"},
        ],
    },
]

# Expert pool: ONLY the rules that require precise negation reasoning.
# Simple EXCEPT and AND rules are excluded — the model handles those at 100%.
EXPERT_RULES = [
    r for r in COMPOUND_RULES
    if r.get("also_requires", {}).get("negate") or "except_if_any" in r
]


def _check_condition(email, condition):
    """Whether an email satisfies a filter condition dict."""
    field = email.get(condition["filter_key"], "")
    val = condition["filter_value"]
    if condition.get("filter_op") == "contains":
        result = val.lower() in str(field).lower()
    else:
        result = field == val
    return not result if condition.get("negate") else result


def _subjects_matching(sender_email, condition, *, want_match):
    """Subjects in SUBJECTS[sender_email] that satisfy (or don't) a subject condition."""
    subjects = SUBJECTS.get(sender_email, ["(no subject)"])
    return [s for s in subjects if _check_condition({"subject": s}, condition) == want_match]


def email_matches_rule(email, rule):
    """Whether this email must receive the rule's action (not skip)."""
    if not _check_condition(email, rule):
        return False
    # AND condition: secondary condition must also be true
    if "also_requires" in rule and not _check_condition(email, rule["also_requires"]):
        return False
    # EXCEPT condition: exception condition must be false
    if "except_if" in rule and _check_condition(email, rule["except_if"]):
        return False
    # EXCEPT-ANY condition: any matching exception → skip
    if "except_if_any" in rule and any(_check_condition(email, c) for c in rule["except_if_any"]):
        return False
    return True


def _pick_sender_by_email(email_addr):
    for name, addr in SENDERS + DECOY_SENDERS:
        if addr == email_addr:
            return name, addr
    return "Unknown", email_addr


def _make_email(eid, sender_email, subject=None, rng=None):
    name, addr = _pick_sender_by_email(sender_email)
    if subject is None:
        _rng = rng or random.Random()
        subject = _rng.choice(SUBJECTS.get(addr, ["(no subject)"]))
    return {
        "id": eid,
        "sender_name": name,
        "sender_email": addr,
        "subject": subject,
    }


def get_random_rule(rng: random.Random, difficulty: str = "easy") -> dict:
    """Pick one rule from the appropriate pool using the provided rng instance."""
    if difficulty == "expert":
        pool = list(EXPERT_RULES)      # negated AND + except_if_any only
    elif difficulty == "hard":
        pool = RULES + HARD_RULES + COMPOUND_RULES
    else:
        pool = list(RULES)
    return rng.choice(pool)


def generate_inbox(
    n_emails: int = 10,
    rng: random.Random | None = None,
    rule: dict | None = None,
    difficulty: str = "easy",
) -> list[dict]:
    """
    Generate inbox.  difficulty:
      easy   — fully random; rules may match zero emails
      medium — 2–3 matching emails + decoy lookalikes + shuffle
      hard   — 3–5 matches, more decoys, subject rules, compound rules mixed in
      expert — compound rules only; 2 full matches + 3–4 trap emails + decoys
    """
    if rng is None:
        rng = random.Random()

    if difficulty == "easy" or rule is None:
        emails = []
        for i in range(n_emails):
            name, addr = rng.choice(SENDERS)
            subject = rng.choice(SUBJECTS[addr])
            emails.append({"id": i, "sender_name": name, "sender_email": addr, "subject": subject})
        return emails

    is_compound = "also_requires" in rule or "except_if" in rule or "except_if_any" in rule
    is_medium   = difficulty == "medium"
    is_expert   = difficulty == "expert"

    # Guaranteed full-match emails built separately so truncation can never drop them.
    full_emails: list = []
    emails: list = []
    eid = 0

    if is_compound:
        sender       = rule["filter_value"]
        all_subjects = SUBJECTS.get(sender, ["(no subject)"])

        full_subjects = [s for s in all_subjects
                         if email_matches_rule({"sender_email": sender, "subject": s}, rule)]
        trap_subjects = [s for s in all_subjects if s not in full_subjects]

        full_subjects = full_subjects or all_subjects
        trap_subjects = trap_subjects or all_subjects

        for _ in range(2):
            full_emails.append(_make_email(eid, sender, subject=rng.choice(full_subjects), rng=rng))
            eid += 1

        n_trap = rng.randint(2, 3) if is_medium else rng.randint(3, 4)
        for _ in range(n_trap):
            emails.append(_make_email(eid, sender, subject=rng.choice(trap_subjects), rng=rng))
            eid += 1
    else:
        n_match = rng.randint(2, 3) if is_medium else rng.randint(3, 5)
        for _ in range(n_match):
            if rule.get("filter_op") == "contains":
                keyword = rule["filter_value"]
                sender  = rng.choice(SENDERS)[1]
                subject = f"Reminder: {keyword} sale ends soon"
                emails.append(_make_email(eid, sender, subject=subject, rng=rng))
            else:
                emails.append(_make_email(eid, rule["filter_value"], rng=rng))
            eid += 1

    # Expert: flood inbox with lookalikes that share the same subject pool.
    # Only sender address differs — forces exact-match precision.
    if is_expert and rule.get("filter_key") == "sender_email":
        target       = rule["filter_value"]
        lookalike_pool = LOOKALIKES.get(target, [])
        if lookalike_pool:
            n_look = rng.randint(min(3, len(lookalike_pool)), min(7, len(lookalike_pool)))
            chosen = rng.sample(lookalike_pool, n_look)
            for _, addr in chosen:
                subj = rng.choice(SUBJECTS.get(addr, ["(no subject)"]))
                emails.append(_make_email(eid, addr, subject=subj, rng=rng))
                eid += 1
        # Reduce generic decoys since lookalikes already fill the quota
        n_decoy = rng.randint(1, 2)
    elif is_medium:
        n_decoy = rng.randint(2, 3)
    else:
        n_decoy = rng.randint(3, 5)

    decoy_pool = [s for _, s in DECOY_SENDERS if s != rule.get("filter_value")]
    for _ in range(n_decoy):
        emails.append(_make_email(eid, rng.choice(decoy_pool), rng=rng))
        eid += 1

    if difficulty in ("hard", "expert") and rule.get("action") != "label:newsletter":
        for _ in range(2):
            emails.append(_make_email(eid, "news@techdigest.com", rng=rng))
            eid += 1

    slots_for_rest = n_emails - len(full_emails)
    while len(emails) < slots_for_rest:
        name, addr = rng.choice(SENDERS)
        emails.append({"id": eid, "sender_name": name, "sender_email": addr,
                        "subject": rng.choice(SUBJECTS[addr])})
        eid += 1

    rng.shuffle(emails)
    combined = full_emails + emails[:slots_for_rest]
    rng.shuffle(combined)
    for i, email in enumerate(combined):
        email["id"] = i
    return combined


def generate_episode(
    n_emails: int = 8,
    seed: int | None = None,
    difficulty: str = "easy",
) -> tuple[list[dict], dict]:
    """Pick a rule then build a matching inbox.  Thread-safe: uses a private rng."""
    rng  = random.Random(seed)
    rule = get_random_rule(rng, difficulty=difficulty)
    emails = generate_inbox(n_emails=n_emails, rng=rng, rule=rule, difficulty=difficulty)
    return emails, rule
