# email_data.py
import random

# Triage actions (scored by verifier). "open" reveals sender in UI/agent view.
TRIAGE_ACTIONS = ["delete", "archive", "label:newsletter", "skip"]
OPEN_ACTION = "open"
ACTION_NAMES = TRIAGE_ACTIONS + [OPEN_ACTION]


def format_email_line(email, opened=False, prefix=""):
    """Format one email for agent prompts; hide sender until opened."""
    base = f"{prefix}id={email['id']} | subject: {email['subject']}"
    if opened:
        return f"{base} | {email['sender_name']} <{email['sender_email']}>"
    return f"{base} | [sender hidden — open to reveal]"

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

# Lookalike senders — easy to confuse with real rule targets
DECOY_SENDERS = [
    ("Promo Team", "promo@sale.net"),
    ("Sale Alerts", "sale@promo.com"),
    ("GitHub Actions", "notifications@github.com"),
    ("GitHub Bot", "github-noreply@github.com"),
    ("Tech Digest", "digest@techdigest.com"),
    ("LinkedIn Mail", "mail@linkedin.com"),
    ("Deals Daily", "deals@spammy.org"),
    ("Promo Offers", "offers@sale.com"),
]

SUBJECTS = {
    "alice@work.com": ["Q3 report ready", "Quick question", "Meeting notes"],
    "bob@work.com": ["PR review request", "Lunch tomorrow?", "Bug in prod"],
    "news@techdigest.com": ["This week in AI", "Top 10 tools", "Dev digest #42"],
    "promo@sale.com": ["50% OFF today only!", "Flash sale ends tonight", "You won!"],
    "noreply@github.com": ["New issue opened", "PR merged", "CI failed"],
    "deals@spammy.net": ["Congratulations!", "Claim your prize", "Urgent offer"],
    "dave@work.com": ["All hands Friday", "Performance review", "Team update"],
    "notifications@linkedin.com": [
        "You have 3 new connections",
        "Someone viewed your profile",
    ],
    "promo@sale.net": ["50% OFF today only!", "Limited time offer"],
    "sale@promo.com": ["Flash sale ends tonight", "Exclusive deal"],
    "notifications@github.com": ["Workflow run failed", "PR merged"],
    "github-noreply@github.com": ["Security alert", "Dependabot update"],
    "digest@techdigest.com": ["Weekly roundup", "Top 10 tools"],
    "mail@linkedin.com": ["New message", "Connection request"],
    "deals@spammy.org": ["Urgent offer", "Claim your prize"],
    "offers@sale.com": ["You won!", "Last chance"],
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
]


def email_matches_rule(email, rule):
    """Whether this email must receive the rule's action (not skip)."""
    key = rule["filter_key"]
    val = rule["filter_value"]
    field = email.get(key, "")
    if rule.get("filter_op") == "contains":
        return val.lower() in str(field).lower()
    return field == val


def _pick_sender_by_email(email_addr):
    for name, addr in SENDERS + DECOY_SENDERS:
        if addr == email_addr:
            return name, addr
    return "Unknown", email_addr


def _make_email(eid, sender_email, subject=None):
    name, addr = _pick_sender_by_email(sender_email)
    if subject is None:
        subject = random.choice(SUBJECTS.get(addr, ["(no subject)"]))
    return {
        "id": eid,
        "sender_name": name,
        "sender_email": addr,
        "subject": subject,
    }


def get_random_rule(seed=None, difficulty="easy"):
    if seed is not None:
        random.seed(seed)
    pool = list(RULES)
    if difficulty == "hard":
        pool = RULES + HARD_RULES
    return random.choice(pool)


def generate_inbox(n_emails=10, seed=None, rule=None, difficulty="easy"):
    """
    Generate inbox. difficulty:
      easy   — fully random (old behavior); rules may match zero emails
      medium — 2–3 matching emails + decoy lookalikes + shuffle
      hard   — 3–5 matches, more decoys, subject rules, trick subjects
    """
    if seed is not None:
        random.seed(seed)

    if difficulty == "easy" or rule is None:
        emails = []
        for i in range(n_emails):
            name, email = random.choice(SENDERS)
            subject = random.choice(SUBJECTS[email])
            emails.append(
                {
                    "id": i,
                    "sender_name": name,
                    "sender_email": email,
                    "subject": subject,
                }
            )
        return emails

    # medium / hard: rule-aware inbox
    n_match = random.randint(2, 3) if difficulty == "medium" else random.randint(3, 5)
    n_decoy = random.randint(2, 3) if difficulty == "medium" else random.randint(4, 6)

    emails = []
    eid = 0

    # Guaranteed emails that match the rule
    for _ in range(n_match):
        if rule.get("filter_op") == "contains":
            keyword = rule["filter_value"]
            sender = random.choice(SENDERS)[1]
            subject = f"Reminder: {keyword} sale ends soon"
            emails.append(_make_email(eid, sender, subject=subject))
        else:
            emails.append(_make_email(eid, rule["filter_value"]))
        eid += 1

    # Decoy lookalikes (same domain family, wrong address)
    decoy_pool = [s for _, s in DECOY_SENDERS]
    for _ in range(n_decoy):
        emails.append(_make_email(eid, random.choice(decoy_pool)))
        eid += 1

    # Hard: extra newsletter-like emails when rule is not about newsletters
    if difficulty == "hard" and rule.get("action") != "label:newsletter":
        for _ in range(2):
            emails.append(_make_email(eid, "news@techdigest.com"))
            eid += 1

    # Fill remaining slots with random noise
    while len(emails) < n_emails:
        name, addr = random.choice(SENDERS)
        emails.append(
            {
                "id": eid,
                "sender_name": name,
                "sender_email": addr,
                "subject": random.choice(SUBJECTS[addr]),
            }
        )
        eid += 1

    random.shuffle(emails)
    for i, email in enumerate(emails):
        email["id"] = i
    return emails[:n_emails]


def generate_episode(n_emails=8, seed=None, difficulty="easy"):
    """Pick rule then build a matching inbox."""
    rule = get_random_rule(seed=seed, difficulty=difficulty)
    if seed is not None:
        random.seed(seed + 1)
    emails = generate_inbox(
        n_emails=n_emails, seed=None, rule=rule, difficulty=difficulty
    )
    return emails, rule
