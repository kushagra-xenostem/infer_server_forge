#!/usr/bin/env python3
"""Materialize the agent task suite into tasks/suite/<id>/{task.json,repo/,solution/,tests/}.

Each task = a small repo shown to the model + an instruction + hidden pytest tests.
solution/ holds reference versions of changed files; validate with
`agent_tasks.py validate` (original must FAIL hidden tests, reference must PASS).
"""
import json
import shutil
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).parent / "suite"

T = []


def task(id, category, instruction, repo, solution, tests):
    T.append(dict(id=id, category=category, instruction=textwrap.dedent(instruction).strip(),
                  repo=repo, solution=solution, tests=tests))


d = textwrap.dedent

# ---------------------------------------------------------------- bug fixes
task("t01_pagination_off_by_one", "bugfix",
     "Users report that page 1 of `paginate` skips the first items and the last page is sometimes missing. Pages are 1-based. Fix the bug.",
     {"pager.py": d('''
        def paginate(items, page, per_page):
            """Return the items on `page` (1-based) with `per_page` items per page."""
            start = page * per_page
            end = start + per_page
            return items[start:end]


        def page_count(n_items, per_page):
            return n_items // per_page
        ''')},
     {"pager.py": d('''
        def paginate(items, page, per_page):
            """Return the items on `page` (1-based) with `per_page` items per page."""
            start = (page - 1) * per_page
            end = start + per_page
            return items[start:end]


        def page_count(n_items, per_page):
            return -(-n_items // per_page)
        ''')},
     {"test_pager.py": d('''
        from pager import paginate, page_count
        def test_first_page():
            assert paginate(list(range(10)), 1, 3) == [0, 1, 2]
        def test_last_page():
            assert paginate(list(range(10)), 4, 3) == [9]
        def test_count():
            assert page_count(10, 3) == 4
            assert page_count(9, 3) == 3
            assert page_count(0, 3) == 0
        ''')})

task("t02_mutable_default", "bugfix",
     "Calling `add_tag` on different items leaks tags between them. Fix it without changing the function signature's behavior for callers who pass `tags`.",
     {"tags.py": d('''
        def add_tag(item, tag, tags=[]):
            tags.append(tag)
            return {"item": item, "tags": tags}
        ''')},
     {"tags.py": d('''
        def add_tag(item, tag, tags=None):
            tags = [] if tags is None else tags
            tags.append(tag)
            return {"item": item, "tags": tags}
        ''')},
     {"test_tags.py": d('''
        from tags import add_tag
        def test_no_leak():
            assert add_tag("a", "x")["tags"] == ["x"]
            assert add_tag("b", "y")["tags"] == ["y"]
        def test_explicit_list():
            t = ["p"]
            assert add_tag("c", "q", t)["tags"] == ["p", "q"]
        ''')})

task("t03_traceback_fix", "bugfix",
     """
     Running the service crashes with:

     Traceback (most recent call last):
       File "service.py", line 12, in start
         cfg = load_config(raw)
       File "config.py", line 6, in load_config
         "timeout": int(raw["timeout"]),
     KeyError: 'timeout'

     Optional keys must fall back to DEFAULTS. `host` stays required: a missing host must raise ValueError("host is required").
     """,
     {"config.py": d('''
        DEFAULTS = {"port": 8080, "timeout": 30, "debug": False}

        def load_config(raw):
            return {
                "host": raw["host"],
                "timeout": int(raw["timeout"]),
                "port": int(raw["port"]),
                "debug": bool(raw["debug"]),
            }
        '''),
      "service.py": d('''
        from config import load_config

        def start(raw):
            cfg = load_config(raw)
            return f"listening on {cfg['host']}:{cfg['port']}"
        ''')},
     {"config.py": d('''
        DEFAULTS = {"port": 8080, "timeout": 30, "debug": False}

        def load_config(raw):
            if "host" not in raw:
                raise ValueError("host is required")
            merged = {**DEFAULTS, **raw}
            return {
                "host": merged["host"],
                "timeout": int(merged["timeout"]),
                "port": int(merged["port"]),
                "debug": bool(merged["debug"]),
            }
        ''')},
     {"test_config.py": d('''
        import pytest
        from config import load_config
        from service import start
        def test_defaults():
            c = load_config({"host": "h"})
            assert c == {"host": "h", "timeout": 30, "port": 8080, "debug": False}
        def test_override():
            assert load_config({"host": "h", "port": "9000"})["port"] == 9000
        def test_host_required():
            with pytest.raises(ValueError, match="host is required"):
                load_config({"port": 1})
        def test_start():
            assert start({"host": "x"}) == "listening on x:8080"
        ''')})

task("t04_inheritance_init", "bugfix",
     "`AdminUser` objects are missing attributes set by `User`, and `describe()` crashes. Fix `AdminUser`.",
     {"users.py": d('''
        class User:
            def __init__(self, name, email):
                self.name = name
                self.email = email.lower()

            def describe(self):
                return f"{self.name} <{self.email}>"


        class AdminUser(User):
            def __init__(self, name, email, perms):
                self.perms = set(perms)

            def describe(self):
                return super().describe() + f" [admin: {', '.join(sorted(self.perms))}]"
        ''')},
     {"users.py": d('''
        class User:
            def __init__(self, name, email):
                self.name = name
                self.email = email.lower()

            def describe(self):
                return f"{self.name} <{self.email}>"


        class AdminUser(User):
            def __init__(self, name, email, perms):
                super().__init__(name, email)
                self.perms = set(perms)

            def describe(self):
                return super().describe() + f" [admin: {', '.join(sorted(self.perms))}]"
        ''')},
     {"test_users.py": d('''
        from users import AdminUser
        def test_admin():
            a = AdminUser("Ann", "ANN@X.IO", ["write", "read"])
            assert a.email == "ann@x.io"
            assert a.describe() == "Ann <ann@x.io> [admin: read, write]"
        ''')})

task("t05_timezone_bug", "bugfix",
     "`hours_between` gives wrong answers when the timestamps carry different UTC offsets, and crashes for the 'Z' suffix on some inputs. Timestamps are ISO-8601 strings. Return the absolute difference in hours as a float.",
     {"timeutil.py": d('''
        from datetime import datetime

        def hours_between(a, b):
            fmt = "%Y-%m-%dT%H:%M:%S"
            ta = datetime.strptime(a[:19], fmt)
            tb = datetime.strptime(b[:19], fmt)
            return abs((tb - ta).total_seconds()) / 3600
        ''')},
     {"timeutil.py": d('''
        from datetime import datetime

        def _parse(s):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        def hours_between(a, b):
            return abs((_parse(b) - _parse(a)).total_seconds()) / 3600
        ''')},
     {"test_timeutil.py": d('''
        from timeutil import hours_between
        def test_offsets():
            assert hours_between("2026-10-04T12:00:00+02:00", "2026-10-04T12:00:00Z") == 2.0
        def test_z():
            assert hours_between("2026-10-04T00:00:00Z", "2026-10-05T06:30:00Z") == 30.5
        def test_order():
            assert hours_between("2026-10-05T00:00:00+00:00", "2026-10-04T00:00:00+00:00") == 24.0
        ''')})

task("t06_sql_injection", "bugfix",
     "`find_user` breaks for names containing a quote (e.g. O'Brien) and is vulnerable to SQL injection. Fix it using parameterized queries; keep the same return format.",
     {"db.py": d('''
        import sqlite3

        def make_db():
            con = sqlite3.connect(":memory:")
            con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            con.executemany("INSERT INTO users (name) VALUES (?)", [("alice",), ("O'Brien",), ("bob",)])
            return con

        def find_user(con, name):
            rows = con.execute(f"SELECT id, name FROM users WHERE name = '{name}'").fetchall()
            return [{"id": r[0], "name": r[1]} for r in rows]
        ''')},
     {"db.py": d('''
        import sqlite3

        def make_db():
            con = sqlite3.connect(":memory:")
            con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            con.executemany("INSERT INTO users (name) VALUES (?)", [("alice",), ("O'Brien",), ("bob",)])
            return con

        def find_user(con, name):
            rows = con.execute("SELECT id, name FROM users WHERE name = ?", (name,)).fetchall()
            return [{"id": r[0], "name": r[1]} for r in rows]
        ''')},
     {"test_db.py": d('''
        from db import make_db, find_user
        def test_quote():
            assert find_user(make_db(), "O'Brien") == [{"id": 2, "name": "O'Brien"}]
        def test_injection():
            assert find_user(make_db(), "x' OR '1'='1") == []
        def test_plain():
            assert find_user(make_db(), "bob") == [{"id": 3, "name": "bob"}]
        ''')})

task("t07_async_gather", "bugfix",
     "`fetch_all` is supposed to fetch all urls concurrently and return results in the same order as `urls`, but it returns coroutine objects. Fix it so it awaits concurrently (not sequentially).",
     {"fetcher.py": d('''
        import asyncio

        async def fetch(url, delay=0.2):
            await asyncio.sleep(delay)
            return f"<{url}>"

        async def fetch_all(urls):
            results = []
            for u in urls:
                results.append(fetch(u))
            return results
        ''')},
     {"fetcher.py": d('''
        import asyncio

        async def fetch(url, delay=0.2):
            await asyncio.sleep(delay)
            return f"<{url}>"

        async def fetch_all(urls):
            return list(await asyncio.gather(*(fetch(u) for u in urls)))
        ''')},
     {"test_fetcher.py": d('''
        import asyncio, time
        from fetcher import fetch_all
        def test_order_and_concurrency():
            t = time.perf_counter()
            r = asyncio.run(fetch_all(["a", "b", "c", "d", "e"]))
            assert r == ["<a>", "<b>", "<c>", "<d>", "<e>"]
            assert time.perf_counter() - t < 0.6
        ''')})

# ---------------------------------------------------------------- implement from spec
task("t08_merge_intervals", "implement",
     "Implement `merge_intervals` according to its docstring.",
     {"intervals.py": d('''
        def merge_intervals(intervals):
            """Merge overlapping or touching closed intervals.

            intervals: list of (start, end) tuples with start <= end, in any order.
            Returns a sorted list of merged (start, end) tuples.
            Touching intervals like (1, 2) and (2, 3) merge into (1, 3).
            """
            raise NotImplementedError
        ''')},
     {"intervals.py": d('''
        def merge_intervals(intervals):
            out = []
            for s, e in sorted(intervals):
                if out and s <= out[-1][1]:
                    out[-1] = (out[-1][0], max(out[-1][1], e))
                else:
                    out.append((s, e))
            return out
        ''')},
     {"test_intervals.py": d('''
        from intervals import merge_intervals
        def test_basic():
            assert merge_intervals([(5, 6), (1, 3), (2, 4)]) == [(1, 4), (5, 6)]
        def test_touch():
            assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]
        def test_nested():
            assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]
        def test_empty():
            assert merge_intervals([]) == []
        ''')})

task("t09_lru_cache", "implement",
     "Implement the `LRUCache` class: `get(key)` returns the value or -1 and marks the key as recently used; `put(key, value)` inserts/updates and, when over capacity, evicts the least recently used key. Both operations must be O(1).",
     {"lru.py": d('''
        class LRUCache:
            def __init__(self, capacity: int):
                raise NotImplementedError

            def get(self, key):
                raise NotImplementedError

            def put(self, key, value):
                raise NotImplementedError
        ''')},
     {"lru.py": d('''
        from collections import OrderedDict

        class LRUCache:
            def __init__(self, capacity: int):
                self.cap = capacity
                self.d = OrderedDict()

            def get(self, key):
                if key not in self.d:
                    return -1
                self.d.move_to_end(key)
                return self.d[key]

            def put(self, key, value):
                self.d[key] = value
                self.d.move_to_end(key)
                if len(self.d) > self.cap:
                    self.d.popitem(last=False)
        ''')},
     {"test_lru.py": d('''
        from lru import LRUCache
        def test_lru():
            c = LRUCache(2)
            c.put(1, 1); c.put(2, 2)
            assert c.get(1) == 1
            c.put(3, 3)
            assert c.get(2) == -1
            c.put(4, 4)
            assert c.get(1) == -1
            assert c.get(3) == 3 and c.get(4) == 4
        def test_update():
            c = LRUCache(2)
            c.put(1, 1); c.put(2, 2); c.put(1, 10); c.put(3, 3)
            assert c.get(1) == 10 and c.get(2) == -1
        ''')})

task("t10_log_parser", "implement",
     """
     Implement `parse_line` for access-log lines like:

     2026-10-04 17:02:11 INFO [api] GET /v1/users 200 13.5ms

     Return a dict with keys: date (str), time (str), level (str), component (str), method (str), path (str), status (int), latency_ms (float).
     Return None for lines that don't match the format.
     """,
     {"logparse.py": d('''
        def parse_line(line):
            raise NotImplementedError
        ''')},
     {"logparse.py": d('''
        import re
        _RE = re.compile(r"^(\\d{4}-\\d{2}-\\d{2}) (\\d{2}:\\d{2}:\\d{2}) (\\w+) \\[([^\\]]+)\\] (\\w+) (\\S+) (\\d{3}) ([\\d.]+)ms$")

        def parse_line(line):
            m = _RE.match(line.strip())
            if not m:
                return None
            d, t, lv, c, me, p, s, l = m.groups()
            return dict(date=d, time=t, level=lv, component=c, method=me, path=p, status=int(s), latency_ms=float(l))
        ''')},
     {"test_logparse.py": d('''
        from logparse import parse_line
        def test_ok():
            assert parse_line("2026-10-04 17:02:11 INFO [api] GET /v1/users 200 13.5ms") == dict(
                date="2026-10-04", time="17:02:11", level="INFO", component="api", method="GET",
                path="/v1/users", status=200, latency_ms=13.5)
        def test_int_latency():
            r = parse_line("2026-01-01 00:00:00 WARN [db-pool] POST /x?y=1 503 120ms")
            assert r["component"] == "db-pool" and r["status"] == 503 and r["latency_ms"] == 120.0
        def test_bad():
            assert parse_line("garbage line") is None
            assert parse_line("") is None
        ''')})

task("t11_word_wrap", "implement",
     "Implement `wrap(text, width)` per the docstring.",
     {"wrap.py": d('''
        def wrap(text, width):
            """Greedy word wrap.

            - Split on whitespace; join words on a line with single spaces.
            - Each returned line is at most `width` chars, except a single word longer
              than `width` is placed alone on its own line (never split words).
            - Returns a list of lines; empty/whitespace-only text returns [].
            """
            raise NotImplementedError
        ''')},
     {"wrap.py": d('''
        def wrap(text, width):
            lines, cur = [], ""
            for w in text.split():
                if not cur:
                    cur = w
                elif len(cur) + 1 + len(w) <= width:
                    cur += " " + w
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines
        ''')},
     {"test_wrap.py": d('''
        from wrap import wrap
        def test_basic():
            assert wrap("the quick brown fox jumps", 10) == ["the quick", "brown fox", "jumps"]
        def test_long_word():
            assert wrap("a supercalifragilistic b", 5) == ["a", "supercalifragilistic", "b"]
        def test_ws():
            assert wrap("  a   b  ", 3) == ["a b"]
            assert wrap("   ", 4) == []
        ''')})

task("t12_handler_validation", "implement",
     """
     Implement `handle_create_user(body)` for a JSON API. `body` is a dict (already parsed). Rules:
     - `name`: required non-empty string after stripping -> stored stripped.
     - `age`: required int (bool is NOT acceptable) with 0 <= age <= 150.
     - `email`: optional; if present must contain exactly one '@' with non-empty parts.
     Return (status_code, response_dict):
     - On success: (201, {"name": ..., "age": ..., "email": <value or None>})
     - On validation failure: (422, {"errors": {field: message}}) listing ALL invalid fields (any message text).
     - If body is not a dict: (400, {"errors": {"body": "must be an object"}})
     """,
     {"api.py": d('''
        def handle_create_user(body):
            raise NotImplementedError
        ''')},
     {"api.py": d('''
        def handle_create_user(body):
            if not isinstance(body, dict):
                return 400, {"errors": {"body": "must be an object"}}
            errs = {}
            name = body.get("name")
            if not isinstance(name, str) or not name.strip():
                errs["name"] = "required"
            age = body.get("age")
            if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 150:
                errs["age"] = "invalid"
            email = body.get("email")
            if email is not None:
                parts = email.split("@") if isinstance(email, str) else []
                if len(parts) != 2 or not all(parts):
                    errs["email"] = "invalid"
            if errs:
                return 422, {"errors": errs}
            return 201, {"name": name.strip(), "age": age, "email": email}
        ''')},
     {"test_api.py": d('''
        from api import handle_create_user as h
        def test_ok():
            assert h({"name": " Ann ", "age": 30}) == (201, {"name": "Ann", "age": 30, "email": None})
            assert h({"name": "B", "age": 0, "email": "b@x.io"})[0] == 201
        def test_errors_all_fields():
            s, r = h({"name": "  ", "age": True, "email": "a@@b"})
            assert s == 422 and set(r["errors"]) == {"name", "age", "email"}
        def test_age_range():
            assert h({"name": "x", "age": 151})[0] == 422
            assert h({"name": "x", "age": "3"})[0] == 422
        def test_not_dict():
            assert h([1]) == (400, {"errors": {"body": "must be an object"}})
        ''')})

# ---------------------------------------------------------------- feature additions to existing code
task("t13_cache_ttl", "feature",
     "Add optional per-entry TTL to `Cache`: `set(key, value, ttl=None)`; ttl is seconds. Expired entries must behave as missing in `get` (return default) and `__len__`. Use the injected `clock` (a zero-arg callable returning seconds) so it is testable; keep existing behavior when ttl is None.",
     {"cache.py": d('''
        import time

        class Cache:
            def __init__(self, clock=time.monotonic):
                self._clock = clock
                self._data = {}

            def set(self, key, value):
                self._data[key] = value

            def get(self, key, default=None):
                return self._data.get(key, default)

            def __len__(self):
                return len(self._data)
        ''')},
     {"cache.py": d('''
        import time

        class Cache:
            def __init__(self, clock=time.monotonic):
                self._clock = clock
                self._data = {}

            def set(self, key, value, ttl=None):
                exp = None if ttl is None else self._clock() + ttl
                self._data[key] = (value, exp)

            def _purge(self):
                now = self._clock()
                for k in [k for k, (_, e) in self._data.items() if e is not None and e <= now]:
                    del self._data[k]

            def get(self, key, default=None):
                self._purge()
                return self._data[key][0] if key in self._data else default

            def __len__(self):
                self._purge()
                return len(self._data)
        ''')},
     {"test_cache.py": d('''
        from cache import Cache
        class Clock:
            t = 0.0
            def __call__(self): return self.t
        def test_ttl():
            clk = Clock(); c = Cache(clock=clk)
            c.set("a", 1, ttl=10); c.set("b", 2)
            assert c.get("a") == 1 and len(c) == 2
            clk.t = 10.5
            assert c.get("a") is None and c.get("a", "d") == "d"
            assert c.get("b") == 2 and len(c) == 1
        def test_no_ttl_default():
            c = Cache(); c.set("x", 5)
            assert c.get("x") == 5
        ''')})

task("t14_cli_flags", "feature",
     "Add two flags to the CLI: `--limit N` (int, optional) to output at most N matching lines, and `--ignore-case` / `-i` for case-insensitive matching. Keep `main(argv)` returning the list of output lines.",
     {"grepcli.py": d('''
        import argparse

        def main(argv):
            p = argparse.ArgumentParser()
            p.add_argument("pattern")
            p.add_argument("file")
            args = p.parse_args(argv)
            out = []
            with open(args.file) as f:
                for line in f:
                    line = line.rstrip("\\n")
                    if args.pattern in line:
                        out.append(line)
            return out
        ''')},
     {"grepcli.py": d('''
        import argparse

        def main(argv):
            p = argparse.ArgumentParser()
            p.add_argument("pattern")
            p.add_argument("file")
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("-i", "--ignore-case", action="store_true")
            args = p.parse_args(argv)
            pat = args.pattern.lower() if args.ignore_case else args.pattern
            out = []
            with open(args.file) as f:
                for line in f:
                    line = line.rstrip("\\n")
                    hay = line.lower() if args.ignore_case else line
                    if pat in hay:
                        out.append(line)
                        if args.limit is not None and len(out) >= args.limit:
                            break
            return out
        ''')},
     {"test_grepcli.py": d('''
        from grepcli import main
        def _f(tmp_path):
            p = tmp_path / "f.txt"; p.write_text("Error a\\nerror b\\nok\\nERROR c\\n"); return str(p)
        def test_plain(tmp_path):
            assert main(["Error", _f(tmp_path)]) == ["Error a"]
        def test_icase(tmp_path):
            assert main(["error", _f(tmp_path), "-i"]) == ["Error a", "error b", "ERROR c"]
            assert main(["error", _f(tmp_path), "--ignore-case"]) == ["Error a", "error b", "ERROR c"]
        def test_limit(tmp_path):
            assert main(["error", _f(tmp_path), "-i", "--limit", "2"]) == ["Error a", "error b"]
        ''')})

task("t15_dataclass_validation", "feature",
     "Make `Order` validate itself on construction: quantity must be a positive int, unit_price must be >= 0, and currency must be one of SUPPORTED. Raise ValueError otherwise. Also add a `total` property (quantity * unit_price, rounded to 2 decimals).",
     {"order.py": d('''
        from dataclasses import dataclass

        SUPPORTED = {"USD", "EUR", "INR"}

        @dataclass
        class Order:
            sku: str
            quantity: int
            unit_price: float
            currency: str = "USD"
        ''')},
     {"order.py": d('''
        from dataclasses import dataclass

        SUPPORTED = {"USD", "EUR", "INR"}

        @dataclass
        class Order:
            sku: str
            quantity: int
            unit_price: float
            currency: str = "USD"

            def __post_init__(self):
                if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
                    raise ValueError("quantity")
                if self.unit_price < 0:
                    raise ValueError("unit_price")
                if self.currency not in SUPPORTED:
                    raise ValueError("currency")

            @property
            def total(self):
                return round(self.quantity * self.unit_price, 2)
        ''')},
     {"test_order.py": d('''
        import pytest
        from order import Order
        def test_total():
            assert Order("a", 3, 1.1).total == 3.3
        @pytest.mark.parametrize("kw", [dict(quantity=0), dict(quantity=-1), dict(quantity=1.5),
                                        dict(unit_price=-0.01), dict(currency="GBP")])
        def test_invalid(kw):
            base = dict(sku="a", quantity=1, unit_price=1.0, currency="USD"); base.update(kw)
            with pytest.raises(ValueError):
                Order(**base)
        ''')})

# ---------------------------------------------------------------- multi-file refactors
task("t16_rename_across_files", "refactor",
     "Rename the function `calc_total` to `compute_total` everywhere (definition and all call sites). Do not keep the old name.",
     {"pricing.py": d('''
        def calc_total(prices, tax_rate=0.0):
            subtotal = sum(prices)
            return round(subtotal * (1 + tax_rate), 2)
        '''),
      "checkout.py": d('''
        from pricing import calc_total

        def checkout(cart, tax_rate):
            return {"items": len(cart), "total": calc_total([i["price"] for i in cart], tax_rate)}
        '''),
      "report.py": d('''
        import pricing

        def daily_report(orders):
            return sum(pricing.calc_total(o) for o in orders)
        ''')},
     {"pricing.py": d('''
        def compute_total(prices, tax_rate=0.0):
            subtotal = sum(prices)
            return round(subtotal * (1 + tax_rate), 2)
        '''),
      "checkout.py": d('''
        from pricing import compute_total

        def checkout(cart, tax_rate):
            return {"items": len(cart), "total": compute_total([i["price"] for i in cart], tax_rate)}
        '''),
      "report.py": d('''
        import pricing

        def daily_report(orders):
            return sum(pricing.compute_total(o) for o in orders)
        ''')},
     {"test_rename.py": d('''
        import pricing
        from checkout import checkout
        from report import daily_report
        def test_renamed():
            assert not hasattr(pricing, "calc_total")
            assert pricing.compute_total([1, 2], 0.5) == 4.5
        def test_callers():
            assert checkout([{"price": 10}], 0.1) == {"items": 1, "total": 11.0}
            assert daily_report([[1, 2], [3]]) == 6
        ''')})

task("t17_signature_change", "refactor",
     "`format_price` must support currencies. Add a `currency` parameter (default \"USD\") using the symbols in SYMBOLS (unknown currency -> use the code followed by a space, e.g. 'CHF 5.00'). Update `invoice.py` so each line uses the line's own `currency` field (default USD if missing).",
     {"money.py": d('''
        SYMBOLS = {"USD": "$", "EUR": "€", "INR": "₹"}

        def format_price(amount):
            return f"${amount:,.2f}"
        '''),
      "invoice.py": d('''
        from money import format_price

        def render(lines):
            return [f"{l['item']}: {format_price(l['amount'])}" for l in lines]
        ''')},
     {"money.py": d('''
        SYMBOLS = {"USD": "$", "EUR": "€", "INR": "₹"}

        def format_price(amount, currency="USD"):
            sym = SYMBOLS.get(currency, currency + " ")
            return f"{sym}{amount:,.2f}"
        '''),
      "invoice.py": d('''
        from money import format_price

        def render(lines):
            return [f"{l['item']}: {format_price(l['amount'], l.get('currency', 'USD'))}" for l in lines]
        ''')},
     {"test_money.py": d('''
        from money import format_price
        from invoice import render
        def test_fmt():
            assert format_price(1234.5) == "$1,234.50"
            assert format_price(3, "EUR") == "€3.00"
            assert format_price(5, "CHF") == "CHF 5.00"
        def test_invoice():
            assert render([{"item": "a", "amount": 2}, {"item": "b", "amount": 1, "currency": "INR"}]) == ["a: $2.00", "b: ₹1.00"]
        ''')})

# ---------------------------------------------------------------- performance / robustness
task("t18_quadratic_dupes", "performance",
     "`find_duplicates` is far too slow on large inputs (it must handle 200,000 items in well under a second). Rewrite it to be O(n) while keeping the exact output: each duplicated value once, in order of its first occurrence.",
     {"dupes.py": d('''
        def find_duplicates(items):
            out = []
            for i, x in enumerate(items):
                if x in items[i + 1:] and x not in out:
                    out.append(x)
            return out
        ''')},
     {"dupes.py": d('''
        def find_duplicates(items):
            counts = {}
            for x in items:
                counts[x] = counts.get(x, 0) + 1
            return [x for x, c in counts.items() if c > 1]
        ''')},
     {"test_dupes.py": d('''
        import time
        from dupes import find_duplicates
        def test_correct():
            assert find_duplicates([3, 1, 3, 2, 1, 3]) == [3, 1]
            assert find_duplicates([]) == []
        def test_fast():
            data = list(range(100_000)) * 2
            t = time.perf_counter()
            r = find_duplicates(data)
            assert time.perf_counter() - t < 1.0
            assert r[:3] == [0, 1, 2] and len(r) == 100_000
        ''')})

task("t19_deep_flatten", "performance",
     "`flatten` crashes with RecursionError on deeply nested input (depth 10,000). Rewrite it iteratively; output order must be unchanged (left-to-right).",
     {"flat.py": d('''
        def flatten(x):
            out = []
            for item in x:
                if isinstance(item, list):
                    out.extend(flatten(item))
                else:
                    out.append(item)
            return out
        ''')},
     {"flat.py": d('''
        def flatten(x):
            out, stack = [], [iter(x)]
            while stack:
                for item in stack[-1]:
                    if isinstance(item, list):
                        stack.append(iter(item))
                        break
                    out.append(item)
                else:
                    stack.pop()
            return out
        ''')},
     {"test_flat.py": d('''
        from flat import flatten
        def test_order():
            assert flatten([1, [2, [3, 4]], 5, [], [[6]]]) == [1, 2, 3, 4, 5, 6]
        def test_deep():
            x = [0]
            for i in range(1, 10_000):
                x = [x, i]
            r = flatten(x)
            assert r[:3] == [0, 1, 2] and len(r) == 10_000
        ''')})

# ---------------------------------------------------------------- write tests
task("t20_write_tests", "write-tests",
     "Write a thorough pytest test file `test_slugify.py` for the existing `slugify` function. Your tests must pass against the current (correct) implementation and should catch regressions in each documented rule. Do not modify slugify.py.",
     {"slugify.py": d('''
        import re
        import unicodedata

        def slugify(text, max_len=50):
            """Convert text to a URL slug.

            Rules:
            1. Unicode is normalized and accents stripped (é -> e); other non-ASCII is dropped.
            2. Lowercased.
            3. Runs of characters other than a-z/0-9 become a single '-'.
            4. Leading/trailing '-' removed.
            5. Truncated to max_len, then any trailing '-' removed again.
            """
            t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
            t = t.lower()
            t = re.sub(r"[^a-z0-9]+", "-", t)
            t = t.strip("-")
            return t[:max_len].rstrip("-")
        ''')},
     # reference answer; graded by mutation testing (see agent_tasks.py MUTANTS)
     {"test_slugify.py": d('''
        from slugify import slugify
        def test_accents():
            assert slugify("Café Ünïcode") == "cafe-unicode"
        def test_lower():
            assert slugify("ABC") == "abc"
        def test_runs():
            assert slugify("a  &&  b") == "a-b"
        def test_strip():
            assert slugify("--hello--") == "hello"
        def test_truncate():
            assert slugify("abc def", max_len=4) == "abc"
        ''')},
     {})


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for t in T:
        base = ROOT / t["id"]
        for sub, files in (("repo", t["repo"]), ("solution", t["solution"] or {}), ("tests", t["tests"])):
            for name, content in files.items():
                p = base / sub / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content.lstrip("\n"))
        meta = {k: t[k] for k in ("id", "category", "instruction")}
        (base / "task.json").write_text(json.dumps(meta, indent=2))
    print(f"built {len(T)} tasks in {ROOT}")


if __name__ == "__main__":
    main()
