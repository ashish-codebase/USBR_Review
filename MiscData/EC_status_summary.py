#!/usr/bin/env python3
# summarize_status_messages.py (py3.5+ compatible)
# Modes:
#   --rolling      : every-30-min timer mode; builds TODAY each run; if 00:00–00:29 also (re)builds YESTERDAY
#   --day X        : build exactly day X (YYYY-MM-DD|today|yesterday)
#   --bulk         : iterate all .ghg in a folder (optionally --recurse) and write one big CSV
#
# Collects from system_config/:
#   - sf23_status.json (all top-level keys)
#   - usb_stats.json (bytes + computed usb_used_pct)
#   - ttyS4.json (sonic_* fields)
#   - config.json (fallback sonic_* from uarts[0])
#   - aligner.log (model, sonic info, freq, missing %, local window)
#   - co2app.conf (site_name, latitude, longitude, clock_zone)

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import zipfile

RAW_ROOT_DEFAULT = "/home/licor/data/raw"
SUMMARY_ROOT_DEFAULT = "/home/licor/data/summaries"

STATUS_CANDIDATES = [
    "system_config/sf23_status.json",
    "system_config\\sf23_status.json",
    "/system_config/sf23_status.json",
    "System_Config/sf23_status.json",
    "system_config/SF23_status.json",
]
USB_STATS_CANDIDATES   = ["system_config/usb_stats.json","system_config\\usb_stats.json"]
TTYS4_CANDIDATES       = ["system_config/ttyS4.json","system_config\\ttyS4.json"]
CONFIG_JSON_CANDIDATES = ["system_config/config.json","system_config\\config.json"]
ALIGNER_LOG_CANDIDATES = ["system_config/aligner.log","system_config\\aligner.log"]
CO2APP_CONF_CANDIDATES = ["system_config/co2app.conf","system_config\\co2app.conf"]

# <stamp>_<serial>.ghg  where stamp is YYYY-MM-DDTHHMMSS
FILEPAT = re.compile('(?P<stamp>\\d{4}-\\d{2}-\\d{2}T\\d{6})_(?P<serial>[^.]+)\\.ghg$', re.IGNORECASE)

# -------------------- args --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Summarize SmartFlux system_config across .ghg files")
    p.add_argument("--root", default=RAW_ROOT_DEFAULT, help="Root folder to scan (default: %s)" % RAW_ROOT_DEFAULT)
    p.add_argument("--recurse", action="store_true", help="Recurse into subdirectories")
    # Rolling/day builds
    p.add_argument("--day", help="Build a specific day: YYYY-MM-DD | today | yesterday")
    p.add_argument("--rolling", action="store_true", help="Half-hour mode: build today; if before 00:30, also build yesterday")
    p.add_argument("--daily", action="store_true", help="Deprecated; same as --day today")
    p.add_argument("--since", help="Deprecated for day builds; kept for backward compatibility")
    # Bulk
    p.add_argument("--bulk", action="store_true", help="Iterate all .ghg under --root (and --recurse) to one CSV")
    # Output & housekeeping
    p.add_argument("--out-dir", default=SUMMARY_ROOT_DEFAULT, help="Output directory (default: %s)" % SUMMARY_ROOT_DEFAULT)
    p.add_argument("--output", help="Explicit CSV path (overrides default naming)")
    p.add_argument("--append", action="store_true", help="Append to CSV if it exists")
    p.add_argument("--retention-days", type=int, default=0, help="Delete daily CSVs older than N days")
    p.add_argument("--log-dir", default=".", help="Per-run artifact directory")
    p.add_argument("--quiet", action="store_true", help="Suppress console summary")
    return p.parse_args()

# -------------------- helpers --------------------

def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)

def parse_name_metadata(path):
    m = FILEPAT.search(os.path.basename(path))
    if not m:
        return None, None
    return m.group("stamp"), m.group("serial")

def resolve_day(day_str):
    if not day_str or day_str == "today":
        return dt.date.today()
    if day_str == "yesterday":
        return dt.date.today() - dt.timedelta(days=1)
    try:
        return dt.datetime.strptime(day_str, "%Y-%m-%d").date()
    except Exception:
        sys.exit("ERROR: --day must be 'today', 'yesterday', or YYYY-MM-DD (got %r)" % day_str)

def window_for_day(day_date):
    start = dt.datetime.combine(day_date, dt.time(0,0,0))
    end   = start + dt.timedelta(days=1)
    return start, end

def dt_from_ghg(path, name_stamp):
    if name_stamp:
        try:
            return dt.datetime.strptime(name_stamp, "%Y-%m-%dT%H%M%S")
        except ValueError:
            pass
    try:
        return dt.datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None

def _read_text_from_zip(zf, name):
    with zf.open(name) as f:
        return io.TextIOWrapper(f, encoding="utf-8").read()

def _read_json_from_zip(zf, name):
    with zf.open(name) as f:
        return json.load(io.TextIOWrapper(f, encoding="utf-8"))

def _find_in_zip(zf, candidates):
    names = zf.namelist()
    want = set([c.lstrip("/") for c in candidates])
    for n in names:
        nn = n.replace("\\", "/")
        if nn in want:
            return nn
    # case-insensitive fallback
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        k = c.lstrip("/").lower()
        if k in lower_map:
            return lower_map[k]
    # basename fallback
    basenames = {n.split("/")[-1].lower(): n for n in names}
    for c in candidates:
        b = c.split("/")[-1].lower()
        if b in basenames:
            return basenames[b]
    return None

def read_system_config_files(base):
    out = {}
    if os.path.isdir(base):
        def find_rel(cands):
            for c in cands:
                p = os.path.join(base, c)
                if os.path.exists(p):
                    return p
            return None
        # sf23_status.json
        p = find_rel(STATUS_CANDIDATES)
        if p:
            try: out.update(parse_sf23_status(json.loads(open(p, "r").read())))
            except Exception: pass
        # usb_stats.json
        p = find_rel(USB_STATS_CANDIDATES)
        if p:
            try: out.update(parse_usb_stats(json.loads(open(p, "r").read())))
            except Exception: pass
        # ttyS4.json
        p = find_rel(TTYS4_CANDIDATES)
        if p:
            try: out.update(parse_ttys4(json.loads(open(p, "r").read())))
            except Exception: pass
        # config.json
        p = find_rel(CONFIG_JSON_CANDIDATES)
        if p:
            try: out.update(parse_config_json(json.loads(open(p, "r").read())))
            except Exception: pass
        # aligner.log
        p = find_rel(ALIGNER_LOG_CANDIDATES)
        if p:
            try: out.update(parse_aligner_log(open(p, "r", encoding="utf-8", errors="ignore").read()))
            except Exception: pass
        # co2app.conf
        p = find_rel(CO2APP_CONF_CANDIDATES)
        if p:
            try: out.update(parse_co2app_conf(open(p, "r", encoding="utf-8", errors="ignore").read()))
            except Exception: pass
    else:
        try:
            with zipfile.ZipFile(base, "r") as zf:
                name = _find_in_zip(zf, STATUS_CANDIDATES)
                if name:
                    try: out.update(parse_sf23_status(_read_json_from_zip(zf, name)))
                    except Exception: pass
                name = _find_in_zip(zf, USB_STATS_CANDIDATES)
                if name:
                    try: out.update(parse_usb_stats(_read_json_from_zip(zf, name)))
                    except Exception: pass
                name = _find_in_zip(zf, TTYS4_CANDIDATES)
                if name:
                    try: out.update(parse_ttys4(_read_json_from_zip(zf, name)))
                    except Exception: pass
                name = _find_in_zip(zf, CONFIG_JSON_CANDIDATES)
                if name:
                    try: out.update(parse_config_json(_read_json_from_zip(zf, name)))
                    except Exception: pass
                name = _find_in_zip(zf, ALIGNER_LOG_CANDIDATES)
                if name:
                    try: out.update(parse_aligner_log(_read_text_from_zip(zf, name)))
                    except Exception: pass
                name = _find_in_zip(zf, CO2APP_CONF_CANDIDATES)
                if name:
                    try: out.update(parse_co2app_conf(_read_text_from_zip(zf, name)))
                    except Exception: pass
        except zipfile.BadZipFile:
            pass
    return out

def parse_sf23_status(d):
    return dict(d)

def human_pct(numer, denom):
    try:
        if denom and float(denom) > 0:
            return round(100.0 * float(numer) / float(denom), 2)
    except Exception:
        pass
    return ""

def parse_usb_stats(d):
    u = d.get("usb", {})
    out = {"usb_total_bytes": u.get("total", ""), "usb_used_bytes": u.get("used", ""), "usb_free_bytes": u.get("free", "")}
    out["usb_used_pct"] = human_pct(out["usb_used_bytes"], out["usb_total_bytes"])
    return out

def parse_ttys4(d):
    out = {}
    mapping = {
        "serial_number":"sonic_serial_number",
        "software_version":"sonic_sw_version",
        "output_rate":"sonic_output_rate_hz",
        "sonic_type":"sonic_type",
    }
    for k, outk in mapping.items():
        if k in d:
            out[outk] = d.get(k, "")
    return out

def parse_config_json(d):
    out = {}
    try:
        uart0 = (d.get("uarts") or [])[0]
    except Exception:
        uart0 = None
    if uart0:
        if "serial_number" in uart0:   out["sonic_serial_number"] = uart0.get("serial_number", "")
        if "software_version" in uart0:out["sonic_sw_version"] = uart0.get("software_version", "")
        if "output_rate" in uart0:     out["sonic_output_rate_hz"] = uart0.get("output_rate", "")
        if "sonic_type" in uart0:      out["sonic_type"] = uart0.get("sonic_type", "")
    return out

def parse_aligner_log(text):
    out = {}
    m = re.search(r"Analyzer model:\s*(.+)", text)
    if m: out["aligner_analyzer_model"] = m.group(1).strip()
    m = re.search(r"Sonic type:\s*([^\n]+)", text);      out["aligner_sonic_type"] = m.group(1).strip() if m else ""
    m = re.search(r"Sonic SN:\s*([^\n]+)", text);        out["aligner_sonic_sn"] = m.group(1).strip() if m else ""
    m = re.search(r"Sonic frequency:\s*([0-9.]+)", text)
    if m: out["aligner_sonic_freq_hz"] = m.group(1).strip()
    m = re.search(r"Percentage missing data points:\s*([0-9.]+)\s*%", text)
    if m: out["aligner_missing_pct"] = m.group(1).strip()
    m = re.search(r"Local time range:\s*([0-9:\-\s]+)\s*TO\s*([0-9:\-\s]+)", text)
    if m:
        out["aligner_local_start"] = m.group(1).strip()
        out["aligner_local_end"]   = m.group(2).strip()
    return out

def _pull(token, text):
    m = re.search(r"\(%s\s+([^)]+)\)" % token, text)
    return m.group(1).strip() if m else ""

def parse_co2app_conf(text):
    out = {}
    out["clock_zone"] = _pull("Zone", text)
    out["site_name"]  = _pull("site_name", text)
    out["latitude"]   = _pull("latitude", text)
    out["longitude"]  = _pull("longitude", text)
    return out

def ghg_iter(root, recurse):
    if not os.path.isdir(root):
        return
    if recurse:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".ghg"):
                    yield os.path.join(dirpath, fn)
            for sub in dirnames:
                full = os.path.join(dirpath, sub)
                if os.path.isdir(full) and full.lower().endswith(".ghg"):
                    yield full
    else:
        for fn in os.listdir(root):
            full = os.path.join(root, fn)
            if fn.lower().endswith(".ghg"):
                yield full
            if os.path.isdir(full) and fn.lower().endswith(".ghg"):
                yield full

def union_keys(dicts):
    keys, seen = [], set()
    for d in dicts:
        for k in d.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    return sorted(keys)

def write_csv(records, out_path, append=False):
    ensure_dir(os.path.dirname(out_path))
    status_keys = union_keys([r[1] for r in records if r[1]])
    meta_keys = ["ghg_path", "ghg_name", "ghg_timestamp", "ghg_serial_from_name"]
    header = meta_keys + status_keys
    write_header = not (append and os.path.exists(out_path))
    with open(out_path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for meta, payload in records:
            row = [meta.get(k, "") for k in meta_keys]
            for k in status_keys:
                v = payload.get(k, "") if payload else ""
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                row.append(v)
            w.writerow(row)

def pick_output_for_day(out_dir, day_date):
    ensure_dir(out_dir)
    return os.path.join(out_dir, "sf23_status_%s.csv" % day_date.strftime("%Y-%m-%d"))

def pick_output_for_bulk(out_dir):
    ensure_dir(out_dir)
    return os.path.join(out_dir, "sf23_status_all.csv")

# -------------------- builders --------------------

def build_for_day(day_date, args):
    start, end = window_for_day(day_date)
    records = []
    for item in sorted(ghg_iter(args.root, args.recurse)):
        stamp, serial = parse_name_metadata(item)
        dt_guess = dt_from_ghg(item, stamp)
        if not dt_guess or not (start <= dt_guess < end):
            continue
        payload = read_system_config_files(item)
        ts = stamp if stamp else (dt_guess.strftime("%Y-%m-%dT%H%M%S") if dt_guess else "")
        meta = {
            "ghg_path": os.path.abspath(item),
            "ghg_name": os.path.basename(item),
            "ghg_timestamp": ts,
            "ghg_serial_from_name": serial or "",
        }
        records.append((meta, payload or {}))
    out_csv = (args.output or pick_output_for_day(args.out_dir, day_date))
    write_csv(records, out_csv, append=args.append)
    if not args.quiet:
        print("Wrote %s with %d rows" % (out_csv, len(records)))
    return out_csv, len(records)

def build_bulk(args):
    records = []
    for item in sorted(ghg_iter(args.root, args.recurse)):
        stamp, serial = parse_name_metadata(item)
        dt_guess = dt_from_ghg(item, stamp)
        payload = read_system_config_files(item)
        ts = stamp if stamp else (dt_guess.strftime("%Y-%m-%dT%H%M%S") if dt_guess else "")
        meta = {
            "ghg_path": os.path.abspath(item),
            "ghg_name": os.path.basename(item),
            "ghg_timestamp": ts,
            "ghg_serial_from_name": serial or "",
        }
        records.append((meta, payload or {}))
    out_csv = (args.output or pick_output_for_bulk(args.out_dir))
    write_csv(records, out_csv, append=args.append)
    if not args.quiet:
        print("Wrote %s with %d rows" % (out_csv, len(records)))
    return out_csv, len(records)

# -------------------- main --------------------

def main():
    args = parse_args()

    # Normalize legacy flags
    if args.daily and not args.day:
        args.day = "today"

    ensure_dir(args.out_dir); ensure_dir(args.log_dir)

    if args.bulk:
        build_bulk(args)
        return

    if args.rolling:
        today = dt.date.today()
        build_for_day(today, args)
        now = dt.datetime.now()
        if now.hour == 0 and now.minute < 30:
            yday = today - dt.timedelta(days=1)
            build_for_day(yday, args)
        return

    if args.day:
        day = resolve_day(args.day)
        build_for_day(day, args)
        return

    # Fallbacks for older usage
    if args.since:
        try:
            sdate = dt.datetime.strptime(args.since, "%Y-%m-%d").date()
            build_for_day(sdate, args)
            return
        except Exception:
            pass
        build_for_day(dt.date.today(), args)
        return

    # Default: build today
    build_for_day(dt.date.today(), args)

if __name__ == "__main__":
    main()
