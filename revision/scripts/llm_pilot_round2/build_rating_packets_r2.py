"""S5c round 2 rating-packet builder. KEY-SIDE SCRIPT: reads arms/uids/native ids. Not for the raters.

Ported from the round-1 builder (`review_scratch_batch3_S5B_20261009/e7_packet_build/build_rating_packet.py`
sha256 84c4f6c7…): same item content (payload as sent, delivered text, first draft where the style revision
changed it), same deterministic sentence splitter, same rating-ID alphabet, same truth-free ordering
constraint (E7-A3(1)) and the same verbatim §5 rubric paragraph of the round-1 specification (checked by the
paragraph hash recorded by the round-1 build). Differences, all required by the round-2 specification: A0 only
(20 items), two byte-identical packets (one per rater session), nothing in builder-authored text names a round,
version, arm or instruction change, rating IDs avoid every round-1 ID, and the instructions' description of the
precomputed statistics names the round-2 counts.

Reads the run directory read only; writes only <out-root>/rating_packet_1/, rating_packet_2/ (rater-facing)
and rating_packet_key/ (key-side). Refuses a fake run unless --test-on-fake (then the out-root must lie outside
$PROSPER_OUT/s5c_pilot_round2_20261110/).

    python3 build_rating_packets_r2.py                      # after the released run is COMPLETE
    python3 build_rating_packets_r2.py --run-dir <dryrun_fake> --out-root <scratch> --test-on-fake
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import secrets
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROSPER_OUT = Path(os.environ.get("PROSPER_OUT", HERE.parents[1] / "prosper_revision_output"))
ROUND2_ROOT = PROSPER_OUT / "s5c_pilot_round2_20261110"
ROUND1 = PROSPER_OUT / "s5b_pilot_20261009"
ROUND1_CONTRACT = HERE.parents[1] / "docs" / "analysis" / "s5b_pilot_contract_20261009.md"
ROUND1_PARAGRAPH_SHA = "2a27ada7deedd58b54b4a9a31491df377d047c50388a1f1b816f612cfae031ff"   # round-1 build_checks.json
ARM = "a0_masked_posterior"
N_ITEMS = 20
FIRST_LABEL = "First draft (revised by the system's style check)"
PACKETS = ("rating_packet_1", "rating_packet_2")
RATER_FILES = ("RATING_INSTRUCTIONS.md", "packet.json", "packet.md", "rating_form.csv")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fsha(p) -> str:
    return sha(Path(p).read_bytes())


# ---------------------------------------------------------------- sentence splitter (round-1 rule, verbatim)
ABBR_RE = re.compile(r'(?:^|[\s(\[])(?:vs|e\.g|i\.e|etc|approx|cf|dr|no|fig|al)$', re.IGNORECASE)
BOUNDARY_RE = re.compile(r'([.!?])([)\]"\'”’]*)(\s+)(?=CASE_|[A-Z0-9])')
SPLITTER_RULE = (
    "Split after '.', '?' or '!' (plus any immediately following closing ')', ']', quote or apostrophe) "
    "when followed by whitespace and then 'CASE_', an uppercase ASCII letter or a digit; do not split when "
    "the token ending at that punctuation is one of vs / e.g / i.e / etc / approx / cf / Dr / No / Fig / al "
    "(case-insensitive). Decimals such as 0.95 are never split because no whitespace follows their point. "
    "Sentences are stripped of surrounding whitespace; empty pieces are dropped; text is otherwise unchanged."
)


def split_sentences(text):
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        if ABBR_RE.search(text[start:m.start(1)]):
            continue
        end = m.end(2)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = m.end(3)
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    assert ' '.join(out).replace(' ', '') == text.replace(' ', '').replace('\n', '').replace('\t', ''), 'splitter lost text'
    return out


def stem_of(cid):
    m = re.match(r'^(SJ[A-Za-z0-9]+)_[A-Za-z]+\d*$', cid)
    return m.group(1) if m else None


def raw_summary(resp_path: Path):
    """First-call text for rejected/too_long outputs: `summary` of the raw message.content, else the content."""
    calls = json.loads(resp_path.read_bytes()).get("calls") or []
    raw = next((c.get("raw_content") for c in calls if c.get("raw_content")), None)
    if raw is None:
        return None
    try:
        j = json.loads(raw)
        if isinstance(j, dict) and isinstance(j.get("summary"), str):
            return j["summary"]
    except Exception:                                    # noqa: BLE001
        pass
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, default=ROUND2_ROOT / "run")
    ap.add_argument("--out-root", type=Path, default=ROUND2_ROOT)
    ap.add_argument("--test-on-fake", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="default: drawn once with secrets.randbits(48)")
    a = ap.parse_args(argv)
    run, out_root = a.run_dir.resolve(), a.out_root.resolve()
    if ROUND1.resolve() in run.parents or ROUND1.resolve() in out_root.parents or out_root == ROUND1.resolve():
        raise SystemExit("round 1 is read-only and is not a round-2 input")
    meta = json.loads((run / "meta.json").read_text())
    if not (run / "COMPLETE").exists() or (run / "STOPPED").exists():
        raise SystemExit(f"{run} is not a COMPLETE run")
    if meta.get("fake") and not a.test_on_fake:
        raise SystemExit("fake run: packets are built only from the released real-call run")
    if a.test_on_fake and ROUND2_ROOT.resolve() in [out_root, *out_root.parents]:
        raise SystemExit("--test-on-fake must write outside the round-2 directory")
    if meta.get("arms") != [ARM] or meta.get("n_queries") != N_ITEMS:
        raise SystemExit(f"unexpected run: arms {meta.get('arms')}, n_queries {meta.get('n_queries')}")
    seed = a.seed if a.seed is not None else secrets.randbits(48)

    # ------------------------------------------------------------ load run (read-only)
    rows = [json.loads(l) for l in open(run / "runs.jsonl")]
    sel = [r for r in rows if r['selected'] and r['arm'] == ARM and r['repeat'] == 0]
    assert len(sel) == N_ITEMS and len({r['run_uid'] for r in sel}) == N_ITEMS, len(sel)
    ep = {}
    for l in open(run / 'key' / 'endpoints_auto.jsonl'):
        e = json.loads(l)
        ep[e['query']] = (e['native_id'], e['patient_stem'])
    queries = sorted({r['query'] for r in sel}, key=int)
    assert len(queries) == N_ITEMS and set(queries) <= set(ep)

    verif = {'payload_equal_user_message': 0, 'payload_file_sha_equals_runs': 0, 'calls_checked': 0,
             'developer_prompt_sha': set(), 'excluded_no_text': [], 'first_call_text_used': []}
    items = []
    for r in sorted(sel, key=lambda r: int(r['query'])):
        uid = r['run_uid']
        praw = (run / r['payload_file']).read_bytes()
        assert sha(praw) == r['payload_sha256'], uid
        verif['payload_file_sha_equals_runs'] += 1
        payload = json.loads(praw)
        resp_path = run / r['response_file']
        resp = json.loads(resp_path.read_bytes())
        assert resp['summary_key'] == [r['query'], r['arm'], r['repeat']]
        ok = True
        for c in resp['calls']:
            msgs = c['request']['messages']
            assert msgs[0]['role'] == 'developer' and msgs[1]['role'] == 'user'
            verif['developer_prompt_sha'].add(sha(msgs[0]['content'].encode()))
            um = json.loads(msgs[1]['content'])
            ok &= (um['payload'] == payload) and (list(um['payload']) == list(payload))
            verif['calls_checked'] += 1
        assert ok, uid
        verif['payload_equal_user_message'] += 1
        # specification §6: ok/truncated -> final text; rejected/too_long -> first-call text; nothing else is rated
        state = r['output_state']
        if state in ('ok', 'truncated') and r['text']:
            text, tf = r['text'], r['text_first']
        elif state in ('rejected', 'too_long'):
            text = r['text_first'] or raw_summary(resp_path)
            tf = None
            verif['first_call_text_used'].append(uid)
        else:
            text = None
        if not text:
            verif['excluded_no_text'].append({'uid': uid, 'output_state': state})
            continue
        items.append({'uid': uid, 'arm': r['arm'], 'query': r['query'], 'repeat': r['repeat'],
                      'payload_sha256': r['payload_sha256'], 'payload': payload, 'text': text,
                      'output_state': state,
                      'text_first_if_different': (tf if (tf is not None and tf != text) else None)})
    verif['developer_prompt_sha'] = sorted(verif['developer_prompt_sha'])
    if not a.test_on_fake:
        assert len(verif['developer_prompt_sha']) == 1
        assert verif['developer_prompt_sha'][0] == meta['prompt_sha256_module']['GROUNDED_SYSTEM_PROMPT_NEIGHBORS']

    # ------------------------------------------------------------ linkage relation (truth-free, round-1 rule)
    item_by_uid = {it['uid']: it for it in items}
    by_query = {it['query']: [it['uid']] for it in items}
    relations = []
    for it in items:
        y, p = it['uid'], it['payload']
        ser = json.dumps(p, ensure_ascii=False)
        for q in queries:
            if q == it['query'] or q not in by_query:
                continue
            native, stem = ep[q]
            for rank, n in enumerate(p['neighbors'], 1):
                cid, sid = str(n.get('case_id')), str(n.get('slide_id'))
                if cid == f'CASE_{q}':
                    relations.append((y, q, 'masked_id_equal', f'neighbors[{rank}].case_id', cid))
                if sid == q:
                    relations.append((y, q, 'scan_number_equal', f'neighbors[{rank}].slide_id', sid))
                if cid == native:
                    relations.append((y, q, 'native_id_equal', f'neighbors[{rank}].case_id', cid))
                elif stem and (cid == stem or stem_of(cid) == stem):
                    relations.append((y, q, 'same_patient_stem', f'neighbors[{rank}].case_id', cid))
            for pat, kind in ((rf'CASE_{q}(?![0-9])', 'masked_id_anywhere'),
                              (rf'{re.escape(native)}(?![A-Za-z0-9])', 'native_id_anywhere'),
                              (rf'{re.escape(stem)}(?=_|[^A-Za-z0-9]|$)', 'stem_anywhere'),
                              (rf'"{q}"', 'scan_number_string_anywhere')):
                n_hits = len(re.findall(pat, ser))
                field_kinds = {'masked_id_anywhere': ('masked_id_equal',), 'native_id_anywhere': ('native_id_equal',),
                               'stem_anywhere': ('native_id_equal', 'same_patient_stem'),
                               'scan_number_string_anywhere': ('scan_number_equal',)}[kind]
                n_field = sum(1 for (yy, qq, k, f, v) in relations if yy == y and qq == q and k in field_kinds)
                if n_hits > n_field:
                    relations.append((y, q, kind + '_extra', 'serialized payload', f'{n_hits} hits'))
            for src in ('text', 'text_first_if_different'):
                t = it[src] or ''
                for pat, kind in ((rf'CASE_{q}(?![0-9])', 'masked_id_in_text'),
                                  (rf'{re.escape(stem)}(?=_|[^A-Za-z0-9]|$)', 'stem_or_native_in_text')):
                    if re.search(pat, t):
                        relations.append((y, q, kind, src, pat))
    edges = {}
    for (y, q, kind, field, val) in relations:
        for x in by_query[q]:
            edges.setdefault((x, y), []).append({'linked_query': q, 'kind': kind, 'field_in_after_item': field,
                                                 'value': val})

    # ------------------------------------------------------------ rating ids and order
    r1_ids = {row['rating_id'] for row in csv.DictReader(open(ROUND1 / 'rating_packet_key' / 'rating_id_map.csv'))}
    rng = random.Random(seed)
    ALPH = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    CODES = ['ACT', 'ACC', 'EWS', 'LIVER', 'MEL', 'MISC', 'MPNST', 'MRT', 'NB', 'NF', 'NORMAL', 'OS', 'RB', 'RMS',
             'SS', 'WT', 'HB', 'HCC', 'CCS', 'SJ', 'CASE']
    used, rid = set(r1_ids), {}
    for it in items:
        while True:
            cand = 'R-' + ''.join(rng.choice(ALPH) for _ in range(4))
            if cand in used or any(c in cand[2:] for c in CODES):
                continue
            used.add(cand)
            rid[it['uid']] = cand
            break
    uids = [it['uid'] for it in items]
    attempts = 0
    while True:
        attempts += 1
        perm = uids[:]
        rng.shuffle(perm)
        pos = {u: i for i, u in enumerate(perm)}
        if all(pos[x] < pos[y] for (x, y) in edges):
            break
    pos = {u: i + 1 for i, u in enumerate(perm)}

    # ------------------------------------------------------------ §5 rubric paragraph (round-1 specification, verbatim)
    ctext = ROUND1_CONTRACT.read_text(encoding='utf-8')
    clines = ctext.split('\n')
    i0 = next(i for i, l in enumerate(clines) if l.startswith('Manual containment audit (A0 20 + A1 20;'))
    i1 = next(i for i in range(i0, len(clines)) if clines[i].endswith('no κ and no agreement/independent-validation claim.'))
    para_lines = clines[i0:i1 + 1]
    para = '\n'.join(para_lines)
    assert sha(para.encode()) == ROUND1_PARAGRAPH_SHA, 'rubric paragraph differs from the one quoted in round 1'
    ORDER_SENTENCE = ('The order of items has a constraint. Rate strictly in the given order and do not go back to '
                      'change an earlier rating after reading a later item.')
    NO_ORDER_SENTENCE = 'Rate strictly in the given order and do not go back to change an earlier rating after reading a later item.'

    # ------------------------------------------------------------ rater-facing content (built once, written twice)
    ordered = [item_by_uid[u] for u in perm]
    packet = []
    for k, it in enumerate(ordered, 1):
        s = split_sentences(it['text'])
        tf = it['text_first_if_different']
        sf = split_sentences(tf) if tf is not None else None
        packet.append({'position': k, 'rating_id': rid[it['uid']], 'payload': it['payload'], 'text': it['text'],
                       'text_first_if_different': tf,
                       'sentences': [{'sentence_no': j, 'text': t} for j, t in enumerate(s, 1)],
                       'first_draft_sentences': (None if sf is None else
                                                 [{'sentence_no': f'F{j}', 'text': t} for j, t in enumerate(sf, 1)])})
    files = {}
    files['packet.json'] = json.dumps(packet, ensure_ascii=False, indent=2) + '\n'
    md = ['# Summary rating packet', '',
          f'{len(packet)} items, in the order to be rated. Read `RATING_INSTRUCTIONS.md` first. '
          'Record ratings in `rating_form.csv` (one row per numbered sentence).', '']
    for it in packet:
        md += ['---', '', f'## Item {it["position"]} of {len(packet)} — {it["rating_id"]}', '',
               '### Input shown to the model (payload, as sent)', '', '```json',
               json.dumps(it['payload'], ensure_ascii=False, indent=2), '```', '', '### Summary (delivered)', '']
        md += [f'{x["sentence_no"]}. {x["text"]}' for x in it['sentences']] + ['']
        if it['first_draft_sentences'] is not None:
            md += [f'### {FIRST_LABEL}', '']
            md += [f'{x["sentence_no"]}. {x["text"]}' for x in it['first_draft_sentences']] + ['']
    files['packet.md'] = '\n'.join(md)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['rating_id', 'sentence_no', 'sentence_text', 'code', 'probability_statement_tag', 'support_trace', 'note'])
    n_sent = n_first = 0
    for it in packet:
        for x in it['sentences']:
            w.writerow([it['rating_id'], x['sentence_no'], x['text'], '', '', '', ''])
            n_sent += 1
        for x in (it['first_draft_sentences'] or []):
            w.writerow([it['rating_id'], x['sentence_no'], x['text'], '', '', '', ''])
            n_first += 1
    files['rating_form.csv'] = buf.getvalue()
    n_fd = sum(it['first_draft_sentences'] is not None for it in packet)
    fd_line = ('' if n_fd == 0 else
               (f'- One item also shows a "{FIRST_LABEL}". Rate its sentences too; they are numbered F1, F2, ... in the form.\n'
                if n_fd == 1 else
                f'- {n_fd} items also show a "{FIRST_LABEL}". Rate their sentences too; they are numbered F1, F2, ... in the form.\n'))
    files['RATING_INSTRUCTIONS.md'] = f"""# Rating instructions

## What you are rating

Each item is one short summary written by a language model. With each summary you see the input the model
was given for that case (the "payload": the masked current case, its five nearest reference slides with their
labels, sites and similarity values, and the neighbour-based probability line). For every sentence of the
summary, decide whether what it says is supported by the input shown, and record that in `rating_form.csv`.

You rate each summary against its own payload only, not against what you may know about the case or the
reference library.

## Files

- `packet.md` (readable) and `packet.json` (same content, machine-readable): {len(packet)} items, in order.
  Each item has a rating ID (for example `{packet[0]['rating_id']}`), the payload as sent, and the delivered
  summary split into numbered sentences.
{fd_line}- `rating_form.csv`: one row per sentence, with `rating_id`, `sentence_no` and `sentence_text` filled in.
  Fill in `code`, `probability_statement_tag` (leave empty if not a probability statement), `support_trace`
  and `note`. The file is UTF-8.
- Sentences were split mechanically (rule below). If a split looks wrong, rate the pieces as given and say
  so in `note`.

## Context for reading the payload

- The same fixed instructions were given to the model for every item (not reproduced here). Two field
  conventions stated in them, quoted: "similarity score (lower is better)" and "treatment=Y means treated, N
  means untreated".
- Besides the payload, the system also passed the model a few statistics computed mechanically from the
  payload (diagnosis counts and the most frequent diagnosis, mean/minimum/maximum similarity, the number of
  neighbours whose diagnosis differs from the most frequent one, the number of neighbours whose discrepant
  field is Y) and a 200-word target. They add no information beyond the payload; a sentence that restates
  them is checked against the payload, directly or by arithmetic.

## Rubric (contract section 5, quoted verbatim)

{chr(10).join('> ' + l if l else '>' for l in para_lines)}

## Order

{ORDER_SENTENCE if edges else NO_ORDER_SENTENCE}

## Sentence-splitting rule

{SPLITTER_RULE}
"""

    # ------------------------------------------------------------ self-checks (blinding)
    checks = {}
    quoted = '\n'.join('> ' + l if l else '>' for l in para_lines)
    forbidden = ['a0_masked_posterior', 'a1_opaque', 'a2_cf_plurality', 'repeat', 'swap', 'plurality',
                 'counterfactual', 'leak', 'flag', 'warning', 'S=',
                 # round blinding (builder-authored text only; item content is verbatim model output/payload)
                 'round', 'rounds', 'second', 'version', 'revised instruction', 'new instruction', 'pilot', 's5b',
                 's5c', 'pass 031', 'previous', 'differing']
    round1_terms = set(forbidden[:11])
    all_uids = [r['run_uid'] for r in rows]

    def strip_item_content(s):
        for it in packet:
            for t in [json.dumps(it['payload'], ensure_ascii=False, indent=2), it['text'], it['text_first_if_different'] or '']:
                if t:
                    s = s.replace(t, '')
            for x in it['sentences'] + (it['first_draft_sentences'] or []):
                s = s.replace(x['text'], '')
        return s

    a_res = {}
    for f in RATER_FILES:
        s = files[f]
        if f == 'RATING_INSTRUCTIONS.md':
            unq = s.replace(quoted, '').replace(SPLITTER_RULE, '')
        elif f == 'packet.json':
            unq = json.dumps([{k: v for k, v in it.items() if k not in ('payload', 'text', 'text_first_if_different',
                                                                       'sentences', 'first_draft_sentences')}
                              for it in packet]) + ' ' + ' '.join(sorted({k for it in packet for k in it}))
        else:
            unq = strip_item_content(s)
        res = {}
        for term in forbidden + all_uids:
            fl = re.IGNORECASE if term != 'S=' else 0
            pat = re.escape(term) if (term in round1_terms or term in all_uids) else rf'\b{re.escape(term)}\b'
            in_builder = bool(re.search(pat, unq, fl))
            n_total = len(re.findall(pat, s, fl))
            if n_total:
                res[term] = {'total_hits': n_total, 'in_builder_authored_text': in_builder}
            assert not in_builder, (f, term)
        a_res[f] = res
    checks['a_forbidden_strings_builder_text'] = a_res
    checks['c_counts'] = {'items': len(packet), 'sentences_delivered': n_sent, 'sentences_first_draft': n_first,
                          'items_first_draft': n_fd, 'excluded_no_text': verif['excluded_no_text'],
                          'first_call_text_used': verif['first_call_text_used']}
    checks['d_order'] = {'precedence_edges': len(edges), 'satisfied': sum(pos[x] < pos[y] for (x, y) in edges),
                         'rejection_sampling_attempts': attempts}
    checks['e_payload_custody'] = {k: v for k, v in verif.items()}
    checks['quoted_paragraph'] = {'contract': str(ROUND1_CONTRACT), 'contract_lines': [i0 + 1, i1 + 1],
                                  'sha256_of_paragraph': sha(para.encode()), 'equals_round1_quote': True}
    checks['rating_ids_disjoint_from_round1'] = not (set(rid.values()) & r1_ids)

    # ------------------------------------------------------------ write (two identical packets + key)
    targets = [out_root / p for p in PACKETS] + [out_root / 'rating_packet_key']
    for t in targets:
        t.mkdir(parents=True, exist_ok=False)
    for p in PACKETS:
        for f in RATER_FILES:
            (out_root / p / f).write_text(files[f], encoding='utf-8')
        with open(out_root / p / 'sha256.txt', 'w') as fh:
            for f in RATER_FILES:
                fh.write(f'{fsha(out_root / p / f)}  {f}\n')
    for f in RATER_FILES + ('sha256.txt',):
        assert fsha(out_root / PACKETS[0] / f) == fsha(out_root / PACKETS[1] / f), f
    checks['f_hashes_rater'] = {f: fsha(out_root / PACKETS[0] / f) for f in RATER_FILES}
    checks['f_packets_identical'] = True
    KEY = out_root / 'rating_packet_key'
    with open(KEY / 'rating_id_map.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['rating_id', 'run_uid', 'arm', 'query', 'repeat', 'position', 'output_state', 'payload_sha256',
                    'text_sha256'])
        for u in perm:
            it = item_by_uid[u]
            w.writerow([rid[u], u, it['arm'], it['query'], it['repeat'], pos[u], it['output_state'],
                        it['payload_sha256'], sha(it['text'].encode('utf-8'))])
    oc = {'seed': seed, 'seed_source': 'argument' if a.seed is not None else 'secrets.randbits(48)',
          'relation': "X before Y iff X's query identity (CASE_<scan>, scan number, native ID or patient stem) "
                      "appears in item Y (round-1 E7-A3(1) rule, A0 items only)",
          'raw_relations': [{'after_item': y, 'linked_query': q, 'kind': k, 'field': f, 'value': v}
                            for (y, q, k, f, v) in relations],
          'precedence_edges': [{'before': x, 'before_rating_id': rid[x], 'after': y, 'after_rating_id': rid[y],
                                'before_position': pos[x], 'after_position': pos[y], 'reasons': rs}
                               for (x, y), rs in sorted(edges.items())],
          'final_permutation': [{'position': pos[u], 'rating_id': rid[u], 'run_uid': u} for u in perm]}
    (KEY / 'order_constraints.json').write_text(json.dumps(oc, indent=1, ensure_ascii=False) + '\n')
    (KEY / 'build_checks.json').write_text(json.dumps(checks, indent=1, ensure_ascii=False, default=str) + '\n')
    summary = {'n_items': len(packet), 'n_sentences_delivered': n_sent, 'n_sentences_first_draft': n_first,
               'n_precedence_edges': len(edges), 'attempts': attempts, 'run_dir': str(run),
               'run_fake': bool(meta.get('fake')), 'script_sha256': fsha(__file__),
               'round1_builder_sha256': '84c4f6c7cb4f41229fac146a5e5adc46af3f8fc475b7bfcec6b5404ea022e011'}
    (KEY / 'build_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    with open(KEY / 'sha256.txt', 'w') as fh:
        for f in sorted(p for p in KEY.iterdir() if p.name != 'sha256.txt'):
            fh.write(f'{fsha(f)}  {f.name}\n')
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
