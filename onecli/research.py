"""Deterministic source packets. Source text is evidence, never instructions.

Ranking is a navigation aid, not a semantic coverage verifier. Missing sections
and omitted paragraphs must stay visible so the caller can request evidence.
"""
import re
from html.parser import HTMLParser


def sections(html):
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.title = []
            self.body = []
            self.heading = False
            self.hidden = 0
            self.anchor = ""
            self.current = None

        def flush(self):
            if self.current is not None:
                text=re.sub(r'[ \t]+',' ',''.join(self.body)).strip()
                self.rows.append({'id':self.anchor or str(len(self.rows)+1),
                                  'heading':self.current,'text':text})
            self.body=[]
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "svg", "nav"):
                self.hidden += 1
            if self.hidden:
                return
            if tag in ('h1','h2','h3','h4','h5','h6'):
                self.flush()
                self.heading = True
                self.title = []
                self.anchor = dict(attrs).get("id", "")
            elif tag in ("p", "li", "pre", "div", "br"):
                self.body.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "svg", "nav"):
                self.hidden = max(0, self.hidden - 1)
                return
            if self.hidden:
                return
            if tag in ('h1','h2','h3','h4','h5','h6'):
                self.current = re.sub(r"\s+", " ", "".join(self.title)).strip()
                self.heading = False
            elif tag in ("p", "li", "pre", "div"):
                self.body.append("\n")

        def handle_data(self,data):
            if not self.hidden:
                (self.title if self.heading else self.body).append(re.sub(r'\s+',' ',data))

    p = Parser()
    p.feed(html)
    p.flush()
    return p.rows


def select_sections(html, query='', requested='', limit=5500):
    """Preserve whole sections; never claim lexical matches establish coverage."""
    rows=sections(html)
    if not rows:
        return "No section headings found. Evidence unavailable in section mode."
    terms=set(re.findall(r'\w+',query.lower()))
    requested_ids=[s.strip() for s in requested.split(',') if s.strip()]
    ranked=sorted(rows,key=lambda s:-len(terms & set(re.findall(r'\w+',s['heading'].lower()+' '+s['text'].lower()))))
    if requested_ids:
        ranked = [s for s in rows if s["id"] in requested_ids]
    elif terms:
        ranked = [
            s
            for s in ranked
            if terms
            & set(re.findall(r"\w+", s["heading"].lower() + " " + s["text"].lower()))
        ]
    picked = []
    used = 0
    for s in ranked:
        part=f"[{s['id']}] {s['heading']}\n{s['text']}"
        if used+len(part)+2<=limit:
            picked.append((s["id"], part))
            used += len(part) + 2
    ids={sid for sid,_ in picked}
    catalog='\n'.join(f"{s['id']}: {s['heading']} ({'included' if s['id'] in ids else 'omitted'})" for s in rows)
    if len(catalog)>1500:
        catalog=catalog[:1450]+'\n[index truncated; more sections exist]'
    missing=[sid for sid in requested_ids if sid not in ids]
    return ('Section index (omitted sections may contain required qualifications):\n'+catalog+
            '\nSelection is lexical, not proof that every question is answered. '
            'Check each subquestion against evidence; request omitted section IDs if needed.\n'+
            ('Requested but unavailable or over budget: '+', '.join(missing)+'\n' if missing else '')+
            '\n\n'.join(part for _,part in picked))

