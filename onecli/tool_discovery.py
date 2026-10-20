"""Request-local MCP discovery; selection never grants execution permission."""
import json
import math
import re

NAME = 'tool_search'
SCHEMA = {
    'type': 'function', 'function': {
        'name': NAME,
        'description': 'Find available MCP tools by keywords or exact tool name. Enables matching tool schemas on the next request. Search again to replace the selection.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'Tool name or keywords describing the needed capability.'}
        }, 'required': ['query']},
    },
}

# How many matching schemas one search may enable. Matches the default limit of
# the comparable search in Pi's codemode tool, so a broad query still narrows to
# a handful the model can read; the request budget below stays the hard ceiling.
MAX_SELECTED = 8
EXACT_NAME_BONUS = 1000.0
_K1, _B = 1.5, 0.75


def _tokens(text):
    return re.findall(r'[\w-]+', text.lower())


class Discovery:
    def __init__(self, schemas, window):
        self.catalog = {s['function']['name']: s for s in schemas
                        if s['function']['name'].startswith('mcp__')}
        self.base = [s for s in schemas if s['function']['name'] not in self.catalog]
        self.enabled = bool(self.catalog) and len(json.dumps(schemas)) > window
        self.selected = []
        # Keep schemas to a quarter-window character estimate, leaving the
        # bulk of the context for instructions, history and output.
        self.budget = max(0, window - len(json.dumps(self.base + [SCHEMA])))
        self.original = schemas
        self._docs = {name: _tokens(name + ' ' + schema['function'].get('description', ''))
                      for name, schema in self.catalog.items()}
        self._avg_len = (sum(len(doc) for doc in self._docs.values()) / len(self._docs)) \
            if self._docs else 0.0
        self._df = {}
        for doc in self._docs.values():
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def _bm25(self, terms, doc):
        """Rank a tool by the query terms it carries, rare terms weighing more.

        A raw match count ranks a tool that repeats a word every catalog entry
        shares above the one tool that names the capability being asked for;
        the inverse document frequency is what makes the specific tool win.
        """
        if not terms or not doc:
            return 0.0
        total = 0.0
        length = len(doc)
        for term in terms:
            tf = doc.count(term)
            df = self._df.get(term, 0)
            if not tf or not df:
                continue
            idf = math.log(1 + (len(self._docs) - df + 0.5) / (df + 0.5))
            total += idf * (tf * (_K1 + 1)) / (
                tf + _K1 * (1 - _B + _B * length / self._avg_len))
        return total

    def schemas(self):
        return self.base + [SCHEMA] + self.selected if self.enabled else self.original

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            return 'Provide a nonempty query.'
        exact = query.strip()
        terms = _tokens(query)
        ranked = []
        for name, schema in self.catalog.items():
            score = self._bm25(terms, self._docs[name])
            if name == exact:
                score += EXACT_NAME_BONUS
            if score > 0:
                ranked.append((-score, name, schema))
        self.selected = []
        used = 0
        omitted = []
        for _, name, schema in sorted(ranked, key=lambda row: (row[0], row[1])):
            size = len(json.dumps(schema))
            if len(self.selected) >= MAX_SELECTED or used + size > self.budget:
                omitted.append(name)
                continue
            self.selected.append(schema)
            used += size
        return json.dumps({'enabled': [s['function']['name'] for s in self.selected], 'other_matches': omitted[:10],
                           'note': 'Only selected schemas are enabled; narrow the query for other matches.'})
