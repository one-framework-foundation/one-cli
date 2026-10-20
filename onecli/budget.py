"""Durable, conservative authorization ledger for a shared request tree.

The caller must supply a *known upper bound*, never a likely-cost estimate.
Each actual network attempt needs a fresh operation ID, including retries.
Idempotent reservation lookup does not authorize replay of a network request.
Reservations are never refunded: timeouts, process death and unknown provider
outcomes continue to consume their entire bound. This ledger does not itself
enforce provider limits or defend against an actor replacing its database.
"""

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path
import sqlite3


NANODOLLARS = 1_000_000_000
MAX_NANODOLLARS = (1 << 63) - 1


class BudgetError(RuntimeError):
    """Budget state or input cannot safely authorize a request."""


class BudgetExceeded(BudgetError):
    """The request's bound exceeds the remaining shared budget."""


def usd_to_nanos(value, *, round_up: bool = False) -> int:
    """Convert USD conservatively, rejecting invalid or unrepresentable values.

    Budgets round down; request bounds round up. Strings and Decimal preserve
    exact input. Floats use their decimal spelling, never binary multiplication.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise BudgetError("USD amount must be a finite positive number")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount <= 0:
            raise BudgetError("USD amount must be a finite positive number")
        # Bound the value before arithmetic, including pathological exponents.
        if amount > Decimal(MAX_NANODOLLARS) / NANODOLLARS:
            raise BudgetError("USD amount exceeds ledger capacity")
        with localcontext() as ctx:
            ctx.prec = max(40, len(amount.as_tuple().digits) + 10)
            nanos = int((amount * NANODOLLARS).to_integral_value(
                rounding=ROUND_CEILING if round_up else ROUND_FLOOR))
    except (InvalidOperation, ValueError, OverflowError, TypeError) as exc:
        raise BudgetError("Invalid USD amount") from exc
    if not 0 < nanos <= MAX_NANODOLLARS:
        raise BudgetError("USD amount is below ledger precision")
    return nanos


class RequestBudget:
    """SQLite reservations shared by all processes using the same ledger path.

    The first initializer fixes the integer nanodollar limit forever. Every
    subsequent initializer must supply a limit rounding down to that same value.
    Use create=False for children and crash recovery, so missing/empty state
    cannot accidentally start a new budget. Only an explicit new-tree action
    should use create=True. BEGIN IMMEDIATE serializes each check-and-insert
    across processes; synchronous FULL makes the commit precede authorization.
    Missing/corrupt state during use raises BudgetError instead of resetting it.
    """

    def __init__(self, path, limit_usd, *, create=True):
        self.path = Path(path).absolute()
        self.limit_nanos = usd_to_nanos(limit_usd)
        with self._transaction(create=create) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                if not create:
                    raise BudgetError("Existing budget ledger has no state")
                conn.execute("CREATE TABLE budget_meta ("
                             "id INTEGER PRIMARY KEY CHECK(id = 1), "
                             "version INTEGER NOT NULL CHECK(version = 1), "
                             "limit_nanos INTEGER NOT NULL "
                             "CHECK(typeof(limit_nanos) = 'integer' AND limit_nanos > 0))")
                conn.execute("CREATE TABLE reservations ("
                             "operation_id TEXT PRIMARY KEY NOT NULL, "
                             "upper_cost_nanos INTEGER NOT NULL "
                             "CHECK(typeof(upper_cost_nanos) = 'integer' AND upper_cost_nanos > 0))")
                conn.execute("INSERT INTO budget_meta VALUES (1, 1, ?)", (self.limit_nanos,))
            self._read_state(conn)

    @contextmanager
    def _transaction(self, *, create=False):
        conn = None
        try:
            mode = "rwc" if create else "rw"
            conn = sqlite3.connect(self.path.as_uri() + "?mode=" + mode,
                                   uri=True, timeout=30, isolation_level=None)
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise BudgetError("Budget ledger unavailable or corrupt") from exc
        finally:
            if conn is not None:
                # close rolls back an incomplete transaction, including a failed
                # commit. The caller never receives authorization on an error.
                conn.close()

    @staticmethod
    def _check_operation_id(operation_id):
        if (not isinstance(operation_id, str) or not operation_id.strip()
                or len(operation_id) > 1024 or "\x00" in operation_id):
            raise BudgetError("Operation ID must be a nonempty string of at most 1024 characters")

    def _read_state(self, conn):
        meta = conn.execute("SELECT id, version, limit_nanos FROM budget_meta").fetchall()
        if (len(meta) != 1 or any(type(v) is not int for v in meta[0])
                or meta[0] != (1, 1, self.limit_nanos)):
            raise BudgetError("Budget limit is immutable or ledger metadata is invalid")
        reservations = {}
        total = 0
        for operation_id, cost in conn.execute(
                "SELECT operation_id, upper_cost_nanos FROM reservations"):
            self._check_operation_id(operation_id)
            if (operation_id in reservations or type(cost) is not int
                    or not 0 < cost <= MAX_NANODOLLARS):
                raise BudgetError("Invalid reservation state")
            reservations[operation_id] = cost
            total += cost
        if total > self.limit_nanos:
            raise BudgetError("Ledger reservations exceed its immutable limit")
        return reservations, total

    def reserve(self, operation_id: str, upper_cost_usd) -> int:
        """Commit a known request bound; return its integer nanodollar amount."""
        return self.reserve_nanos(operation_id, usd_to_nanos(upper_cost_usd, round_up=True))

    def reserve_nanos(self, operation_id: str, upper_cost_nanos: int) -> int:
        self._check_operation_id(operation_id)
        if type(upper_cost_nanos) is not int or not 0 < upper_cost_nanos <= MAX_NANODOLLARS:
            raise BudgetError("Request bound must be positive integer nanodollars")
        with self._transaction() as conn:
            reservations, total = self._read_state(conn)
            if operation_id in reservations:
                if reservations[operation_id] != upper_cost_nanos:
                    raise BudgetError("Operation ID already has a different request bound")
            else:
                if upper_cost_nanos > self.limit_nanos - total:
                    raise BudgetExceeded("Request upper bound exceeds remaining shared budget")
                conn.execute("INSERT INTO reservations VALUES (?, ?)",
                             (operation_id, upper_cost_nanos))
        return upper_cost_nanos

    def reserved_nanos(self) -> int:
        with self._transaction() as conn:
            _, total = self._read_state(conn)
            return total

