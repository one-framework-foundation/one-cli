"""Descriptor-relative local storage. Never follow symlinks during traversal.

This protects built-in file operations from symlink substitution, not arbitrary
host programs or a hostile process running as the same user. Paths containing
symlinks are intentionally refused, except the operator-selected ~/.one-cli
state root; callers can explicitly name the real path.
"""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat
import uuid


def _open_component(part, flags, dir_fd):
    """Open a path component relative to dir_fd, refusing to follow symlinks
    except a root-owned symlink inside a root-owned, non-writable directory
    (e.g. macOS /var -> private/var, Linux /bin -> usr/bin). An unprivileged
    attacker cannot substitute a link there, so following it is safe; any
    other symlink is refused exactly as before.
    """
    try:
        return os.open(part, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno not in (errno.ENOTDIR, errno.ELOOP):
            raise
        try:
            link_info = os.stat(part, dir_fd=dir_fd, follow_symlinks=False)
            parent_info = os.fstat(dir_fd)
        except OSError:
            raise exc
        if not (
            stat.S_ISLNK(link_info.st_mode)
            and link_info.st_uid == 0
            and parent_info.st_uid == 0
            and parent_info.st_mode & 0o022 == 0
        ):
            raise exc
        return os.open(part, flags & ~os.O_NOFOLLOW, dir_fd=dir_fd)


@contextmanager
def directory(path, *, create=False, private=False):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # Do not normalize '..' before opening preceding components: abspath
    # would erase a symlink that must be refused before visiting its parent.
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    # Only the operator-selected state root may be a dotfiles symlink. Resolve
    # that root once, then retain descriptor-relative no-follow traversal for
    # everything below it (including sessions and their final files).
    state_root = Path.home() / ".one-cli"
    resolved_root = None
    if path.is_relative_to(state_root) and ".." not in path.parts:
        resolved_root = state_root.resolve()
        path = resolved_root / path.relative_to(state_root)
    parts = path.parts
    if create and ".." in parts:
        # Authorization covers the final destination, not mkdir side effects
        # in ancestors subsequently canceled by '..'. Refuse before mutation.
        raise PermissionError("creation paths cannot contain '..'; name the destination directly")
    fd = os.open(parts[0], flags)
    try:
        for index, part in enumerate(parts[1:], 1):
            if create:
                try:
                    os.mkdir(part, mode=0o700 if private else 0o777, dir_fd=fd)
                except FileExistsError:
                    pass
            child = _open_component(part, flags, fd)
            os.close(fd)
            fd = child
            if private and resolved_root is not None and Path(*parts[:index + 1]) == resolved_root:
                if os.fstat(fd).st_uid != os.getuid():
                    raise PermissionError("private state directory is not owned by this user")
                os.fchmod(fd, 0o700)
        if private:
            if os.fstat(fd).st_uid != os.getuid():
                raise PermissionError("private storage directory is not owned by this user")
            os.fchmod(fd, 0o700)
        yield fd
    finally:
        os.close(fd)


@contextmanager
def text_file(path, mode="r", *, private=False, errors="strict", newline=None):
    path = Path(path)
    if mode not in {"r", "w", "a", "x", "r+"}:
        raise ValueError("unsupported secure file mode")
    creating = mode in {"w", "a", "x"}
    if creating and ".." in path.parts:
        # Check the entire path, including a final '..', before creating parents.
        raise PermissionError("creation paths cannot contain '..'; name the destination directly")
    writing = mode != "r"
    flags = os.O_RDWR if writing else os.O_RDONLY
    flags |= os.O_NOFOLLOW | os.O_NONBLOCK
    if creating:
        flags |= os.O_CREAT
    if mode == "x":
        flags |= os.O_EXCL
    if mode == "a":
        flags |= os.O_APPEND
    with directory(path.parent, create=creating, private=private) as parent:
        fd = os.open(path.name, flags, 0o600 if private else 0o666, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise PermissionError("file operation requires a regular file")
            if writing and info.st_nlink != 1:
                raise PermissionError("refusing to modify a multiply-linked file")
            if private:
                if info.st_uid != os.getuid():
                    raise PermissionError("private storage file is not owned by this user")
                os.fchmod(fd, 0o600)
            if mode == "w":
                os.ftruncate(fd, 0)
            fh = os.fdopen(fd, mode, encoding="utf-8", errors=errors, newline=newline)
            fd = None
            with fh:
                yield fh
        finally:
            if fd is not None:
                os.close(fd)


def replace_text(path, text, *, errors="strict", newline=None):
    """Replace a workspace file with text, the old file or the new one surviving.

    The descriptor-relative twin of atomic_write_text for the file tools: the
    parents are walked without following a symlink, an existing target must be
    a regular, singly-linked file, and the temp file beside it is fsynced and
    renamed over it. A write that fails part way (a file-size limit, ENOSPC, a
    kill) leaves the original intact. A replaced file keeps its mode.
    """
    path = Path(path)
    if ".." in path.parts:
        raise PermissionError("creation paths cannot contain '..'; name the destination directly")
    with directory(path.parent, create=True) as parent:
        try:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                raise PermissionError("file operation requires a regular file")
            if info.st_nlink != 1:
                raise PermissionError("refusing to modify a multiply-linked file")
        tmp = f".{path.name}.{uuid.uuid4().hex}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o666 if info is None else stat.S_IMODE(info.st_mode), dir_fd=parent)
        try:
            if info is not None:
                os.fchmod(fd, stat.S_IMODE(info.st_mode))
            with os.fdopen(fd, "w", encoding="utf-8", errors=errors, newline=newline) as fh:
                fd = None
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            tmp = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp is not None:
                try:
                    os.unlink(tmp, dir_fd=parent)
                except FileNotFoundError:
                    pass
    return path


def atomic_write_text(path, text, *, private=False, exclusive=False):
    """Replace path with text so a crash leaves the old file or the new one.

    The temp file sits beside the target (same filesystem), is fsynced, then
    renamed over it with os.replace. The temp is removed on any failure.
    exclusive=True refuses to replace an existing file (hard-link publish, so
    the check and the create are one atomic step).
    """
    # State lives in operator-owned directories (~/.one-cli may be a dotfiles
    # symlink), so the parent is followed as Path.write_text did; only the temp
    # file itself is created exclusively and never through a symlink.
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # A replaced file keeps the mode the operator gave it.
        keep = 0o600 if private else stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        keep = None
    mode = 0o666 if keep is None else keep
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if keep is not None:
                os.fchmod(fh.fileno(), keep)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if exclusive:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_history(readline, path, *, limit=5000):
    """Persist the newest readline entries with private, no-follow storage.

    Writing through our descriptor avoids readline creating a world-readable
    file before chmod, or following an existing history symlink.
    """
    count = readline.get_current_history_length()
    with text_file(path, "w", private=True) as fh:
        for index in range(max(1, count - limit + 1), count + 1):
            entry = readline.get_history_item(index)
            if entry is not None:
                fh.write(entry + "\n")
