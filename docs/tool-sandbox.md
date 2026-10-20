# Offline shell snapshots

`onecli.sandbox.shell.run_shell(source, staging_root, command, ...)` is a caller-owned
execution primitive. The caller fixes and approves both canonical absolute paths;
the model supplies only shell text. Source must be a dedicated, credential-free
workspace owned and frozen by the trusted caller throughout copying. Source and
staging must be disjoint. Home, root and home ancestors are rejected; that check
does not establish that an arbitrary other directory is safe to disclose.

The sandbox uses a neutral `/workspace` mount and a bubblewrap namespace. It
does not share the caller's network, `/etc` or authentication grants, and it
does not alter the host's network rules.

Each invocation copies regular single-link files and directories to a new private
staging directory. All symlinks, hardlinks, sockets and special files are rejected.
The snapshot byte and entry caps apply before execution. No source files are
mounted into the jail. Final file opens reject symlinks and verify inode identity;
this is not a guarantee against concurrent malicious host directory replacement.

The jail unshares network, PID, user, IPC, UTS and cgroup namespaces, drops all
capabilities, prohibits nested user namespaces, clears environment and inherited
file descriptors, and reads only the trusted `/usr` runtime plus synthetic proc
and device mounts. Host home and `/etc` are absent. `/workspace` is the sole
persistent writable mount. `/usr` is trusted operating-system installation content;
operators must not place private credentials or listening sockets there.

Shell stdout and stderr share a bounded capture. Wall timeout and output overflow
kill the launcher process group; PID namespace teardown removes descendants.
The host invokes `prlimit` with hard limits: zero core dumps, 16 MiB per file,
512 MiB virtual address space per process, 30 CPU seconds per process, 64 file
descriptors per process. These are **not aggregate quotas**. Many processes or files
can still exhaust shared host resources; for aggregate cgroup limits use
`run_bounded_shell` (see bounded-sandbox.md), and filesystem quotas are the host's.

There is no fallback if bubblewrap/prlimit is absent or namespace setup fails.
Namespace setup errors return nonzero with captured diagnostics. The returned
`SandboxResult` carries `returncode`, `output`, `workspace`, `timed_out` and
`output_limited`. Workspace output is retained for explicit host inspection; there
is no automatic writeback, import or execution of generated artifacts. Generated
symlinks, sockets and special files remain untrusted; a snapshot taken from that
workspace rejects them.
No package downloads or networked build tools work in this mode.

This primitive provides an offline execution boundary, not a complete hostile-code
platform, provider-price attestation or blanket spending guarantee. It does not
sandbox the host CLI itself, external model providers, hooks or MCP services.

`read_artifact(result, relative_path, max_bytes=1048576)` reads one selected output
as bytes for host review, without importing it. The host retains a genuine result
from its completed call; caller-constructed results are not authenticated
capabilities. Every directory component is opened descriptor-relative with
`O_NOFOLLOW`, anchored from the canonical workspace root. Absolute paths, empty,
`.` and `..` components, symlink parents/finals, nonregular files and hardlinked
files are rejected. Both metadata and actual reads enforce the byte limit
(maximum 16 MiB). No sandbox process or hostile host writer may concurrently
mutate the result tree during review. Rendering, importing or executing reviewed
bytes requires separate validation; reading does not certify their contents.
