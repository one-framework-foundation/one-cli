# workgate

`one workgate` applies the same versioned workflow policy for people, agents,
CI and other tools. It does not run a model. A work record embeds the complete
policy and its SHA-256 hash, so an active workflow cannot silently move to a
new policy version.

Start a record with a concrete outcome and stop condition:

```bash
one workgate start --goal "find unsafe failover gaps" \
  --expected "a merged regression guard" \
  --stop "stop after one round without a product change"
```

Check an event directly, from `@evidence.json`, or from stdin. Exit `0` means
pass/warn; exit `2` means block or invalid input. `--json` makes stdout a
single machine-readable object.

```bash
one workgate --json check --event before_merge --evidence \
  '{"regression_failed_before_fix":true,"ci_green":true,"scratch_only":false}'
```

The built-in `experimental-code@1` profile caps meta-evaluation, rejects tests
that mock private functions, requires fail-before proof, and distinguishes
durable work from scratch output. `one workgate profiles` lists built-ins.

Other domains use the same engine by passing a JSON policy file to
`start --policy path.json`. A policy names an integer version and maps events
to declarative rules. Existing records retain their embedded policy; changing
a policy file affects only newly started work.
