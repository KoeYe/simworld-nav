# nav-render/v0 golden fixtures

Byte-exact serialisations of the wire examples in `docs/LIVE_UE_SPEC.md`
section 3. Serialisation convention: JSON with 2-space indent, sorted keys,
one trailing newline (`embodiedbench.runtime.live.protocol.dumps`). The
`sha256` in `render_response.json` is the SHA-256 of the empty string -- a
recognisable placeholder, since the spec's example elides the value.

These same files are copied verbatim into the SimWorld2 repo
(`feat/nav-live-rollouts`), so both sides of the protocol are pinned to one
set of bytes. Change them in both places or not at all.
