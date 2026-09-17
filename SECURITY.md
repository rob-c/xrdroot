# Security

## Reporting a vulnerability

Mail <robert.andrew.currie@gmail.com> with a description and, if you have one,
a reproducer. Please do not open a public issue for anything that lets one
party read or write another party's data. Expect an acknowledgement within a
few working days.

## What this package is trusted with

It parses a file format, and a file format is untrusted input. Everything to
do with credentials, TLS and talking to a server belongs to
[PyXRootDClient](https://github.com/rob-c/xrd) underneath it, whose
[SECURITY.md](https://github.com/rob-c/xrd/blob/main/SECURITY.md) is the
document for those. The threat model here has one party: whoever wrote the
bytes.

## What the implementation guarantees

**A malformed file is an error, not a crash or a wrong answer.** Every length
and offset read out of a file is checked against what is actually there before
it is used, and bytes that are not the format they claim raise `FormatError` -
a truncated download and an HTML error page from a proxy both look like this.

**Sizes in a file are claims, and are checked as such.** A compressed block
whose bytes are not all there is refused before anything is decompressed, and
one that gives back a different number of bytes than it promised raises
`FormatError` rather than being passed on. A tree is walked a basket at a time
whatever the file says it holds, so reading one costs the entries asked for
rather than the file.

**A class this reader cannot decode is refused by name.** A layout the file
does not describe, or an object that streams itself in a way this does not
implement, raises `UnsupportedFeatureError` naming the class. A plausible
misreading of physics data is worse than a refusal, and silently skipping a
column is the worst of both.

**Nothing in a file is executed, and nothing is imported because a file asked
for it.** A ROOT file names C++ classes; those names are matched against what
this package knows how to read, never used to resolve or load anything.
