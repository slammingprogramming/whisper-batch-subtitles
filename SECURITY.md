# Security Policy

## Supported Versions

This project is actively maintained on the `main` branch. Security fixes target the latest released
version; there is no long-term support for older releases at this time. If you're not on the latest
version, please update before reporting an issue in case it's already fixed.

## Reporting a Vulnerability

**Please do not open a public GitHub issue with details of a security vulnerability.** Public issues are
for functional bugs and feature requests — a vulnerability report there is visible to everyone, including
anyone who might exploit it, before a fix ships.

Instead:

1. **Open a minimal GitHub issue that states only that a security issue exists**, without any technical
   detail — for example, "Security issue: reaching out via SimpleX." This starts a paper trail and lets
   the report be attributed to your GitHub account.
2. **Contact the maintainer privately over SimpleX**: <https://smp14.simplex.im/a#3gZ-zeHs4QrFZKLAN0o3SC_XQJXhj1eYBVTO_c0FAtg>
3. **Verify each other** before any vulnerability details are exchanged. This is a mutual
   public-key-signature exchange referencing the issue opened in step 1, which:
   - Confirms you control the GitHub account that opened the issue, so the report can be attributed to
     you correctly.
   - Confirms you're talking to the actual maintainer, not an impersonator.
4. Once verified, we'll discuss the full details of the vulnerability over SimpleX.

After you've completed this verification once, you're verified going forward — future reports can go
straight to a SimpleX message without repeating the handshake.

### What to expect

- Acknowledgment of your report once the SimpleX verification is complete.
- Ongoing communication as the issue is investigated and fixed.
- Credit in the eventual fix's release notes, if you'd like it (let us know your preference — credited by
  name/handle, credited anonymously, or not mentioned at all).
- Coordinated disclosure: please give us a reasonable window to ship a fix before any public write-up.

### Why the verification step?

It lets a report be both **attributable** (linked to a real GitHub identity, so credit and follow-up work
correctly) and **private** (full vulnerability details never touch a public issue tracker or an
unauthenticated channel) at the same time, without requiring you to create a permanent account anywhere
just to report one issue.

## Scope

This is a local command-line tool that processes media files on the machine it runs on. Security-relevant
areas most worth a careful look:

- Command construction for `ffmpeg`/`ffprobe` subprocess calls (`whisper_batch_subtitles/ffmpeg.py`,
  `whisper_batch_subtitles/media.py`)
- The optional local web dashboard (`whisper_batch_subtitles/webui.py`, the `serve` command) — it has **no
  authentication** by design and is meant for `127.0.0.1`/trusted-network use only; if you find a way it
  leaks data beyond what an unauthenticated local reader is already expected to see, or a way to reach it
  from outside its bound interface, that's worth reporting
- Handling of untrusted file paths/names during recursive media discovery
- Anything involving API keys or auth tokens (DeepL, pyannote) in config files, logs, or saved profiles
