# Wuji Hand 2.2 (Beta) — fingertip tactile modules never enumerate

**Device:** Wuji Hand 2, right, serial `WH2KA01260824017`, hw_version 0.2.0.
**Firmware:** v2.5.1 at first; upgraded with Wuji CLI 2026.11.9 to **v3.0.0** on 2026-11-13 (upgrade OK, 124 s).
**SDK:** wuji_sdk 2026.11.9 (Windows, Python 3.12). Connection: Ethernet/UDP, hand at 192.168.1.111, host 192.168.1.50.
**Hardware observed:** every fingertip has the silicone sensor pad over a metal core with an orange flex circuit
running into the core (photos available). Cables were checked by the owner.

## Symptoms (identical on v2.5.1 and v3.0.0, before and after a software reboot, with joints disabled and enabled)

- `node_online` = 16236015 (0xF7BDEF): joint nodes 1–4 online on all five finger buses, **tactile slot (node 5) absent on all five**.
- `slave_version` lists 20 entries (joints only).
- `Hand2CommSummary.tactile_online_mask = 0`, `tactile_timeout_total = [0,0,0,0,0]` (the master never polls the modules).
- `hand.fingertip_online()` → `[False]*5`.
- v2.5.1: `fingertip_<f>_status().get()` → `InternalError`; `fingertip_<f>_info().get()` still returned the descriptors
  (thumb 40 points device_type 4, others 34 points device_type 3, point_array format, 100 Hz).
- v3.0.0: `fingertip_<f>_status()/info()` → `NodeOfflineError: tactile node offline: <finger> (node not on the bus; check that
  the fingertip module is connected)`; `GET <finger>.tactile.echo` → "Node offline: the target node is not on the bus, so the
  request was never sent".
- `fingertip/<finger>/data` subscriptions deliver 0 frames.
- Joints are healthy: 20/20 online, a stitched pen-spin clip plays with mean joint error 1.85° (v3.0.0; 1.86° on v2.5.1).
- New on v3.0.0: encoder-1 data-quality warnings E-01-08/21/23-0003 on index_pip (F1S3), pinky_mcp_flex (F4S1),
  pinky_pip (F4S3); they clear automatically.

## Reports in this folder

- `probe_2026-11-13_184739.txt` — full dump on v2.5.1 (nodes, topics, params, descriptors, comm summary)
- `reboot_probe_2026-11-13_192609.txt` — before/after a software reboot (v2.5.1)
- `wuji_cli_2026-11-13_195354.txt` — `wuji doctor`, `wuji upgrade --check`, parameter reads
- `wuji_upgrade_2026-11-13_200001.txt` — the 2.5.1 → 3.0.0 upgrade log and the post-upgrade probe

## Question for Wuji support

Was this unit built with the fingertip tactile electronics (tactile node 5 on each finger bus), or only with the
sensor pads/flex? If the electronics are present, what connects node 5 to the bus and how should it be checked or
re-seated? Is a retrofit or RMA needed?
