"""
DL perception. Owner: Anvita Suresh.

Must match the contract in DL_INTERFACE.md:
    detect(frame, cfg) -> {'present': bool,
                           'distance': int or None,
                           'type': 'ground' | 'flying' | None,
                           'height': int or None}

Replace the NotImplementedError below with the DL implementation (CNN,
YOLO, VLM, etc.). Configuration for the model lives under the `dl:`
section in app/config.yaml. Do not change existing keys.
"""


def detect(frame, cfg):
    raise NotImplementedError(
        'perception_dl.detect is a stub. '
        'Implement the DL detector here. See DL_INTERFACE.md and TODO_DL.md.'
    )
