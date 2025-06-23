import json
from utils.matching_utils import match_features


def load_features(path):
    with open(path, "r") as f:
        return json.load(f)

broadcast_feats = load_features("data/broadcast_features.json")
tacticam_feats = load_features("data/tacticam_features.json")

player_id = 0
final_matches = {"broadcast": {}, "tacticam": {}}

for frame in tacticam_feats:
    tac_features = tacticam_feats[frame]

    # Find best matching broadcast frame (here using same frame name)
    if frame not in broadcast_feats:
        continue

    broad_features = broadcast_feats[frame]
    matches = match_features(tac_features, broad_features)

    tac_frame_result = {}
    broad_frame_result = {}

    for i, (tac_idx, broad_idx, _) in enumerate(matches):
        pid = f"player_{player_id}"
        tac_frame_result[pid] = tac_features[tac_idx]["bbox"]
        broad_frame_result[pid] = broad_features[broad_idx]["bbox"]
        player_id += 1

    final_matches["tacticam"][frame] = tac_frame_result
    final_matches["broadcast"][frame] = broad_frame_result

with open("outputs/mappings.json", "w") as f:
    json.dump(final_matches, f, indent=2)