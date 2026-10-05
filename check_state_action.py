import pyarrow.parquet as pq
import numpy as np
import glob
import os

root = r"D:\datasets\toutV2.1\data"
files = sorted(
    glob.glob(os.path.join(root, "**", "episode_*.parquet"), recursive=True)
)[:10]

print("episodes =", len(files))
print("episode | joint | eef_pos | eef_ori")
print("-" * 70)

for p in files:
    d = pq.read_table(p).to_pandas()

    # Joint: (N, 14)
    joint_state = np.stack(
        d["observation.states.joint.position"].to_numpy()
    ).astype(np.float64)

    joint_action = np.stack(
        d["actions.joint.position"].to_numpy()
    ).astype(np.float64)

    # EEF position: (N, 2, 3)
    eef_pos_state = np.stack([
        np.stack(x) for x in d["observation.states.end.position"]
    ]).astype(np.float64)

    eef_pos_action = np.stack([
        np.stack(x) for x in d["actions.end.position"]
    ]).astype(np.float64)

    # EEF orientation: (N, 2, 4)
    eef_ori_state = np.stack([
        np.stack(x) for x in d["observation.states.end.orientation"]
    ]).astype(np.float64)

    eef_ori_action = np.stack([
        np.stack(x) for x in d["actions.end.orientation"]
    ]).astype(np.float64)

    joint_equal = np.array_equal(joint_state, joint_action)
    eef_pos_equal = np.array_equal(eef_pos_state, eef_pos_action)
    eef_ori_equal = np.array_equal(eef_ori_state, eef_ori_action)

    print(
        f"{os.path.basename(p)} | "
        f"{joint_equal} | "
        f"{eef_pos_equal} | "
        f"{eef_ori_equal}"
    )

    if not (joint_equal and eef_pos_equal and eef_ori_equal):
        print(
            "  max diff:",
            "joint=", np.max(np.abs(joint_state - joint_action)),
            "eef_pos=", np.max(np.abs(eef_pos_state - eef_pos_action)),
            "eef_ori=", np.max(np.abs(eef_ori_state - eef_ori_action))
        )

print("-" * 70)
print("Done.")
