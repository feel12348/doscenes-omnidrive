_base_ = ['./mask_eva_lane_det_vlm.py']

# Only evaluate samples whose scenes have non-empty doScenes instructions.
# Update this path if your merged CSV lives elsewhere.
doscenes_csv = 'data/annotated_doscenes.csv'

data = dict(
    val=dict(
        doscenes_csv=doscenes_csv,
        only_doscenes_samples=True,
        enable_doscenes_instruction=True,
    ),
    test=dict(
        doscenes_csv=doscenes_csv,
        only_doscenes_samples=True,
        enable_doscenes_instruction=True,
    ),
)

model = dict(
    save_path='./test_log/exp04/',
)
