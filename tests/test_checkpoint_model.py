from pathlib import Path

import torch

from reefformer.itransformer import ITransformer


ROOT = Path(__file__).resolve().parents[1]


def test_archived_checkpoint_loads_strictly():
    path = ROOT / "checkpoints/hainan_coraltemp/iT_L90_seed42.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ITransformer(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    sample = torch.zeros(2, 90, 154)
    with torch.no_grad():
        output = model(sample)
    assert output.shape == (2, 30, 154)
    assert torch.isfinite(output).all()
