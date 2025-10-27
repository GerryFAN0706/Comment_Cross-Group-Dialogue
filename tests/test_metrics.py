import pandas as pd
from src.metrics.network_metrics import reciprocity_stats, branching_factor

def test_reciprocity_basic():
    # triangle u->v, v->u, u->w
    edges = pd.DataFrame({"u":["u","v","u"], "v":["v","u","w"]})
    r = reciprocity_stats(edges, op_id=None, exclude_op=False)
    assert 0 < r["R"] <= 1

def test_branching_basic():
    edges = pd.DataFrame({"u":["a","b","c"], "v":["OP","OP","b"]})
    bf = branching_factor(edges, op_id="OP")
    assert "BF" in bf and "BF_proxy" in bf
