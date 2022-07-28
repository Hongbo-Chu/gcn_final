# from reduce_backbone.resnet import buildresnet
from reduce_backbone.vit import buildvit
# from reduce_backbone.swin import buildswin
_model = {
#     "resnet34": buildresnet,
    "vit": buildvit,
#     "swin":buildswin
}

def build_model(bb_type):
    assert (bb_type in _model), "invalid backbone type, only vit, swin, resnet34"
    return _model[bb_type]()