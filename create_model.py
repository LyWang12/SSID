from transformer import *

def Creat_model(name=None, num_classes=5):
    if name == 'swin_tiny_patch4_window7_224':
        net = swin_tiny_patch4_window7_224(num_classes=num_classes)
    elif name == 'swin_small_patch4_window7_224':
        net = swin_small_patch4_window7_224(num_classes=num_classes)
    elif name == 'swin_base_patch4_window7_224':
        net = swin_base_patch4_window7_224(num_classes=num_classes)
    elif name == 'swin_base_patch4_window12_384':
        net = swin_base_patch4_window12_384(num_classes=num_classes)
    elif name == 'swin_base_patch4_window7_224_in22k':
        net = swin_base_patch4_window7_224_in22k(num_classes=num_classes)
    elif name == 'swin_base_patch4_window12_384_in22k':
        net = swin_base_patch4_window12_384_in22k(num_classes=num_classes)
    elif name == 'swin_large_patch4_window7_224_in22k':
        net = swin_large_patch4_window7_224_in22k(num_classes=num_classes)
    else:
        raise NameError("Unknow Model Name!")
    return net
