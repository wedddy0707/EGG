import gc
import torch

def memReport ():
    i = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            i += 1
            # print(type(obj), obj.size())
    print('mem check',i)
    gc.collect()