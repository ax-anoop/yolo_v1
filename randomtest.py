import torch 
import model as mc 

m = mc.create()
o = m(torch.randn([1,3,448,448]))
print(o.shape)