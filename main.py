import model 
import torch 

def main():
    m = model.create()
    print(m(torch.randn([1,3,448,448])).shape)

if __name__ == '__main__':
    main()