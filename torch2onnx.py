import torch
import argparse
import Network
from config import BOARD_SIZE,LEARNING_RATE,L2_WEIGHT,N_EVALUATE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--batch_size',required=True,type=int)
    parser.add_argument('--output',required=True,type=str)
    args = parser.parse_args()


    model = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    controller = Network.Controller(model, LEARNING_RATE,L2_WEIGHT,optim=None)
    controller.load_file(args.model_file)
    model.eval()

    #export ONNX
    dummy_input = torch.autograd.Variable(torch.randn(N_EVALUATE, 4,BOARD_SIZE[0], BOARD_SIZE[1]))
    if controller.use_cuda:
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model,dummy_input,args.output,verbose=True)
