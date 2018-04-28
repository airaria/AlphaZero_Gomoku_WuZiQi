import torch
import argparse
import Network
from config import BOARD_SIZE,LEARNING_RATE,L2_WEIGHT,N_EVALUATE
import sys
from PIL import Image
from io import BytesIO

def onnx_to_caffe2(onnx_model, output, init_net_output):
    onnx_model_proto = ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())

    init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model_proto)
    init_net_output.write(init_net.SerializeToString())
    output.write(predict_net.SerializeToString())

    return init_net,predict_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--batch_size',required=True,type=int)
    parser.add_argument('--output_file',required=True,type=str)
    parser.add_argument('--show_net',action='store_true')
    args = parser.parse_args()

    #Load pytorch net
    model = Network.PoliycValueNet(BOARD_SIZE[0], BOARD_SIZE[1], 4)
    controller = Network.Controller(model, LEARNING_RATE,L2_WEIGHT,optim=None)
    controller.load_file(args.model_file)

    #export onnx
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(N_EVALUATE, 4,BOARD_SIZE[0], BOARD_SIZE[1])
        if controller.use_cuda:
            dummy_input = dummy_input.cuda()
        torch.onnx.export(model,dummy_input,args.output_file,verbose=True)

    #import caffe2 and onnx
    try:
        from caffe2.python import core, workspace
        from caffe2.proto.caffe2_pb2 import NetDef
        from caffe2.python import net_drawer
        from caffe2.python.onnx.backend import Caffe2Backend as c2
        from onnx import ModelProto
    except ImportError:
        print("Caffe2 or onnx is not installed, stop converting to pb file.")
        sys.exit()

    #convert to  caffe2's pb format
    onnx_model = open(args.output_file,'rb')
    output_net = open(args.output_file.split('.')[0]+'_net.pb','wb')
    output_init_net = open(args.output_file.split('.')[0]+'_init_net.pb','wb')
    init_net, predict_net = onnx_to_caffe2(onnx_model, output_net, output_init_net)

    #init_net = NetDef()
    #init_net.ParseFromString(init_net_data)
    #predict_net = NetDef()
    #predict_net.ParseFromString(predict_net_data)

    #show network
    if args.show_net:
        graph = net_drawer.GetPydotGraph(predict_net, rankdir="LR")
        Image.open(BytesIO(graph.create_png())).show()
