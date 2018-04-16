import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os,glob

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PoliycValueNet(nn.Module):
    def __init__(self,width,height,in_channel):
        super(PoliycValueNet, self).__init__()
        self.width = width
        self.height = height

        self.init_block = nn.Sequential(
            conv3x3(in_channel, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res1 = BasicBlock(64,64)
        self.res2 = BasicBlock(64,64)

        downsample3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128))
        self.res3 = BasicBlock(64,128,stride=1,downsample=downsample3)
        self.res4 = BasicBlock(128,128)

        self.policy_head = nn.Sequential(
            nn.Conv2d(128,4,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(4*self.width*self.height, self.width*self.height)

        self.value_head = nn.Sequential(
            nn.Conv2d(128,2,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Linear(2*self.width*self.height,1)

    def forward(self,x):
        x = self.init_block(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        policy = self.policy_head(x)
        policy = policy.view(-1,4*self.height*self.width)
        policy = self.policy_fc(policy)

        value = self.value_head(x)
        value = value.view(-1,2*self.height*self.width)
        value = F.tanh(self.value_fc(value))

        return policy,value


class Controller():
    def __init__(self, model, lr,L2_weight=0,optim=None):
        self.model = model
        self.lr = lr
        self.L2 = L2_weight
        if optim is None:
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=lr,weight_decay=L2_weight)
        else:
            self.optimizer = optim

        self.use_cuda = torch.cuda.is_available()
        self.value_criterion = nn.MSELoss()

        if self.use_cuda:
            self.model.cuda()

    def loss(self,source_v,target_z,source_p,target_pi):
        value_loss  = self.value_criterion(source_v,target_z)
        policy_loss = ((-F.log_softmax(source_p,dim=-1) * target_pi).sum(dim=-1)).mean()
        return value_loss + policy_loss

    def predict(self,x):
        '''
        :param x: numpy array of shape (batch_size, channels, height, width)
        :return: policy, value.
                 policy is probs, numpy array of shape (batch_size, width*height(+1))
                 value between [-1,1], numpy array of shape (batch_size, 1)
        '''
        self.model.eval()
        x = Variable(torch.from_numpy(x).float())
        if self.use_cuda:
            x = x.cuda()
        policy,value = self.model(x)
        self.model.train()
        policy = F.softmax(policy,dim=-1).data.cpu().numpy()
        value =   value.data.cpu().numpy()
        return policy,value

    def train_on_batch(self,x,z,pi):
        '''
        all are of type np.float32
        :param x: numpy array of shape (batch_size, channels, width, beight)
        :param z: numpy array of shape (batch_size, 1) , {-1,1}
        :param pi: [0,1] probs, numpy array of shape (batch_size, width*height(+1))
        '''
        self.optimizer.zero_grad()
        x = Variable(torch.from_numpy(x))
        z = Variable(torch.from_numpy(z))
        pi = Variable(torch.from_numpy(pi))

        if self.use_cuda:
            x = x.cuda()
            z = z.cuda()
            pi = pi.cuda()

        policy,value = self.model(x)
        #print (value.shape,z.shape,policy.shape,pi.shape)
        loss = self.loss(value,z,policy,pi)
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu()[0]#loss.data.cpu() is torch.FloatTensor of shape(1,)

    def value_fn(self,games,many=False):
        if many:
            features_list = []
            ap_list = []
            value_list = []
            rot_list = []
            for game in games:
                features,rots = game.build_features(rot_and_flip=True)
                features_list.append(features.reshape(4, game.height, game.width))
                rot_list.append(rots)

            features = np.array(features_list)
            policy, value = self.predict(features)

            for k in range(len(games)):

                probs_matrix = policy[k].reshape(game.height,game.width)
                if rot_list[k][1] > 0.5:
                    probs_matrix = np.flip(probs_matrix, axis=-1)
                probs_matrix=np.rot90(probs_matrix,-rot_list[k][0])
                probs = probs_matrix.reshape(-1)[games[k].legal_positions.reshape(-1)]

                lp = games[k].POS_COORD[games[k].legal_positions.reshape(-1)]
                cur_player = games[k].cur_player
                ap_list.append([((cur_player, lp[i][0], lp[i][1]), probs[i]) for i in range(len(lp))])
                value_list.append(value[k][0])
            return ap_list, value_list

        else:
            game = games
            features = game.build_features(rot_and_flip=False).reshape(-1,4,game.height,game.width)
            policy, value = self.predict(features)

            probs = policy[0][game.legal_positions.reshape(-1)]
            lp = game.POS_COORD[game.legal_positions.reshape(-1)]

            return [((game.cur_player,lp[i][0],lp[i][1]),probs[i]) for i in range(len(lp))],\
                   value[0][0]

    def save2file(self,fn,max_to_keep):
        dir = os.path.dirname(fn)
        model_files = sorted(glob.glob(os.path.join(dir,"*.pkl")))
        print ("models found:",model_files)
        print ("To save:",fn)
        if len(model_files) > max_to_keep:
            os.remove(model_files[0])
        torch.save(self.model.state_dict(), fn)

    def load_file(self,fn):
        if self.use_cuda is False:
            self.model.load_state_dict(torch.load(fn,map_location=lambda storage, loc: storage))
        else:
            self.model.load_state_dict(torch.load(fn))
        print ('model {} loaded'.format(fn))


if __name__ == '__main__':
    x = Variable(torch.from_numpy(np.random.random((32,4,7,7)).astype(dtype=np.float32)))
    model = PoliycValueNet(7,7,4)
    controller = Controller(model,0.001)
    r = model(x)

    controller.predict(x.data.numpy())

    x = np.random.randint(0,2,(32,4,9,9)).astype(np.float32)
    z = np.random.randint(-1,2,(32,1)).astype(np.float32)
    pi = np.random.random((32,9*9)).astype(np.float32)

    controller.predict(x)
    controller.train_on_batch(x,z,pi)[0]
