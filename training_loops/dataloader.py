from .regimports import *

class DGLBatch:
    def __init__(self,graph=None,node_idxs=None,features=None,yb=None):
        self.graph = graph
        self.features = features
        self.node_idxs = node_idxs
        self.yb=yb

class DGLNodeLoader:
    def __init__(self,graph,node_idxs,shuffle=True,batch_size=512):
        self.graph = graph
        self.node_idxs=node_idxs
        self.shuffle,self.batch_size = shuffle,batch_size
        
    def __len__(self):
        return len(self.node_idxs)
    def __iter__(self):
        # get all the nodes
        N = len(self)
        # create a list of nodes
        if self.shuffle:
            idxs = torch.randperm(N)
        else:
            idxs = torch.arange(N)
        for idx in range(0,N,self.batch_size):
            sampled_idxs = self.node_idxs[idxs[idx:(idx+self.batch_size)]]
            yield DGLBatch(graph=self.graph,features=self.graph.ndata['feat'],
                        node_idxs=sampled_idxs,yb=self.graph.ndata['labels'])
