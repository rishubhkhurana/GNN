import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.data.sampler import NeighborSampler
import torch_geometric
from torch_geometric.nn import GATConv,BatchNorm,JumpingKnowledge
from torch import nn,optim
from torch.nn import Softmax
import seaborn as sns
import matplotlib.pyplot as plt

def analysis(data,split_idx:dict,name:str):
    if name=='ogbn-arxiv' or name=='ogbn-products':
        print(f'output analysis {name}')
        for i in ['train','valid','test']:
            figure=plt.figure(figsize=(10,3))
            sns.countplot(x=data.y[split_idx[i]].squeeze(1).numpy())
            plt.title(f'{i} y')
            plt.show()
    return data

def preprocess(data,split_idx:dict,name:str):
    '''
    Normalize node features.
    preprocess dataset based on name.
    --args--
    data=PygNodePropPredDataset(name)[0]
    name: name of ogbn dataset.
    '''
    print(f'preprocessing {name}')
    if name=='ogbn-arxiv':
        #add directed edge in other direction
        #add edge attribute
        print('adding edges in other direction')
        print('normalizing')
    elif name=='ogbn-products':
        print('normalizing')
    return data

def create_dataloder(data,split_idx:dict,sizes=[-1,-1,-1],batch_size=2048)->dict:
    '''
    return train,test and valid dataloaders in a dict.
    --args--
    data=PygNodePropPredDataset(name)[0]
    split_idx: is a dictonary with train,valid and test ids. output of get_idx_split
    sizes: len(sizes)=number of GNN layers. -1 implies no downsampling at that layer.
    '''
    loader_dict={}
    train_batchsize=batch_size
    for i in ['train','valid','test']:
        idx=split_idx[i]
        batch_size=train_batchsize if i=='train' else 2*train_batchsize
        shuffle=True if i=='train' else False
        loader=NeighborSampler(data.edge_index,node_idx=idx,sizes=sizes, batch_size=batch_size,
                               shuffle=shuffle)
        loader_dict[i]=loader
    return loader_dict

class Gat(nn.Module):
    def __init__(self,inp_dim=3,filters=[16,16,16],heads=[2,2,2],drop=0.1,edge_drop=0.1,bn=True,
                 skip=True,jump=True,jk_mode='lstm'):
        '''
        Gat with edge drop,skip connection and jumping knowledge connections.
        --args--
        skip: if True. skip connection will be present. skip connections at a layer(i) can be
            identity if input and output dims match.
        jump: if True. connections are created from every layer to output layer.
        jk_mode: 'lstm' or 'max'. refer "Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`
        '''
        super().__init__()#all params are added to _modules internally, this makes sure its initialized.
        assert len(heads)==len(filters)
        self.gat_modules=nn.ModuleList()
        self.bn=bn
        self.skip=skip
        self.jump=jump
        if self.bn:
            self.bn_modules=nn.ModuleList()
        if self.skip:
            self.skip_conns=nn.ModuleList()
        if self.jump:
            self.jump_lyr=JumpingKnowledge(jk_mode,filters[-1],len(filters)+1)
            self.proj_lyrs=nn.ModuleList()#will be used to project all layers, including input, to final out dim.
            if inp_dim!=filters[-1]:
                self.proj_lyrs.append(nn.Linear(inp_dim,filters[-1]))
            else:
                self.proj_lyrs.append(nn.Identity())

        for i,j in enumerate(filters):
            if i==0:
                self.gat_modules.append(GATConv(in_channels=inp_dim,out_channels=filters[i],heads=heads[i],concat=False,dropout=edge_drop))
            else:
                self.gat_modules.append(GATConv(in_channels=filters[i-1],out_channels=filters[i],heads=heads[i],concat=False,dropout=edge_drop))
            
            if self.bn:
                bn_dim=filters[i]
                self.bn_modules.append(BatchNorm(in_channels=bn_dim))
            if self.skip:
                skip_in_dim=inp_dim if i==0 else filters[i-1]
                if skip_in_dim!=filters[i]:# y=GatConv(x)+W*x. W*x tranforms x to the same dimension as GatConv(x)
                    self.skip_conns.append(nn.Linear(in_features=skip_in_dim,out_features=filters[i],bias=False))
                else:# y=GatConv(x)+x
                    self.skip_conns.append(nn.Identity())         
            if self.jump:
                if filters[i]!=filters[-1]:#if current layer out dim!= final layer out dim, then a projection is needed.
                    self.proj_lyrs.append(nn.Linear(filters[i],filters[-1]))
                else:
                    self.proj_lyrs.append(nn.Identity())
                
        self.leaky=nn.LeakyReLU()
        self.drop=nn.Dropout(p=edge_drop)
        
    def forward(self,x,adjs):
        
        if self.jump:
            jk_conns=[]
            jk_conns.append(self.proj_lyrs[0](x[:adjs[-1].size[1]]))
        
        for i,adj in enumerate(adjs):
            if self.skip:
                prev_x=x
            x=self.gat_modules[i](x,adj.edge_index.to(device=x.device))
            if self.skip:
                x+=self.skip_conns[i](prev_x)
            x=x[:adj.size[1]]
            
            if self.bn:
                x=self.bn_modules[i](x)
            x=self.drop(x)
            x=self.leaky(x)
            if self.jump:
                jk_conns+=[self.proj_lyrs[i+1](x[:adjs[-1].size[1]])]
        
        if self.jump:
            x=self.jump_lyr(jk_conns)
            
        return x

class Fcn(nn.Module):
    def __init__(self,inp_dim=16,layers=[8,40],bn=True,drop=0.1):
        '''
        A fully connected network. Last layer will not have a non linearity.
        '''
        super().__init__()
        self.lyrs=[]
        for i,j in enumerate(layers):
            if i==0:
                self.lyrs.append(nn.Linear(inp_dim,layers[i]))
            else:
                self.lyrs.append(nn.Linear(layers[i-1],layers[i]))
            if i!=len(layers)-1:
                self.lyrs.append(nn.LeakyReLU())
                self.lyrs.append(nn.BatchNorm1d(layers[i]))
                self.lyrs.append(nn.Dropout(p=drop))
        self.lyrs=nn.Sequential(*self.lyrs)#pass the list to Sequential.   
    def forward(self,x):
        return self.lyrs(x)    

class arxiv_classifier(nn.Module):
    def __init__(self,gnn_inp_dim=3,gnn_filters=[16,16,16],gnn_heads=[3,3,3],gnn_drop=0.1,gnn_edge_drop=0.1,gnn_bn=True
                ,fc_layers=[8,40],fc_bn=True,fc_drop=0.1):
        '''
        This model applies the Gat followed by the Fcn.
        '''
        super().__init__()
        self.gnn=Gat(inp_dim=gnn_inp_dim,filters=gnn_filters,heads=gnn_heads,drop=gnn_drop,
                     edge_drop=gnn_edge_drop,bn=gnn_bn)
        self.fcn=Fcn(inp_dim=gnn_filters[-1],layers=fc_layers,bn=fc_bn,drop=fc_drop)
    def forward(self,x,adjs):
        gnn_out=self.gnn(x,adjs)
        return self.fcn(gnn_out)

def accuracy(pred:torch.tensor,truth:torch.tensor):
    return sum(pred==truth).item()/len(truth)

def one_epoch(data:PygNodePropPredDataset,loader:NeighborSampler,model:nn.Module,
              loss_func,opt,sched=None,eval_func=accuracy,train=True):
    '''
    Runs trough all batches in one epoch. Returns avg of eval_func across all batches.
    --args--
    data: PygNodePropPredDataset()[0]
    loader: graph dataloader. returns a subgraph in each iteration
    sched: a scheduler from optim.lr_scheduler
    train: if train is False, gradients are not computed and model.eval is called.  
    '''
    sum_eval=0
    datapoints=0
    if train:
        model.train()
    else:
        model.eval()
        
    for size,n_id,adjs in loader:
        inp=data.x[n_id].to(device=device)
        pred=model(inp,adjs)
        truth=data.y[n_id[:size]].to(device=device).squeeze(dim=1)
        loss=loss_func(pred,truth)
        if train:
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
        sum_eval+=accuracy(torch.argmax(pred,dim=1),truth)*size#.item converts a 0d tensor to a python number 
        datapoints+=size
    return sum_eval/datapoints

if __name__=='__main__':

	#create a folder called node_dataset and all datasets will be downloaded there, if they don't exist.
	dataset=PygNodePropPredDataset(name='ogbn-arxiv',root='../node_dataset/')
	split_idx = dataset.get_idx_split()#a dictonary with train,valid and test ids
	data = dataset[0]

	batch_size=4096
	epochs=25
	model_savename='model_gat_jk.pt'

	loader=create_dataloder(data,split_idx,batch_size=batch_size,sizes=[-1,-1,-1,-1])
	device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model=arxiv_classifier(gnn_inp_dim=128,gnn_filters=[32,32,32,32],gnn_heads=[4,4,4,4])
	model=model.to(device=device)
	opt=optim.Adam(model.parameters(),lr=0.02)
	scheduler=optim.lr_scheduler.CyclicLR(optimizer=opt,base_lr=0.01,max_lr=0.05,step_size_up=len(loader['train'])/2,cycle_momentum=False)
	closs=nn.CrossEntropyLoss()

	best_accuracy=0.0
	for i in range(epochs):
	    model.train()
	    train_loss=one_epoch(data,loader['train'],model,closs,opt,sched=scheduler)
	    model.eval()
	    with torch.no_grad():
	        val_loss=one_epoch(data,loader['valid'],model,closs,opt,train=False)
	    if val_loss>best_accuracy:
	        best_accuracy=val_loss
	        print(f'Epoch {i}. Current best accuracy: {best_accuracy}')
	        torch.save(model.state_dict(),model_savename)

