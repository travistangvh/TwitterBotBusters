import torch.nn as nn
import torch
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
import torch.nn.functional as F

class BotGAT_MLP_Skip(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT_MLP_Skip, self).__init__()

        ### MLP ###
        self.mlp_before = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        ### GAT ###

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        ### MLP ###
        self.mlp_after = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.mlp_before(x)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        y = x
        x = self.dropout(x)
        x = self.gat2(x + y, edge_index)
        x = self.linear_relu_output1(x)
        y = x
        x = self.mlp_after(x + y)
        x = self.linear_output2(x)
        return x

class BotGAT_MLP(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT_MLP, self).__init__()
        ### MLP ###
        self.mlp_before = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        ### GAT ###

        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        ### MLP ###
        self.mlp_after = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.mlp_before(x)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.mlp_after(x)
        x = self.linear_output2(x)
        return x

class BotGAT(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x

class BotGAT_GCN_Ensemble(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT_GCN_Ensemble, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        ### GCN ###
        self.linear_relu_des_gcn = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet_gcn = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop_gcn = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop_gcn = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input_gcn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1_gcn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.gcn1_gcn = GCNConv(hidden_dim, hidden_dim)
        self.gcn2_gcn = GCNConv(hidden_dim, hidden_dim)
        self.dropout_gcn = nn.Dropout(p=dropout)

        self.linear_ensemble = nn.Linear(hidden_dim, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)

        ### GCN ###
        d_gcn = self.linear_relu_des_gcn(des)
        t_gcn = self.linear_relu_tweet_gcn(tweet)
        n_gcn = self.linear_relu_num_prop_gcn(num_prop)
        c_gcn = self.linear_relu_cat_prop_gcn(cat_prop)
        x_gcn = torch.cat((d_gcn, t_gcn, n_gcn, c_gcn), dim=1)
        x_gcn = self.dropout_gcn(x_gcn)
        x_gcn = self.linear_relu_input_gcn(x_gcn)
        x_gcn = self.gcn1_gcn(x_gcn, edge_index)
        x_gcn = self.dropout_gcn(x_gcn)
        x_gcn = self.gcn2_gcn(x_gcn, edge_index)
        x_gcn = self.linear_relu_output1_gcn(x_gcn)

        stack = torch.cat((x, x_gcn), dim=0)
        x = self.linear_ensemble(stack)
        return x


class BotGCN(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3, skip_connection=False):
        super(BotGCN, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU()
        )
        self.linear_relu_output2 = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.LeakyReLU()
        )
        self.linear_relu_output3 = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU()
        )

        self.linear_skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_skip2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)

        self.linear_relu_output4 = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim//8, 2)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim//2, hidden_dim//2)
        self.dropout = nn.Dropout(p=dropout)

        self.skip_connection = skip_connection
      

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)

        if self.skip_connection:
            skip_connection_1 = self.linear_skip1(x)
        x = self.linear_relu_input(x)
        x = self.gcn1(x, edge_index)
        if self.skip_connection: 
            x = F.leaky_relu(x)
        x = self.dropout(x)
        if self.skip_connection: 
            x += skip_connection_1  # Adding skip connection

        x = self.gcn2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.dropout(x)

        if self.skip_connection:
            skip_connection_2 = self.linear_skip2(x)
        x = self.gcn3(x, edge_index)
        if self.skip_connection:
            x = F.leaky_relu(x)
        x = self.dropout(x)
        if self.skip_connection:
            x += skip_connection_2

        x = self.linear_relu_output2(x)
        x = self.linear_relu_output3(x)
        x = self.linear_relu_output4(x)
        x = self.linear_output2(x)
        return x



class BotRGCN(nn.Module):
    def __init__(self, hidden_dim,
                 des_size=768,
                 tweet_size=768,
                 num_prop_size=5,
                 cat_prop_size=3,
                 dropout=0.3,
                 num_relations=2):
        super(BotRGCN, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x