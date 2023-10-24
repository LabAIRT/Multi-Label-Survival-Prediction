import torch
from torch import nn
from torchmtlr import MTLR

#**Update
#Number of clin_var
n_clin_var = 16

def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )

def flatten_layers(arr):
    return [i for sub in arr for i in sub]



class Dual_MTLR(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.cnn = nn.Sequential(#block 1
                                 conv_3d_block(hparams['in_channels'], 32, kernel_size=hparams['k1']),
                                 conv_3d_block(32, 64, kernel_size=hparams['k2']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),

                                 #block 2
                                 conv_3d_block(64, 128, kernel_size=hparams['k1']),
                                 conv_3d_block(128, 256, kernel_size=hparams['k2']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),

                                 nn.AdaptiveAvgPool3d(1),

                                 nn.Flatten()                               

                            )

        if hparams['n_dense'] <=0:
            
            self.fc_layers1 = nn.Dropout(hparams['dropout']) 
            
            self.mtlr1 = MTLR(hparams['hidden_size'] + n_clin_var, hparams['time_bins'])
            self.mtlr2 = MTLR(hparams['hidden_size'] + n_clin_var, hparams['time_bins'])
            self.mtlr3 = MTLR(hparams['hidden_size'] + n_clin_var, hparams['time_bins'])
            self.mtlr4 = MTLR(hparams['hidden_size'] + n_clin_var, hparams['time_bins'])
            self.mtlr5 = MTLR(hparams['hidden_size'] + n_clin_var, hparams['time_bins'])

        elif hparams['n_dense'] ==1:
            self.fc_layers1 = nn.Sequential(nn.Linear(hparams['hidden_size'] + n_clin_var, 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr5 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            
        elif hparams['n_dense'] > 1:    
            self.fc_layers1 = nn.Sequential(nn.Linear(hparams['hidden_size'] + n_clin_var , 128*hparams['dense_factor']), 
                          nn.BatchNorm1d(128*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']),
                          nn.Linear(128*hparams['dense_factor'] , 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr5 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])


    def forward(self, sample):
        
        sample_img, clin_var = sample
        img = torch.cat((sample_img['target_mask'][:,0:1,:], sample_img['input'][:,0:1,:]), dim=1) # concate CT and GTVp_Mask
        cnn = self.cnn(img)

        ftr_concat = torch.cat((cnn, clin_var), dim=1)

        x = self.fc_layers1(ftr_concat)

        risk_out1 = self.mtlr1(x)
        risk_out2 = self.mtlr2(x)
        risk_out3 = self.mtlr3(x)
        risk_out4 = self.mtlr4(x)
        risk_out5 = self.mtlr5(x)
        
        return risk_out1,  risk_out2, risk_out3, risk_out4, risk_out5


"""
Inspired from the work of
Credits:
@article{
  kim_deep-cr_2020,
	title = {Deep-{CR} {MTLR}: a {Multi}-{Modal} {Approach} for {Cancer} {Survival} {Prediction} with {Competing} {Risks}},
	shorttitle = {Deep-{CR} {MTLR}},
	url = {https://arxiv.org/abs/2012.05765v1},
	language = {en},
	urldate = {2021-03-16},
	author = {Kim, Sejin and Kazmierski, Michal and Haibe-Kains, Benjamin},
	month = dec,
	year = {2020}
}
"""