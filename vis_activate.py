import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from glob import glob
import argparse


plt.style.use('seaborn-v0_8')

#フォント設定
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
#plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 50 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます


#軸設定
plt.rcParams['xtick.direction'] = 'in' #x軸の目盛りの向き
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛りの向き
#plt.rcParams['axes.grid'] = True # グリッドの作成
#plt.rcParams['grid.linestyle']='--' #グリッドの線種
#plt.rcParams["xtick.minor.visible"] = True  #x軸補助目盛りの追加
#plt.rcParams["ytick.minor.visible"] = True  #y軸補助目盛りの追加
#plt.rcParams['xtick.top'] = True  #x軸の上部目盛り
#plt.rcParams['ytick.right'] = True  #y軸の右部目盛り


#軸大きさ
plt.rcParams["xtick.major.width"] = 1.0             #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.0             #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0             #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0             #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10               #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ#
plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
plt.rcParams["axes.linewidth"] = 1.0                #囲みの太さ


#凡例設定
plt.rcParams["legend.fancybox"] = False  # 丸角OFF
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
plt.rcParams["legend.markerscale"] = 20 #markerサイズの倍率

from model import OnlyMLP
from matplotlib import rc
import matplotlib.animation as animation
import io
import cv2

def show_animiation(view_data, fname="test"):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # imgs = view_data.squeeze(1)  # チャネル次元を消す
    imgs = view_data
    # imgs = imgs.permute(0,2,3,1) # Permuting to (Bx)HxWxC format
    frames = [
        [ax.imshow(imgs[i], "gray"), ax.text( 0, 1, "epoch:{}".format(i*2000))] for i in range(len(imgs))
    ]
    # plt.xticks([])  # x軸の軸をなくす 
    # plt.yticks([])  # y軸の軸をなくす
    ani = animation.ArtistAnimation(fig, artists=frames,  interval=150, blit=True)
    ani.save("{}.gif".format(fname), writer="pillow")
    plt.close()
    return ani


class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-3 
        self.weight_decay =  0.0
        self.p=67 
        self.d_emb = 500
        self.d_model =  48
        self.is_symmetric_input = True

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 1


        self.num_layers = 0
        self.batch_style = 'full' # ['full', 'random'] 
        self.d_vocab = self.p
        self.n_ctx = 2
        self.d_mlp = 1*self.d_model
        self.num_heads = 1
        assert self.d_model % self.num_heads == 0
        self.d_head = self.d_model//self.num_heads  
        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 1 #0.5576312536233431
        self.prune_rate = 0.4                               
        self.weight_ratio = -1#0.6493382079831002 #0.4152939027995708

        self.use_ln = False

        self.random_answers = np.random.randint(low=0, high=self.p, size=(self.p, self.p))

        self.fns_dict = {'add': lambda x,y:(x+y)%self.p, 'subtract': lambda x,y:(x-y)%self.p, \
                         'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p, 'rand':lambda x,y:self.random_answers[x][y],\
                          "only_add": lambda x,y:(x+y)}
        
        
        # pruning
        self.pruner = "mag" # ["rand", "mag", "snip", "grasp", "synflow"]
        self.sparsity = 0.4 #0.7 #0.29#0.4#0.598#1#0.3
        self.schedule = "linear" # ["linear", "exponential"]
        self.scope = "global" # ["global", "local"]
        self.epochs =  1           
        self.reinitialize =  True
        self.train_mode = False
        self.shuffle = False
        self.invert = False
        if self.is_symmetric_input:
            self.batch_size =  (self.p**2 - self.p)//2
        else:
            self.batch_size =  self.p**2

def cross_entropy(labels,logits,num_neuron=48):
    mean = 0
    cnt = 0
    epsilon = 1e-8
    for i in range(num_neuron):
        mean += np.sum(- labels[i] * np.log(logits[i]+epsilon))
        cnt += 1
    mean /= cnt
    return mean

def frec(weight_path,config,output_file,output_file2,is_mask=False):
        model = OnlyMLP(num_layers=config.num_layers, d_vocab=config.d_vocab, \
                        d_model=config.d_model, d_emb=config.d_emb, \
                        act_type=config.act_type,  use_ln=config.use_ln, \
                        weight_scale=config.weight_scale)
        print(f"Loading model from {weight_path}")
        model.load_state_dict(torch.load(weight_path)["model"])
        if is_mask:
            W_E = model.state_dict()["embed.W_E"]
            W_inproj = model.state_dict()["inproj.W"]
            W_outproj = model.state_dict()["outproj.W"]
            W_U = model.state_dict()["unembed.W_U"]
            W_E_mask = model.state_dict()["embed.weight_mask"]
            W_inproj_mask = model.state_dict()["inproj.weight_mask"]
            W_outproj_mask = model.state_dict()["outproj.weight_mask"]
            W_U_mask = model.state_dict()["unembed.weight_mask"]
            W_in =  (W_inproj*W_inproj_mask) @ (W_E*W_E_mask)
            num_neuron, d_in = W_in.shape
            W_out = (W_outproj*W_outproj_mask).T @ (W_U*W_U_mask)
        else:
            W_E = model.state_dict()["embed.W_E"]
            W_inproj = model.state_dict()["inproj.W"]
            W_outproj = model.state_dict()["outproj.W"]
            W_U = model.state_dict()["unembed.W_U"]
            W_in =  W_inproj @ W_E
            num_neuron, d_in = W_in.shape
            W_out = W_outproj.T @ W_U
        max_val = W_in.max()
        min_val = W_in.min()
        #max_val = W_out.max()
        #min_val = W_out.min()
        fig,axis = plt.subplots(8,6,figsize=(24,18))
        for i_neuron in range(num_neuron):
            axis[i_neuron%8][i_neuron//8].set_ylim(min_val,max_val)
            axis[i_neuron%8][i_neuron//8].plot(W_in[i_neuron], label=str(i_neuron))#plot(W_out[i_neuron], label=str(i_neuron))
            #axis[i_neuron%8][i_neuron//8].axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        plt.savefig(buf, format="png", dpi=180)  # matplotlibから出力される画像のバイナリデータをメモリに格納する.
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする)
        img = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.

        plt.close()
        f_s = 67 # サンプリングレート f_s[Hz] (任意)
        t_fin = 1 # 収録終了時刻 [s] (任意)
        dt = 1/f_s # サンプリング周期 dt[s]
        N = int(f_s * t_fin) # サンプル数 [個]
        fig,axis = plt.subplots(8,6,figsize=(24,18))
        base_init = []
        for i_neuron in range(num_neuron):      
            y = W_in[i_neuron]#W_out[i_neuron]#W_out[i_neuron]  #W_in[i_neuron]#W_out[i_neuron]  
            y_fft = np.fft.fft(y) # 離散フーリエ変換
            freq = np.fft.fftfreq(N, d=dt) # 周波数を割り当てる（※後述）
            Amp = abs(y_fft/(N/2)) # 音の大きさ（振幅の大きさ）w
            axis[i_neuron%8][i_neuron//8].set_ylim(-0.1,max_val)
            axis[i_neuron%8][i_neuron//8].plot(freq[1:int(N/2)], Amp[1:int(N/2)]) # A-f グラフのプロット
            #base_init.append(nn.Softmax(torch.from_numpy(Amp[1:int(N/2)]))
            if (Amp[1:int(N/2)]).sum() > 0:
                base_init.append((Amp[1:int(N/2)])/(Amp[1:int(N/2)]).sum())
        plt.savefig(output_file2, bbox_inches='tight', pad_inches=0.05)
        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        fig.savefig(buf, format="png", dpi=180)  # matplotlibから出力される画像のバイナリデータをメモリに格納する.
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする
        img2 = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.

        plt.close()
        return base_init, img, img2
    
def main(grok_weight_path=None,ticket_folder=None,weight_folder=None,output_folder=None):
    weight_paths = glob(os.path.join(weight_folder,"*.pth"))
    tickets_weight_paths = glob(os.path.join(ticket_folder,"*.pth"))
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    os.makedirs(output_folder, exist_ok=True)
    config = Exp()
    grok_base_init,_,_ = frec(grok_weight_path,config,\
                    os.path.join(output_folder,"grok.png"),os.path.join(output_folder,"grok_frec.png"))
    i = 0
    epochs = []
    base_losses = []
    ticket_losses = []
    base_img1s = []
    base_img2s = []
    ticket_img1s = []
    ticket_img2s = []

    epoch = 0
    for (weight_path,ticket_weight_path) in zip(weight_paths,tickets_weight_paths):
        if "init" in weight_path:
            continue
        print(f"epoch:{epoch}")
        epoch = i * 2000
        base_init, base_img1, base_img2 = frec(weight_path,config,\
                        os.path.join(output_folder,f"base_{epoch}.png"),os.path.join(output_folder,f"base_{epoch}_frec.png"))
        base_img1s.append(base_img1)
        base_img2s.append(base_img2)
        #print(f"base_init:{base_init}")
        ticket_init, ticket_img1, ticket_img2 = frec(ticket_weight_path,config,\
                        os.path.join(output_folder,f"ticket_{epoch}.png"),os.path.join(output_folder,f"ticket_{epoch}_frec.png"),is_mask=True) 
        ticket_img1s.append(ticket_img1)
        ticket_img2s.append(ticket_img2)
        i += 1
        entropy = loss(torch.from_numpy(np.array(base_init)),torch.from_numpy(np.array(base_init)))
        entropy2 = loss(torch.from_numpy(np.array(ticket_init)),torch.from_numpy(np.array(ticket_init)))
        print(f"base_entropy:{entropy}, tikcet_entropy:{entropy2}")
        epochs.append(epoch)
        base_losses.append(entropy.item())
        ticket_losses.append(entropy2.item())
    
        fig = plt.figure(figsize=(10,10))
        plt.plot(epochs,base_losses,label="Base Model")
        plt.plot(epochs,ticket_losses,label="Grokking ticket")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy")
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(output_folder,"loss.png"), bbox_inches='tight', pad_inches=0.05)
        plt.close()
    show_animiation(base_img1s,os.path.join(output_folder,"base"))
    show_animiation(base_img2s,os.path.join(output_folder,"base_frec"))
    show_animiation(ticket_img1s,os.path.join(output_folder,"ticket"))
    show_animiation(ticket_img2s,os.path.join(output_folder,"ticket_frec"))
        
if __name__ == "__main__":  
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--grok_weight_path", type=str, default=None)
    argparse.add_argument("--weight_folder", type=str, default=None)
    argparse.add_argument("--ticket_folder", type=str, default=None)
    argparse.add_argument("--output_folder", type=str, default=None)

    args = argparse.parse_args()
    grok_weight_path = args.grok_weight_path
    weight_folder = args.weight_folder
    ticket_folder = args.ticket_folder
    output_folder = args.output_folder
    main(grok_weight_path=grok_weight_path,ticket_folder=ticket_folder,weight_folder=weight_folder,output_folder=output_folder)