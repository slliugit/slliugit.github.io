# MusicECAN: Music Denoising Network for Recordings in Wild with Efficient Channel Attention

## Abstract
In this work, we address the long-standing problem of automatic music denoising. Previous audio denoising works mainly focus on speech instead of music, and none of them considers the music recordings in the wild. To this end, we first propose MusicECAN, an automatic music denoising network that enables sound quality enhancement for recordings in the wild. The novel architecture includes two parts, namely feature learning module and noise filtering module, which can model, refine and denoise the noisy input efficiently yet effectively.  pecifically, in order to capture sufficient noisy music information, an ECA-U-SAM based feature learning module is designed by introducing an Efficient Channel Attention (ECA) mechanism in raditional U-net with a Supervised Attention Module (SAM). To train our proposed MusicECAN, we collect M&N, a dataset containing various videos and recordings of clean music and noise. By combining different clean and noise recording pairs, we can effectively simulate possible environments of music performance with different background noise. Various quantitative and qualitative comparisons demonstrate that our MusicECAN outperforms the state-of-the-art audio denoising methods.

<p align="center">
<img src="image/flow.png" alt="Schema represention"
width="1000px"></p>

## Demo Video
Watch the [demo video]([https://youfiles.herokuapp.com/videodictionary/?m=Video_Player_Drive&state=%7B%22ids%22:%5B%221a_orDlQuxdL3N9SXs_wPHapoQd31r59V%22%5D,%22action%22:%22open%22,%22resourceKeys%22:%7B%7D%7D](https://www.kaggle.com/datasets/shulinliu/musicecan-demo)). 
<p align="center">
<img src="image/demo.png" alt="Schema represention"
width="600px"></p>

## Denoising Recordings

You can denoise your recordings in the cloud using the Colab notebook. 
[Listening more test examples samples](http://research.spa.aalto.fi/publications/papers/icassp22-denoising/)


## Code
Coming soon!
