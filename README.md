# MusicECAN: Automatic Recorded Music Denoising Network with Efficient Channel Attention

## Abstract
In this work, we address the long-standing problem of automatic recorded music denoising. Previous audio denoising works focus on speech focused on speech primarily instead of music, neglecting the scenario of amateur music recording. To this end, we first propose MusicECAN, an automatic recorded music denoising network designed to enhance the quality of recorded music. The novel architecture comprises two key components, namely a feature learning module and a noise filtering module, which can model, refine and denoise the noisy input efficiently yet effectively. Specifically, in order to capture sufficient noisy music information, an ECA-U-SAM based feature learning module is designed by introducing an Efficient Channel Attention (ECA) mechanism in traditional U-net with a Supervised Attention Module (SAM). For the training of our MusicECAN, we collect M&N, a dataset containing various recordings of clean music and noise. Through the combination of different clean and noise recording pairs, we can effectively simulate possible environments of music performances with different background noise. Extensive quantitative and qualitative comparisons demonstrate that our MusicECAN outperforms the state-of-the-art audio denoising methods.

<p align="center">
<img src="image/fig5.png" alt="Schema represention"
width="1000px"></p>

## Demo Video
Watch the [demo video](https://www.kaggle.com/datasets/shulinliu/musicecan-demo). 
<p align="center">
<img src="image/github.png" alt="Schema represention"
width="600px"></p>

[Listening more denoising examples](https://slliugit.github.io/).

## M&N: A music dataset for denoising music recordings in wild
We introduce a dataset M&N, which can effectively meet the requirements of music denoising for recordings in the wild. The dataset comprises various videos and recordings of clean music and noise assembled from free sound effects website and existing cross-modal audio generation dataset [FAIR-PLAY](https://github.com/facebookresearch/FAIR-Play). For video data, we separate the visual and audio tracks of the video. We anticipate that the dataset will be useful for denoising task and also serve as ground-truth for evaluating performances.

For music data, we collect totally 3.43 hours of clean music recordings in wav format with a sampling rate of 44.1 kHz and bit depth of 16 bits, mono channel. There are 9 categories of music recordings: piano, drum kit, harp, cello, Chinese lute, trumpet, Chinese zither, multi-instrument and song.</p> 

<a href="data/Clean music recording metadata.csv">Download clean music recordings metadata</a>

[Download clean music recordings except from FAIR-PLAY](https://drive.google.com/file/d/1tj794OUdU6WpUZhXcYjQqeG5BrOzwF4y/view?usp=sharing)</a>

It is worth noting that the clean music recordings we provide here are each 10 seconds long to ensure that researchers who need them can crop them to different lengths, such as five seconds, as needed.

For noise data, we collect totally 1000 seconds of noise recordings in wav format with a sampling rate of 44.1 kHz. According to audio content, the noise data is divided into five categories: 

* Electrical noise: Recordings of electrical circuit noise such as clicking, hissing noise and crackling noise caused by the irregularities in the storage medium. This kind of noise often occurs when the user is recording music in a relatively quiet room while the device is malfunctioning.

* Crowd noise: Recordings include the sound of crowd chatter, cheering and children's laughter, etc. Such noise often exists when users record live music in crowded venues such as shopping malls, theaters, or plazas.</li>

* Weather noise: Recordings include the sound of rain, wind, thunder and other weather sounds. It is common for these types of noises to be heard when users record music in non-sound-insulating places.

* Traffic noise: Recordings include vehicle start-up sounds, road traffic sounds, motorcycle sounds, etc., which are used to simulate the traffic noise when the user is recording audio near the driveway.

* Stationary noise: Recordings of random noise for which the probability that the noise voltage lies within any given interval does not change with time, such as white noise.</li></p>

<a href="data/Noise recording metadata.csv">Download noise recording recordings metadata</a>

[Download noise recordings](https://drive.google.com/file/d/1trm8f7QXvECjfTmrACwT_I9ox951ZbhL/view?usp=sharing)</a>

<p align="center">
<img src="image/fig3.png" alt="distribution" 
width="1000px"></p>
    
## Code
Coming soon!
