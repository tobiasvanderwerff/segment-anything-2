Summary: I can increase the batch size by at least 4x by mitigating memory spikes from a a few layers where there is a very large scaled dot product attention op. However, unfortunately, increasing the batch size does not lead to higher FPS (see below.

## TODO
- x check if compile still has memory spike
	- requires sm_80 or higher :(
- try on video? bigger batches could perhaps help there
	- but doesn't video use sequential frames, i.e. process 1-by-1?


same_hiera_l
CUDA time (using torch.cuda.Event):
	before: 
		- 1.24 FPS (bsz 4)
	after: 
		- 1.22 FPS (bsz 8)
		- 1.22 (bsz 16)
wall-clock time:
	before: 
		- 1.25 FPS (bsz 1)
		- 1.24 FPS (bsz 4)
	after: 
		- 1.22 FPS (bsz 16)
doing the attention in blocks of 4 also does not help (1.23 FPS)


same_hiera_b+
bsz 1: 2.75 FPS
bsz 2: 2.76 FPS
bsz 4: 2.71

same_hiera_s
bsz 1: 4.84
bsz 4: 4.72

## Debug log

`hieradet.py`:
- q_pool_blocks == [2, 8, 44]
- window_size==0 for i==23
