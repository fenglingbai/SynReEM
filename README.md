# SynReEM: Synapse Reconstruction via Instance Structure Encoding in Anisotropic vEM Images

We develop SynReEM, an end-to-end synapse reconstruction framework for anisotropic volume electron microscopy scenarios. 
<br />



We will make the code publicly available as soon as our article is published.


## The basic idea of SynReEM

<div  align="center">    
	<img src="https://github.com/fenglingbai/SynReEM/blob/main/fig/p2_motivation.png" width = "600px" />
</div>

Comparison of SynReEM and traditional detection methods. (a) Detection model utilizes anchors to identify instances and bounding boxes for scope definition, employing NMS for duplicate detections. (b) SynReEM incorporates core regions (depicted as black or colored lines), attention regions (illustrated as semi-transparent red areas), and aggregation algorithms to achieve similar objectives.

## The basic architecture of SynReEM

<div  align="center">    
	<img src="https://github.com/fenglingbai/SynReEM/blob/main/fig/p3_SynReEM.png" width = "600px" />
</div>

SynReEM architecture. We re-encode instance labels according to the synapse structure to obtain AEMC labels (the orange box). The dual-branch head predicts AEMC and semantic labels with the help of shared feature maps. The prediction results are jointly decoded to obtain instance reconstruction (the blue box). To utilize the prior structural constraints of AEMC, we design continuity constraints and inclusion constraints to improve the model's contextual awareness and prediction rationality through online pseudo-labeling (the green box).