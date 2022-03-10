# parking-slot-detection-with-lightweight-Keypoint-detect
The Model:

We based on MoveNet,use a MoblienetV2 as the backbone,and the same Fpn structure in MoveNet.

We use three detect header,respectively detected the heatmap,the center point of the entrance line and the orientation of the parking slot.

The header is consist of two 1x1 conv and a dilation conv,the dilation conv is used to extend the field.

We modify the soft wing loss as Weighted soft Wing loss,combined with Weighted SEloss to train the model with 200 epoches.

The Parameters of the model is only 0.255M!!

And the Flops of the model is only 456.37MFlops!!


Visualize:

![image](https://user-images.githubusercontent.com/61531491/157662799-3d8935d4-ae07-4f8a-80f4-0bfec716099d.png)
![image](https://user-images.githubusercontent.com/61531491/157663050-a6b96790-c8b9-448a-b72a-d659df829f28.png)
![image](https://user-images.githubusercontent.com/61531491/157663112-d513adc6-28d4-4ce4-892f-96c3fd0ba61d.png)


The Precision on the ps2.0: 97.63%


The code will update soon!!
