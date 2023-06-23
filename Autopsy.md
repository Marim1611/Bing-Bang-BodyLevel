
<h1 align="center"> ⚰️ Stacked Generalization Model Autopsy </h2>

Although our stacked generalization model was the most spectacular. This project was also a competition and we were shocked to hear that it got 96.6% only. 

<p align="justify"> 
We went forward with stability analysis on stacking to find that a drop from 99% to 96.6% should be highly unlikely: after knowing the size of the test split we set $nsplit$ accordingly and performated many repetitions of cross-validation so that in the end we have went over 100 different possible instances of the test set. The mean and standard deviation metric were $0.9865$ and $0.005$ and by Chebyshev, a deviation by $k\sigma$ where $k=floor((0.9865-0.966)/0.005)=4$ occurs with at most probability $1/k^2=0.0625$ which is very low.
</p>

![output](https://github.com/Marim1611/Bing-Bang-BodyLevel/assets/49572294/282463b0-40e7-4796-b418-2192583134cb)


<p align="justify"> 
Hence, we requested the test set and started testing it for randomization and stratification relative to the original dataset. All tests passed. We found out that the issue was that the script we have submitted did not use accurate means and stds for standardization as they were unoticingly generated form the validation set only prior to submission. The difference between them and the original means and stds are ridiculously small peaking at a 2% difference but that was sufficient to drive the stacking model insane.
</p>

Under the right means and stds, the model achieves 99% accuracy on the test set and without hacking the project by using BMI features or such.

![image](https://github.com/Marim1611/Bing-Bang-BodyLevel/assets/49572294/4ff3bac0-12de-4b11-ba15-d524dddd11f4)


