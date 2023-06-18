
<h1 align="center"> ⚰️ Stacked Generalization Model Autopsy </h2>

Although our stacked generalization model was the most spectacular. This project was also a competition and we were shocked to hear that it got 96.6% only. 

<p align="justify"> 
We went forward with stability analysis on stacking to find that a drop from 99% to 96.6% should be highly unlikely: after knowing the size of the test split we set $nsplit$ accordingly and performated many repetitions of cross-validation so that in the end we have went over 100 different possible instances of the test set. The mean and standard deviation metric were $0.9865$ and $0.005$ and by Chebyshev, a deviation by $k\sigma$ where $k=floor((0.9865-0.966)/0.005)=4$ occurs with at most probability $1/k^2=0.0625$ which is very low.
</p>

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246648601-2b0aee68-38df-42e7-907f-710faa241e86.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230618%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230618T070252Z&X-Amz-Expires=300&X-Amz-Signature=68ff6e58fe9c63dff6c39a740e43a94582cd655a890d8b2fee07a9fa5b0946dd&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=618807942)

<p align="justify"> 
Hence, we requested the test set and started testing it for randomization and stratification relative to the original dataset. All tests passed. We found out that the issue was that the script we have submitted did not use accurate means and stds for standardization as they were unoticingly generated form the validation set only prior to submission. The difference between them and the original means and stds are ridiculously small peaking at a 2% difference but that was sufficient to drive the stacking model insane.
</p>

Under the right means and stds, the model achieves 99% accuracy on the test set and without hacking the project by using BMI features or such.

![output (2)](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246649162-bd3159e9-1c46-47b5-a155-490f5a61e0d5.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230618%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230618T071920Z&X-Amz-Expires=300&X-Amz-Signature=4ee566115aab10f575c0bb02d217cc308bc6b3b706eb754ec2a9a9adcc3182a9&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=618807942)


