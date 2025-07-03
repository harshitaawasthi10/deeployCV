# deeploycv
This is a project I did under the google developers guild during my winter break in the first semester.
We created a GAN model that was used for traffic data imputation. 

The dataset used for this was: https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset

We first measured the traffic volume for various timestamps in the video, and then created a GASF matrix for each array. This was the data on whcih the GAN was trained on after masking 20% of the matrix. 
here is an example of the generated gasf matrix, from the array 

[0.40406090021133423, 0.2020304650068283, 0.2020304650068283, 0.10101523250341415, 0.2020304650068283, 0.10101523250341415, 0.10101523250341415, 0.30304569005966187, 0.0, 0.10101523250341415]

![image](https://github.com/user-attachments/assets/5cb36192-a8c9-4659-9edb-7fdcbb2bfb0c)


here is the generated array post masking

[0.39016711711883545, 0.168606236577034, 0.1482408195734024, 0.08135004341602325, 0.13546916842460632, 0.08532050997018814, 0.08639180660247803, 0.2183794528245926, 0.1401035338640213, 0.14255665242671967]
