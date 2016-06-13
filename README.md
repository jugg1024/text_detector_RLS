## text_detector_RLS <br>
Algorithm of Scene Text Detection <br>

****
###　　　　　　　 　　　　　Author:Gen.Li
###　　　　　　　　 　 E-mail:826987854@qq.com
===========================

####1.Description:   
this program show a demo of text detection using the Algorithm in    
`SCENE TEXT DETECTION WITH EXTREMAL REGION BASED CASCADED FILTERING` <br>

####2.Needed Materials (important): <br>
(1)Models <br>
Models are on :  `3rd_party_tools\` <br>
`broad_pca_data_Dec_06`             `pca_data of broad type` <br>
`narrow_pca_data_Dec_06`            `pca_data of narrow type` <br>
`broad_svmStruct_Dec_06`            `broad SVM` <br>
`narrow_svmStruct_Dec_06`           `narrow SVM` <br>
`detnet_layers`                     `cnn model` <br>
 

(2) Matlab platform is needed. the code tests on `Matlab 2014b`. <br>

(3) You need a `windows7 x64` system with `CUDA 6.5` or a `linux` system with `CUDA7.0`. <br>
 if you have other version of CUDA on either system, you need to compile the `Matconvnet` and get your own mex file, because the off-the-shelf model in `[1]` is used, for more information, see the use of model in [Deep features for text spotting](https://bitbucket.org/jaderberg/eccv2014_textspotting/overview "ECCV2014")  <br>

(3) data <br>
you can use `img_1.jpg` available or your own data , you need to change the path in `single_img_pipeline.m` <br>

####3. Use <br>
 run `single_img_pipeline.m`   <br>

[1] M.Jaderberg, A.Vedaldi, and A.Zisserman, “Deep features for text spotting” in Proceeding of ECCV, 2014, pp. 512–528.
