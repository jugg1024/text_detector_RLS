# text_detector_RLS
Algorithm of scene text detection

1.Description: 
this program show a demo of text detection using the Algorithm in SCENE TEXT DETECTION WITH EXTREMAL REGION BASED CASCADED FILTERING

2.Needed Matirials (important):
(1)Models
first download trained models from  :
there are five files in the zip file.
broad_pca_data_Dec_06             pca_data of broad type
narrow_pca_data_Dec_06            pca_data of narrow type
broad_svmStruct_Dec_06            broad SVM
narrow_svmStruct_Dec_06           narrow SVM
detnet_layers                     cnn model
put them on :  3rd_party_tools\


(2) Matlab platform is needed. the code tests on matlab2014b.

(3) You need a windows7 x64 system with CUDA6.5 or a linux system with CUDA7.0.
 if you have other version of CUDA on either system, you need to compile the Matconvnet and get your own mex file, because the off-the-shelf model in [1] is used, for more information, see the use of off-the-shelf model in https://bitbucket.org/jaderberg/eccv2014_textspotting/overview

(3) data
you can use the img_1.jpg available or your own data , you need to change the path 

3. Use
 run single_img_pipeline.m   

4. Contact
Gen.Li  ligen2014@ia.ac.cn    
 

[1] M.Jaderberg, A.Vedaldi, and A.Zisserman, “Deep features for text spotting,” in Proceeding of ECCV, 2014, pp. 512–528.