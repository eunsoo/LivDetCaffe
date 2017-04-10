# python real_Testfile.py -y LivDet2011 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s BiometrikaTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2011 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s DigitalTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2011 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s ItaldataTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2011 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s SagemTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2013 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_130000.caffemodel -s BiometrikaTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2013 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s CrossMatchTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2013 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s ItaldataTrain \
# -r /home/park/data/LIVDETECT/conv5/ 

# No Training
python real_Testfile.py -y LivDet2013 -d deploy_two.prototxt \
-w caffebi/vgg2Origin_iter_100000.caffemodel -s CrossMatchTrain \
-r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2015 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_150000.caffemodel -s CrossMatch \
# -r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2015 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_100000.caffemodel -s Digital_Persona \
# -r /home/park/data/LIVDETECT/conv5/ 


# Image load error
python real_Testfile.py -y LivDet2015 -d deploy_two.prototxt \
-w caffebi/vgg2Origin_iter_120000.caffemodel -s GreenBit \
-r /home/park/data/LIVDETECT/conv5/ 

# python real_Testfile.py -y LivDet2015 -d deploy_two.prototxt \
# -w caffebi/vgg2Origin_iter_120000.caffemodel -s Hi_Scan \
# -r /home/park/data/LIVDETECT/conv5/ 

# No dataset 
python real_Testfile.py -y LivDet2015 -d deploy_two.prototxt \
-w caffebi/vgg2Origin_iter_150000.caffemodel -s Time_Series \
-r /home/park/data/LIVDETECT/conv5/ 