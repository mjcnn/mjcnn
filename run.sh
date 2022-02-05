MAVEN_OPTS="-Dcom.amd.aparapi.enableShowGeneratedOpenCL=true" mvn exec:java -Dexec.mainClass=apps.Classifier  -Dexec.args="/Users/msilaghi/vgg16_proto.json /Users/msilaghi/vgg16_weights_sq2.json"
