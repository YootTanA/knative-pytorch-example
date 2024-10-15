dirs=$(ls  -l | grep -v resnet50 | grep -v kustomize | grep '^d' | awk '{print $9}')

for d in $dirs; do 
  echo "-----"
  echo $d
  find $d/ -type f -exec sed -i "s/resnetImage/resnet50Image/g" {} \;

  echo "-----"
done;
