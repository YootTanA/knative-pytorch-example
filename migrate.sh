dirs=$(ls  -l | grep -v resnet50 | grep -v kustomize | grep '^d' | awk '{print $9}')

for d in $dirs; do 
  echo "-----"
  echo $d
  cp -r resnet50/* $d/

  echo "-----"
done;
