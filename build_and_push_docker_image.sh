TAG="$1"
USERNAME='mmaaz60'
#!/bin/sh

docker build -t $USERNAME/mdetr:"$TAG" .
docker push $USERNAME/mdetr:"$TAG"
