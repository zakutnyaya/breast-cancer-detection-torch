IMAGE=zakutniaia/cancer-detection-training

build:
	DOCKER_BUILDKIT=1 docker build -t ${IMAGE} ./

upload:
	docker push ${IMAGE}

run:
	docker run --rm -ti --platform linux/amd64 ${IMAGE}
