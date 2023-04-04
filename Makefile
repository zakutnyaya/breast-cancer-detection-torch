IMAGE={breast-cancer-detection}

build:
	DOCKER_BUILDKIT=1 docker build .

upload:
	docker push ${IMAGE}

run:
	docker run --rm -it -p 8050:8050 -t ${IMAGE}
