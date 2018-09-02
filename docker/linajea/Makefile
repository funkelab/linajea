TAG="funkey/linajea:v0.1"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	rsync -auv --delete ../../linajea .
	cp -p ../../requirements.txt .
	cp -p ../../setup.py .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
